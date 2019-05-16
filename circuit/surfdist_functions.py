import nibabel as nib
import numpy as np
import gdist
import numba

def load_freesurfer_label(annot_input, label_name, cortex=None):
    """
    Get source node list for a specified freesurfer label.

    Inputs
    -------
    annot_input : freesurfer annotation label file
    label_name : freesurfer label name
    cortex : not used
    """

    if cortex is not None:
        print("Warning: cortex is not used to load the freesurfer label")

    labels, color_table, names = nib.freesurfer.read_annot(annot_input)
    names = [i.decode('utf-8') for i in names]
    label_value = names.index(label_name)
    label_nodes = np.array(np.where(np.in1d(labels, label_value)), dtype=np.int32)

    return label_nodes


def get_freesurfer_label(annot_input, verbose = True):
    """
    Print freesurfer label names.
    """
    labels, color_table, names = nib.freesurfer.read_annot(annot_input)
    if verbose:
        print(names)
    return names

@numba.jit(parallel=True)
def surf_keep_cortex(surf, cortex):
    """
    Remove medial wall from cortical surface to ensure that shortest paths are only calculated through the cortex.

    Inputs
    -------
    surf : Tuple containing two numpy arrays of shape (n_nodes,3). Each node of the first array specifies the x, y, z
           coordinates one node of the surface mesh. Each node of the second array specifies the indices of the three
           nodes building one triangle of the surface mesh.
           (e.g. the output from nibabel.freesurfer.io.read_geometry)
    cortex : Array with indices of vertices included in within the cortex.
             (e.g. the output from nibabel.freesurfer.io.read_label)
    """

    # split surface into vertices and triangles
    vertices, triangles = surf

    # keep only the vertices within the cortex label
    cortex_vertices = np.array(vertices[cortex], dtype=np.float64)

    # keep only the triangles within the cortex label
    cortex_triangles = triangles_keep_cortex(triangles, cortex)

    return cortex_vertices, cortex_triangles

@numba.jit(parallel=True)
def triangles_keep_cortex(triangles, cortex):
    """
    Remove triangles with nodes not contained in the cortex label array
    """

    # for or each face/triangle keep only those that only contain nodes within the list of cortex nodes
    input_shape = triangles.shape
    triangle_is_in_cortex = np.all(np.reshape(np.in1d(triangles.ravel(), cortex), input_shape), axis=1)

    cortex_triangles_old = np.array(triangles[triangle_is_in_cortex], dtype=np.int32)

    # reassign node index before outputting triangles
    new_index = np.digitize(cortex_triangles_old.ravel(), cortex, right=True)
    cortex_triangles = np.array(np.arange(len(cortex))[new_index].reshape(cortex_triangles_old.shape), dtype=np.int32)

    return cortex_triangles

@numba.jit(parallel=True)
def translate_src(src, cortex):
    """
    Convert source nodes to new surface (without medial wall).
    """
    src_new = np.array(np.where(np.in1d(cortex, src))[0], dtype=np.int32)

    return src_new

def recort(input_data, surf, cortex):
    """
    Return data values to space of full cortex (including medial wall), with medial wall equal to zero.
    """
    data = np.zeros(len(surf[0]))
    data[cortex] = input_data
    return data

@numba.jit(parallel=True)
def find_node_match(simple_vertices, complex_vertices):
    """
    Thanks to juhuntenburg.
    Functions taken from https://github.com/juhuntenburg/brainsurfacescripts

    Finds those points on the complex mesh that correspond best to the
    simple mesh while forcing a one-to-one mapping.
    """

    import scipy.spatial

    # make array for writing in final voronoi seed indices
    voronoi_seed_idx = np.zeros((simple_vertices.shape[0],), dtype='int64')-1
    missing = np.where(voronoi_seed_idx == -1)[0].shape[0]
    mapping_single = np.zeros_like(voronoi_seed_idx)

    neighbours = 0
    col = 0

    while missing != 0:

        neighbours += 100
        # find nearest neighbours
        inaccuracy, mapping = scipy.spatial.KDTree(
            complex_vertices).query(simple_vertices, k=neighbours)
        # go through columns of nearest neighbours until unique mapping is
        # achieved, if not before end of neighbours, extend number of
        # neighbours
        while col < neighbours:
            # find all missing voronoi seed indices
            missing_idx = np.where(voronoi_seed_idx == -1)[0]
            missing = missing_idx.shape[0]
            if missing == 0:
                break
            else:
                # for missing entries fill in next neighbour
                mapping_single[missing_idx] = np.copy(
                    mapping[missing_idx, col])
                # find unique values in mapping_single
                unique, double_idx = np.unique(
                    mapping_single, return_inverse=True)
                # empty voronoi seed index
                voronoi_seed_idx = np.zeros(
                    (simple_vertices.shape[0],), dtype='int64')-1
                # fill voronoi seed idx with unique values
                for u in range(unique.shape[0]):
                    # find the indices of this value in mapping
                    entries = np.where(double_idx == u)[0]
                    # set the first entry to the value
                    voronoi_seed_idx[entries[0]] = unique[u]
                # go to next column
                col += 1

    return voronoi_seed_idx, inaccuracy


def dist_calc(surf, cortex, source_nodes):

    """
    Calculate exact geodesic distance along cortical surface from set of source nodes.
    "dist_type" specifies whether to calculate "min", "mean", "median", or "max" distance values
    from a region-of-interest. If running only on single node, defaults to "min".
    """

    cortex_vertices, cortex_triangles = surf_keep_cortex(surf, cortex)
    translated_source_nodes = translate_src(source_nodes, cortex)
    data = gdist.compute_gdist(cortex_vertices, cortex_triangles, source_indices = translated_source_nodes)
    dist = recort(data, surf, cortex)
    del data

    return dist


def zone_calc(surf, cortex, src):
    """
    Calculate closest nodes to each source node using exact geodesic distance along the cortical surface.
    """

    cortex_vertices, cortex_triangles = surf_keep_cortex(surf, cortex)

    dist_vals = np.zeros((len(source_nodes), len(cortex_vertices)))

    for x in range(len(source_nodes)):

        translated_source_nodes = translate_src(source_nodes[x], cortex)
        dist_vals[x, :] = gdist.compute_gdist(cortex_vertices, cortex_triangles, source_indices = translated_source_nodes)

    data = np.argsort(dist_vals, axis=0)[0, :] + 1

    zone = recort(data, surf, cortex)

    del data

    return zone


def dist_calc_matrix(surf, cortex, labels, exceptions = ['Unknown', 'Medial_wall'], verbose = True):
    """
    Calculate exact geodesic distance along cortical surface from set of source nodes.
    "labels" specifies the freesurfer label file to use. All values will be used other than those
    specified in "exceptions" (default: 'Unknown' and 'Medial_Wall').

    returns:
      dist_mat: symmetrical nxn matrix of minimum distance between pairs of labels
      rois: label names in order of n
    """

    cortex_vertices, cortex_triangles = surf_keep_cortex(surf, cortex)

    # remove exceptions from label list:
    label_list = sd.load.get_freesurfer_label(labels, verbose = False)
    rs = np.where([a not in exceptions for a in label_list])[0]
    rois = [label_list[r] for r in rs]
    if verbose:
        print("# of regions: " + str(len(rois)))

    # calculate distance from each region to all nodes:
    dist_roi = []
    for roi in rois:
        source_nodes = sd.load.load_freesurfer_label(labels, roi)
        translated_source_nodes = translate_src(source_nodes, cortex)
        dist_roi.append(gdist.compute_gdist(cortex_vertices, cortex_triangles,
                                                source_indices = translated_source_nodes))
        if verbose:
            print(roi)
    dist_roi = np.array(dist_roi)

    # Calculate min distance per region:
    dist_mat = []
    for roi in rois:
        source_nodes = sd.load.load_freesurfer_label(labels, roi)
        translated_source_nodes = translate_src(source_nodes, cortex)
        dist_mat.append(np.min(dist_roi[:,translated_source_nodes], axis = 1))
    dist_mat = np.array(dist_mat)

    return dist_mat, rois
