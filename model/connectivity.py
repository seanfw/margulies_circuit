
def get_surface_connectivity(surface, cortex_map):

    import networkx as nx
    import numpy as np

    surf = load_surface(surface, cortex_map)
    num_nodes = len(surf[0])

    g = nx.Graph()
    for x,y,z in surf[1]:
        g.add_edge(x,y)
        g.add_edge(y,z)

    surface_connectivity = np.zeros((num_nodes,num_nodes))
    for k in range(len(g.nodes)):
        for i in [j for j in g.neighbors(k)]:
            surface_connectivity[k, i] = 1

    return surface_connectivity

def load_surface(surface, cortex_map):

    import numpy as np
    import nibabel as nib
    import surfdist
    from surfdist import utils

    surf = nib.freesurfer.read_geometry(surface)
    cortex = np.sort(nib.freesurfer.read_label(cortex_map))
    surf_new = surfdist.utils.surf_keep_cortex(surf, cortex)

    return surf_new

def get_longdistance_connectivity(surface, cortex_map, connmat_file, label_file):

    import numpy as np
    import numpy.matlib
    import nibabel as nib

    surf = load_surface(surface, cortex_map)
    num_nodes = len(surf[0])

    # White matter tract connectivity

    tract_connectivity = np.zeros((num_nodes,num_nodes))

    # load Murray data from Balsa:
    conn_mat = np.asarray(nib.load(connmat_file).get_data())

    # load labels:
    # labels = np.concatenate((nib.load('conn_data/Glasser_NetworkPartition_v9.fsa4.L.label.gii').darrays[0].data,
    #                          nib.load('conn_data/Glasser_NetworkPartition_v9.fsa4.R.label.gii').darrays[0].data))

    # load labels for only left hemisphere:
    cortex = np.sort(nib.freesurfer.read_label(cortex_map))
    labels = nib.load(label_file).darrays[0].data[cortex]

    # normalize so that rows sum to 1:
    conn_mat_rowtotal = np.sum(conn_mat,axis=1)
    conn_rowtotal_mat = np.matlib.repmat(conn_mat_rowtotal, len(conn_mat), 1).T
    conn_mat_norm = conn_mat / conn_rowtotal_mat

    # resample to surface:
    tract_connectivity = conn_mat_norm[labels-1,:][:,labels-1]

    return tract_connectivity

def get_num_nodes(surface, cortex_map):

    surf = load_surface(surface, cortex_map)

    return len(surf[0])
