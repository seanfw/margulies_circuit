def get_stim(annot_file, label_name, cortex_file):

    ################################
    ##### ROI for Stimulation: #####
    ################################

    import numpy as np
    import nibabel as nib
    from surfdist_functions import *

    # Let's apply external stimulation to V1 populations E1 & E2
    V1 = load_freesurfer_label(annot_file, label_name)
    cortex = np.sort(nib.freesurfer.read_label(cortex_file))
    V1_index = translate_src(V1, cortex)

    # Preassign external inputs
    return V1_index
