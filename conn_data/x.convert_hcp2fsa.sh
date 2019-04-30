#!/bin/bash

wb_command -cifti-separate Glasser_NetworkPartition_v9.dlabel.nii COLUMN -label CORTEX_LEFT Glasser_NetworkPartition_v9.L.label.gii  -label CORTEX_RIGHT Glasser_NetworkPartition_v9.R.label.gii

tdir=../standard_mesh_atlases/resample_fsaverage

wb_command -label-resample Glasser_NetworkPartition_v9.L.label.gii ${tdir}/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii ${tdir}/fsaverage6_std_sphere.L.41k_fsavg_L.surf.gii ADAP_BARY_AREA Glasser_NetworkPartition_v9.fsa6.L.label.gii -area-metrics ${tdir}/fs_LR.L.midthickness_va_avg.32k_fs_LR.shape.gii ${tdir}/fsaverage6.L.midthickness_va_avg.41k_fsavg_L.shape.gii

wb_command -label-resample Glasser_NetworkPartition_v9.R.label.gii ${tdir}/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii ${tdir}/fsaverage6_std_sphere.R.41k_fsavg_R.surf.gii ADAP_BARY_AREA Glasser_NetworkPartition_v9.fsa6.R.label.gii -area-metrics ${tdir}/fs_LR.R.midthickness_va_avg.32k_fs_LR.shape.gii ${tdir}/fsaverage6.R.midthickness_va_avg.41k_fsavg_R.shape.gii


# fsaverage4:
wb_command -label-resample Glasser_NetworkPartition_v9.L.label.gii ${tdir}/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii ${tdir}/fsaverage4_std_sphere.L.3k_fsavg_L.surf.gii ADAP_BARY_AREA Glasser_NetworkPartition_v9.fsa4.L.label.gii -area-metrics ${tdir}/fs_LR.L.midthickness_va_avg.32k_fs_LR.shape.gii ${tdir}/fsaverage4.L.midthickness_va_avg.3k_fsavg_L.shape.gii

wb_command -label-resample Glasser_NetworkPartition_v9.R.label.gii ${tdir}/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii ${tdir}/fsaverage4_std_sphere.R.3k_fsavg_R.surf.gii ADAP_BARY_AREA Glasser_NetworkPartition_v9.fsa4.R.label.gii -area-metrics ${tdir}/fs_LR.R.midthickness_va_avg.32k_fs_LR.shape.gii ${tdir}/fsaverage4.R.midthickness_va_avg.3k_fsavg_R.shape.gii




# left:
mris_convert ./fsaverage3/lh.sphere ./fsaverage3/lh.sphere.gii
mris_convert -C /Applications/freesurfer/subjects/fsaverage3/surf/lh.area  /Applications/freesurfer/subjects/fsaverage3/surf/lh.pial fsaverage3/lh.area.gii

wb_command -label-resample Glasser_NetworkPartition_v9.fsa6.L.label.gii ${tdir}/fsaverage6_std_sphere.L.41k_fsavg_L.surf.gii ../fsaverage3/lh.sphere.gii ADAP_BARY_AREA Glasser_NetworkPartition_v9.fsa3.L.label.gii -area-metrics ${tdir}/fsaverage6.L.midthickness_va_avg.41k_fsavg_L.shape.gii ../fsaverage3/lh.area.gii

# right:
mris_convert ./fsaverage3/rh.sphere ./fsaverage3/rh.sphere.gii
mris_convert -C /Applications/freesurfer/subjects/fsaverage3/surf/rh.area  /Applications/freesurfer/subjects/fsaverage3/surf/rh.pial ./fsaverage3/rh.area.gii

wb_command -label-resample Glasser_NetworkPartition_v9.fsa6.R.label.gii ${tdir}/fsaverage6_std_sphere.R.41k_fsavg_R.surf.gii ../fsaverage3/rh.sphere.gii ADAP_BARY_AREA Glasser_NetworkPartition_v9.fsa3.R.label.gii -area-metrics ${tdir}/fsaverage6.R.midthickness_va_avg.41k_fsavg_R.shape.gii ../fsaverage3/rh.area.gii

#
#
# # fsa6 to fsa3:
#
# mris_convert /Applications/freesurfer/subjects/fsaverage6/surf/lh.pial /Applications/freesurfer/subjects/fsaverage6/surf/lh.pial.gii
# mris_convert /Applications/freesurfer/subjects/fsaverage6/surf/rh.pial /Applications/freesurfer/subjects/fsaverage6/surf/rh.pial.gii
#
# mris_convert --annot Glasser_NetworkPartition_v9.fsa6.L.label.gii /Applications/freesurfer/subjects/fsaverage6/surf/lh.pial.gii ./lh.Glasser_NetworkPartition_v9.fsa6.annot
# mris_convert --annot Glasser_NetworkPartition_v9.fsa6.R.label.gii /Applications/freesurfer/subjects/fsaverage6/surf/rh.pial.gii ./rh.Glasser_NetworkPartition_v9.fsa6.annot
#
# mri_surf2surf --srcsubject fsaverage6 --sval-annot ./Glasser_NetworkPartition_v9.fsa6.L.label.gii --trgsubject fsaverage3 --tval ./lh.Glasser_NetworkPartition_v9.fsa3.L.label.gii --hemi lh
# mri_surf2surf --srcsubject fsaverage6 --sval-annot ./rh.Glasser_NetworkPartition_v9.fsa6.annot --trgsubject fsaverage3 --tval ./rh.Glasser_NetworkPartition_v9.fsa3.annot --hemi rh --cortex
