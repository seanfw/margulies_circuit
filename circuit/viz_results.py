import matplotlib.pyplot as plt
import configparser
import numpy as np
import brian2

def plot_surfaces(R)

    from surfdist_functions import *
    from nilearn import plotting
    import nibabel as nib

    config = configparser.ConfigParser()
    config.read('./config.ini')
    c = config['SETUP']

    dt = float(c.get('dt')) * brian2.ms  # timestep
    start_time = 0.0
    end_time = float(c.get('trial_length'))/1000
    tm = np.arange((start_time)*brian2.second,(end_time)*brian2.second,dt)

    # plot surfaces:
    fig, ax = plt.subplots(2,10, figsize=(60,10), subplot_kw={'projection': '3d'})
    surf   = nib.freesurfer.read_geometry(c.get('surf_file_inf'))
    cortex = np.sort(nib.freesurfer.read_label(c.get('cort_file')))
    t = np.arange(int(start_time/dt),int(end_time/dt))
    idx = np.int(np.round(len(t)/10))
    tsE = R[t,:,0][0::idx,:] / brian2.Hz
    t_split = t[0::idx]*dt
    vmax = np.max(tsE)
    for i in range(10):
        vals = recort(tsE[i,:], surf, cortex)
        # fix time title; should be in seconds
        plotting.plot_surf_stat_map(c.get('surf_file_inf'), vals, vmax = vmax,
                                    view = 'lateral',
                                    title=('%s' % t_split[i]),
                                    colorbar = False, axes=ax[0][i]
                                   )
        plotting.plot_surf_stat_map(c.get('surf_file_inf'), vals, vmax = vmax,
                                    view = 'medial',
                                    colorbar = False, axes=ax[1][i]
                                   )
    plt.savefig('./figures/stim_brains_E.png')
    plt.close(fig)


def plot_timeseries(R):

    config = configparser.ConfigParser()
    config.read('./config.ini')
    c = config['SETUP']

    dt             = float(c.get('dt')) * brian2.ms  # timestep
    start_time     = 0.0
    end_time       = float(c.get('trial_length'))/1000
    stim_on        = float(c.get('stim_on')) * brian2.second
    stim_off       = float(c.get('stim_off')) * brian2.second
    V1_index       = get_stim(c.get('annot_file'), c.get('label_name'), c.get('cort_file'))
    tm             = np.arange((start_time)*brian2.second,(end_time)*brian2.second,dt)

    # To visualize response to stimulus
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(15,10), dpi= 80, facecolor='w', edgecolor='k')
    plt.rcParams.update({'font.size': 20})

    # Plot E population rates
    tsE = np.mean(R[np.arange(int(start_time/dt),int(end_time/dt),1),:,0][:,V1_index], axis=1)
    plt.plot(tm,tsE,color='r')
    # Plot I population rates
    tsI = np.mean(R[np.arange(int(start_time/dt),int(end_time/dt),1),:,1][:,V1_index], axis=1)
    plt.plot(tm,tsI,color='b')

    # Plot the stimulation time
    plt.plot([stim_on,stim_off],[np.max(tsE)*1.1,np.max(tsE)*1.1],color='r',linewidth=5.0)

    # place text above the stimulation line
    axes = plt.gca()
    axes.text(0.4, 1.05,'External stimulation', transform=axes.transAxes, fontsize=20, verticalalignment='top')

    plt.legend(['E','I'])
    plt.xlabel('time (s)')
    plt.ylabel('firing rate (Hz)')
    # plt.ylim(0, 80)

    plt.savefig('./figures/stim_response.png')
    plt.close(fig)
