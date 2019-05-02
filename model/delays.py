
def get_delays(surface, cortex_map):

    # DELAYS

    # Along the surface, all nodes area neighbours. They are ~1mm apart.
    # Activity propagates along the surface at about 0.33 m/s = 330mm/s = 0.33mm/ms = 0.165 mm/dt
    # Therefore there is a delay of about 6 timesteps for activity propagation along the surface
    # (Girard, Hupé & Bullier, J Neurophysiol, 2001)

    # Along the white matter, there are connections of different lengths.

    # euclidean distance between nodes:

    import numpy as np
    import connectivity
    from scipy.spatial import distance_matrix

    surf = connectivity.load_surface(surface, cortex_map)
    distances = distance_matrix(surf[0], surf[0]) # this is distance in vertices - let's assume = 1mm

    tract_delays = np.round(distances/1.75, decimals=0).astype(int) # in units of timesteps

    # In the white matter, activity propagates at about 3.5m/s, but it might propagate more quickly for
    # longer/larger tracts.
    # 3.5 m/s = 3.5mm/ms = 1.75mm/dt
    # (Girard, Hupé & Bullier, J Neurophysiol, 2001)

    max_delay = np.max(tract_delays) # in timesteps

    # initialise the matrix which stores the delayed
    # synaptic drive variables (transmitted via white matter tracts)
    S_tract_delay = np.zeros((len(surf[0]),len(surf[0])))

    return tract_delays, max_delay, S_tract_delay
