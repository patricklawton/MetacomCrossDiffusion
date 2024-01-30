import numpy as np

# Decorator to return interaction model as a function
def get_interaction_model(module, pvs):
    '''module -> string for 3 species interaction model
       pvs -> dictionary containing model parameters 
       x[0] -> u, x[1] -> v, x[2] -> w
       G_i -> prey growth, R_ij -> functional response
    '''
    def model(x):
        G_u = pvs['r_u']*(1 - x[0]/pvs['K_u'])
        G_v = pvs['r_v']*(1 - x[1]/pvs['K_v'])
        R_uv = pvs['A_uv']/(x[0]+pvs['B_uv'])
        R_uw = pvs['A_uw']/(x[0]+pvs['B_uw'])
        R_vw = pvs['A_vw']/(x[1]+pvs['B_vw'])
        if module == 'chain':
            f = x[0] * (G_u - x[1]*R_uv)
            g = x[1] * (x[0]*pvs['e_uv']*R_uv - x[1]*pvs['d_v'] - x[2]*R_vw)
            h = x[2] * (x[1]*pvs['e_vw']*R_vw - x[2]*pvs['d_w'])
        if module == 'exploitative':
            f = x[0] * (G_u - x[1]*R_uv - x[2]*R_uw)
            g = x[1] * (x[0]*pvs['e_uv']*R_uv - x[1]*pvs['d_v'])
            h = x[2] * (x[0]*pvs['e_uw']*R_uw - x[2]*pvs['d_w'])
        if module == 'apparent':
            f = x[0] * (G_u - x[2]*R_uw)
            g = x[1] * (G_v - x[2]*R_vw)
            h = x[2] * (x[0]*pvs['e_uw']*R_uw + x[1]*pvs['e_vw']*R_vw - x[2]*pvs['d_w'])
        if module == 'omnivory':
            f = x[0] * (G_u - x[1]*R_uv - x[2]*R_uw)
            g = x[1] * (x[0]*pvs['e_uv']*R_uv - x[2]*R_vw - x[1]*pvs['d_v'])
            h = x[2] * (x[0]*pvs['e_uw']*R_uw + x[1]*pvs['e_vw']*R_vw - x[2]*pvs['d_w'])
        return [f, g, h]
    return model

# Function to convert list of (n-1) spherical coordinates to n cartesian coordinates
def spherical_to_cartesian(ang_coord_sample):
    cart_vec = []
    for i in range(len(ang_coord_sample) + 1):
        if i == 0:
            cart_coord = -np.cos(ang_coord_sample[i])
        elif i < (len(ang_coord_sample)):
            cart_coord = -np.cos(ang_coord_sample[i])
            for j in range(i):
                cart_coord = np.sin(ang_coord_sample[j]) * cart_coord
        else:
            cart_coord = -np.sin(ang_coord_sample[i-1])
            for j in range(i-1):
                cart_coord = np.sin(ang_coord_sample[j]) * cart_coord
        if abs(0.0 - cart_coord) < np.finfo(np.float64).eps:
            cart_coord = 0.0
        cart_vec.append(cart_coord)
    return cart_vec

# Function to get the limits of cross diffusive rates, in cartesian corrdinates
def get_cross_limits(ij, adj):
    '''ij -> tuple of C element indices
       adj -> adjacency matrix'''
    i, j = ij
    # j feeds on i -> Cij > 0 (predator avoidance)
    if adj[i,j] == 1.0:
        lim = (0, 1)
    # i feeds on j -> Cij < 0 (prey tracking)
    elif adj[j,i] == 1.0:
        lim = (-1, 0)
    # If no interaction, no dispersal response
    else:
        lim = (np.nan, np.nan)
    return lim

def get_num_spatials(n_cross, sample_density=1e2):
    '''Get the number of spatial parameterizations given a value of 
       n_cross and desired density per sign permutation'''
    num_spatials = int(sample_density * (n_cross + 3)**2.5)
    return num_spatials
