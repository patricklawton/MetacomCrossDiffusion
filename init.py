import numpy as np
from scipy import optimize
import signac as sg
import os
from itertools import product, combinations
from global_functions import * 

# Initialize signac project 
project = sg.init_project()

# Constants
num_parameterizations = 2500 #Per module 
x0_trials = int(1e3) #Number of attempts for steady state solving 
x0_scale = 10 #Sets range of (random) initial values drawn for steady state solving 
param_scale = 10 #Sets range of (random) values for model parameters
nonzero_thresh = 1e-15 #Threshold for accepting a steady state variable as nonzero
param_labels = ['r_u', 'r_v', 'K_u', 'K_v', 'A_uv', 'A_uw', 'A_vw',
                'B_uv', 'B_uw', 'B_vw', 'd_v', 'd_w', 'e_uv', 'e_uw', 'e_vw']
N_n = 100 #Avg density of samples per 0-pi/2 interval
total_samples = get_num_spatials(6, sample_density=N_n)
modules = np.array(['chain', 'exploitative', 'apparent', 'omnivory'])
'''The order of this list defines the map from n cartesian coordinates to (n-1) spherical coordinates.
   The last two spherical coordinates always map to the three diagonal C elements, such that
   phi_(n-2) -> C[0,0], phi_(n-1) -> (C[1,1], C[2,2])'''
C_offdiags = np.array([(0,1), (0,2),
                       (1,0), (1,2),
                       (2,0), (2,1)])
cross_labels = []
for n_cross in np.arange(0, len(C_offdiags)+1):
    if n_cross == 0:
        cross_combs = [[]]
    else:
        cross_combs = [comb for comb in combinations(C_offdiags, n_cross)]
    for cross_comb in cross_combs:
        # Make label for nonzero offdiagonals to store in job data
        if len(cross_comb) == 0:
            cross_label = 'diag'
        else:
            cross_label = ','.join([str(c[0])+str(c[1]) for c in cross_comb])
        cross_labels.append(cross_label)
resample = False

# Sample dispersal parameters and write to shared data
sd_fn = project.fn('shared_data.h5')
if (os.path.isfile(sd_fn) == False) or resample:
    rng = np.random.default_rng()
    ang_coord_samples = rng.uniform(0, np.pi, size=(len(C_offdiags),total_samples))
    ang_coord_samples = np.vstack((ang_coord_samples, rng.uniform(np.pi/2, np.pi, size=total_samples)))
    ang_coord_samples = np.vstack((ang_coord_samples, rng.uniform(np.pi, 3*np.pi/2, size=total_samples)))
    cart_coord_samples = []
    for sample_i in range(ang_coord_samples.shape[1]):
        cart_coord_sample = spherical_to_cartesian(ang_coord_samples[:, sample_i])
        cart_coord_samples.append(cart_coord_sample)
    cart_coord_samples = np.array(cart_coord_samples)
    cart_coord_samples = np.swapaxes(cart_coord_samples, 0, 1)
    with sg.H5Store(sd_fn).open(mode='w') as sd:
        sd['adj_mats'] =  np.array([ 
            [[0.,1,0],[0,0,1],[0,0,0]],
            [[0,1,1],[0,0,0],[0,0,0]],
            [[0,0,1],[0,0,1],[0,0,0]],
            [[0,1,1],[0,0,1],[0,0,0]]
            ])
        sd['modules'] = [str(module) for module in modules]
        sd['cross_labels'] = cross_labels
        sd['C_offdiags'] = C_offdiags
        sd['N_n'] = N_n
        sd['ang_coord_samples'] = ang_coord_samples
        sd['cart_coord_samples'] = cart_coord_samples

# Variable parameters (constant if only one value specified) 
methods = ['numeric']

# Find desired number of random model parameterizations per module
for method, module in product(methods, modules):
    jobs_filter = {'method': method, 'module': module}
    while len(project.find_jobs(jobs_filter)) < num_parameterizations:
        model_params = {}
        for param in param_labels:
            model_params[param] = param_scale * np.random.sample()
        interaction_model = get_interaction_model(module, model_params)
        # Try multiple initial points for nonlinear solving with this parameterization
        steady_state_found = False 
        for i in range(x0_trials):
            x0 = np.random.random(3) * x0_scale
            sol = optimize.root(interaction_model, x0)
            # Make sure steady state is nontrivial (i.e. all state variables nonzero)
            if sol.success and np.all(sol.x > nonzero_thresh):
                steady_state_found = True
                # Make non-spatial Jacobian
                J = optimize.approx_fprime(sol.x, interaction_model)
                # Check for non-spatial stability
                re_evs = np.real(np.linalg.eigvals(J))
                if np.all(re_evs < 0):
                    local_stability = 'stable'
                else:
                    local_stability = 'unstable'
                break
            else:
                continue
        if not steady_state_found:
            local_stability = 'infeasible'

        # Initialize job
        sp = {'module': module, 'model_params': model_params, 
              'method': method, 'local_stability': local_stability}
        if local_stability != 'infeasible':
            sp['steady_state'] = {'u': float(sol.x[0]), 
                                  'v': float(sol.x[1]), 
                                  'w': float(sol.x[2])}
        job = project.open_job(sp)
        job.init()
        job.data['J'] = J
