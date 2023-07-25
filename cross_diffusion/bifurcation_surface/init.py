import numpy as np
from scipy import optimize
import signac as sg
from itertools import product
from project import get_interaction_model

# Initialize signac project 
project = sg.init_project()

# Write adjacency matrices and module labels to project level data
modules = np.array(['chain', 'exploitative', 'apparent', 'omnivory'])
sd_fn = project.fn('shared_data.h5')
with sg.H5Store(sd_fn).open(mode='w') as sd:
    sd['adj_mats'] =  np.array([ 
        [[0.,1,0],[0,0,1],[0,0,0]],
        [[0,1,1],[0,0,0],[0,0,0]],
        [[0,0,1],[0,0,1],[0,0,0]],
        [[0,1,1],[0,0,1],[0,0,0]]
        ])
    sd['modules'] = [str(module) for module in modules]

# Variable parameters (constant if only one value specified) 
N_ns = [1e2] #Avg density of samples per 0-pi/2 interval
methods = ['numeric']

# Constants
num_parameterizations = 1e2 #Per module 
x0_trials = int(1e3) #Number of attempts for steady state solving 
x0_scale = 10 #Sets range of (random) initial values drawn for steady state solving 
param_scale = 10 #Sets range of (random) values for model parameters
nonzero_thresh = 1e-5 #Threshold for accepting a steady state variable as nonzero
#print(nonzero_thresh)
param_labels = ['r_u', 'r_v', 'K_u', 'K_v', 'A_uv', 'A_uw', 'A_vw',
                'B_uv', 'B_uw', 'B_vw', 'd_v', 'd_w', 'e_uv', 'e_uw', 'e_vw']

# Find desired number of random model parameterizations per module
for module_i, module in enumerate(modules):
    #print(module)
    #print('----------------------------')
    while len(project.find_jobs({'module': module})) < num_parameterizations:
        model_params = {}
        for param in param_labels:
            model_params[param] = param_scale * np.random.sample()
        interaction_model = get_interaction_model(module, model_params)
        # Try multiple initial points for nonlinear solving with this parameterization
        steady_state_found = False 
        for i in range(x0_trials):
            x0 = np.random.random(3) * x0_scale
            sol = optimize.root(interaction_model, x0)
            #print(sol)
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
        for N_n, method in product(N_ns, methods):
            sp = {'module': module, 'N_n': N_n, 'model_params': model_params, 
                  'method': method, 'local_stability': local_stability}
            if local_stability != 'infeasible':
                sp['steady_state'] = {'u': float(sol.x[0]), 
                                      'v': float(sol.x[1]), 
                                      'w': float(sol.x[2])}
            job = project.open_job(sp)
            job.init()
            job.data['J'] = J
