import numpy as np
import sympy as sy
import signac
from itertools import product, combinations
from math import comb

# Initialize signac project (i.e. create workspace dir)
project = signac.init_project()

# Write adjacency matrices and module labels to file
modules = np.array(['chain', 'exploitative', 'apparent', 'omnivory'])
with signac.H5Store(project.fn('shared_data.h5')).open(mode='w') as shared_data:
    shared_data['adj_mats'] =  np.array([ 
        [[0.,1,0],[0,0,1],[0,0,0]],
        [[0,1,1],[0,0,0],[0,0,0]],
        [[0,0,1],[0,0,1],[0,0,0]],
        [[0,1,1],[0,0,1],[0,0,0]]
        ])
    shared_data['modules'] = [str(module) for module in modules]

# Other statepoint parameters to combine 
N_ns = [1e2] #Avg density of samples per 0-pi/2 interval
C_offdiags = [(0,1), (0,2),
              (1,0), (1,2),
              (2,0), (2,1)]
#n_cross_arr = [i+1 for i in range(6)] #Number of nonzero cross dispersal elements
n_cross_arr = [0, 1]

# Define sympy symbols
u, w, v = sy.symbols("u w v")
r_u, r_v, K_u, K_v, A_uv, A_uw, A_vw, B_uv, B_uw, B_vw = sy.symbols('r_u, r_v, K_u, K_v, A_uv, A_uw, A_vw, B_uv, B_uw, B_vw')
d_v, d_w, e_uv, e_uw, e_vw = sy.symbols('d_v, d_w, e_uv, e_uw, e_vw')

# Constants
num_parameterizations = 1e2 
num_jobs = sum([comb(len(C_offdiags),n_cross)*len(modules)*num_parameterizations for n_cross in n_cross_arr])
#print(num_jobs)
#import sys; sys.exit()
x0_trials = 10
model_params = [r_u, r_v, K_u, K_v, A_uv, A_uw, A_vw, B_uv, B_uw, B_vw, d_v, d_w, e_uv, e_uw, e_vw]

param_i = 0
while len(project) < num_jobs:
    J_mats = np.zeros((len(modules), 3, 3))
    ev_checks = np.full(len(modules), False)
    while not np.all(ev_checks):
        param_vals = []
        specified_params = [item[0] for item in param_vals]
        cond = [p not in specified_params for p in model_params]
        for param in np.extract(cond, model_params):
            param_vals.append((param, 10.0 * np.random.sample()))
        # Try multiple initial points for nonlinear solving with this parameterization
        for i in range(x0_trials):
            steady_states = np.zeros((len(modules), 3))
            ev_checks = np.full(len(modules), False)
            scales = np.ones(3) * 8
            x0 = np.random.random(3) * scales
            # Look for steady state in each module 
            for module_i, module in enumerate(modules):
                # Define nonlinear interaction terms
                'G_i -> prey growth, R_ij -> functional response, F_ij -> predation gains'
                G_u = r_u*(1 - u/K_u)
                G_v = r_v*(1 - v/K_v)
                R_uv = A_uv/(u+B_uv)
                R_uw = A_uw/(u+B_uw)
                R_vw = A_vw/(v+B_vw)
                if module == 'chain':
                    f = u * (G_u - v*R_uv)
                    g = v * (e_uv*R_uv - v*d_v - w*R_vw)
                    h = w * (e_vw*R_vw - w*d_w)
                if module == 'exploitative':
                    f = u * (G_u - v*R_uv - w*R_uw)
                    g = v * (e_uv*R_uv - v*d_v)
                    h = w * (e_uw*R_uw - w*d_w)
                if module == 'apparent':
                    f = u * (G_u - w*R_uw)
                    g = v * (G_v - w*R_vw)
                    h = w * (e_uw*R_uw + e_vw*R_vw - w*d_w)
                if module == 'omnivory':
                    f = u * (G_u - v*R_uv - w*R_uw)
                    g = v * (e_uv*R_uv - w*R_vw - v*d_v)
                    h = w * (e_uw*R_uw + e_vw*R_vw - w*d_w)

                # Solve for the uniform steady state
                eq_vec = (sy.Eq(f.subs(param_vals),0), 
                          sy.Eq(g.subs(param_vals),0), 
                          sy.Eq(h.subs(param_vals),0))
                var_vec = (u, v, w)
                try:
                    u_0, v_0, w_0 = sy.nsolve(eq_vec, var_vec, x0)
                except:
                    ev_checks[module_i] = False
                    continue
                # Make sure steady state is nontrivial (i.e. all state variables nonzero)
                steady_state = np.array([u_0, v_0, w_0])
                if not np.all(steady_state > 0.05):
                    ev_checks[module_i] = False
                    continue
                else:
                    steady_states[module_i] = (u_0, v_0, w_0)
                    # Make non-spatial Jacobian
                    J_star = sy.Matrix([[sy.diff(f, u), sy.diff(f, v), sy.diff(f, w)],
                                   [sy.diff(g, u), sy.diff(g, v), sy.diff(g, w)],
                                   [sy.diff(h, u), sy.diff(h, v), sy.diff(h, w)]])
                    J_sub = J_star.subs(param_vals + [(u, u_0), (v, v_0), (w, w_0)])
                    # Check for stability
                    evs = np.array([complex(val) for val in J_sub.eigenvals()])
                    if np.any(np.real(evs) > 0):
                        ev_checks[module_i] = False
                        continue
                    else:
                        J_mats[module_i] = np.array(J_sub).astype(np.float64)
                        ev_checks[module_i] = True
            if np.all(ev_checks):
                param_i += 1
                break
    # Initialize workspace
    for n_cross in n_cross_arr:
        # Get possible combinations of cross-dispersal elements 
        if (n_cross == 0) or (n_cross == 0.0):
            cross_combs = [[]]
        else:
            cross_combs = [comb for comb in combinations(C_offdiags, n_cross)]
        for module, C_ijs, N_n in product(modules, cross_combs, N_ns):
            sp = {'module': module, 'C_ijs': C_ijs, 'N_n': N_n, 
                  'x0': x0, 'model_params': {}, 'param_i': param_i}
            for key, val in param_vals:
                sp['model_params'][sy.sstr(key)] = val
            job = project.open_job(sp)
            job.init()
            job.data['J_sub'] = J_mats[modules == module][0]
