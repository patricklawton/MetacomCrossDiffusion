import numpy as np
import sympy as sy
import signac as sg
from itertools import product, combinations
from math import comb

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

# Other statepoint parameters to combine 
N_ns = [1e2] #Avg density of samples per 0-pi/2 interval
methods = ['numeric']

# Define sympy symbols
u, w, v = sy.symbols("u w v")
r_u, r_v, K_u, K_v, A_uv, A_uw, A_vw, B_uv, B_uw, B_vw = sy.symbols('r_u, r_v, K_u, K_v, A_uv, A_uw, A_vw, B_uv, B_uw, B_vw')
d_v, d_w, e_uv, e_uw, e_vw = sy.symbols('d_v, d_w, e_uv, e_uw, e_vw')

# Constants
num_parameterizations = 1e2 
num_jobs = len(modules)*num_parameterizations 
x0_trials = 10 #Number of attempts for steady state solving 
model_params = [r_u, r_v, K_u, K_v, A_uv, A_uw, A_vw, B_uv, B_uw, B_vw, d_v, d_w, e_uv, e_uw, e_vw]

# Find desired number of random model parameterizations
while len(project) < num_jobs:
    # Look for steady state in each module 
    for module_i, module in enumerate(modules):
        # Define nonlinear interaction terms
        'G_i -> prey growth, R_ij -> functional response'
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
        # Look for a parameterization with a stable steady state
        ev_check = False
        while not ev_check:
            param_vals = []
            specified_params = [item[0] for item in param_vals]
            cond = [p not in specified_params for p in model_params]
            for param in np.extract(cond, model_params):
                param_vals.append((param, 10.0 * np.random.sample()))
            # Try multiple initial points for nonlinear solving with this parameterization
            for i in range(x0_trials):
                scales = np.ones(3) * 8
                x0 = np.random.random(3) * scales
                eq_vec = (sy.Eq(f.subs(param_vals),0), 
                          sy.Eq(g.subs(param_vals),0), 
                          sy.Eq(h.subs(param_vals),0))
                var_vec = (u, v, w)
                # Try to solve for a uniform steady state
                try:
                    u_0, v_0, w_0 = sy.nsolve(eq_vec, var_vec, x0)
                except:
                    continue
                # Make sure steady state is nontrivial (i.e. all state variables nonzero)
                steady_state = np.array([u_0, v_0, w_0])
                if not np.all(steady_state > 0.05):
                    continue
                else:
                    # Make non-spatial Jacobian
                    J_symbolic = sy.Matrix([[sy.diff(f, u), sy.diff(f, v), sy.diff(f, w)],
                                            [sy.diff(g, u), sy.diff(g, v), sy.diff(g, w)],
                                            [sy.diff(h, u), sy.diff(h, v), sy.diff(h, w)]])
                    J = J_symbolic.subs(param_vals + [(u, u_0), (v, v_0), (w, w_0)])
                    # Check for stability
                    evs = np.array([complex(val) for val in J.eigenvals()])
                    if np.any(np.real(evs) > 0):
                        continue
                    else:
                        ev_check = True
                        break

        # Initialize job
        for N_n, method in product(N_ns, methods):
            sp = {'module': module, 'N_n': N_n, 'model_params': {}, 'method': method,
                  'steady_state': {'u': float(u_0), 'v': float(v_0), 'w': float(w_0)}}
            for key, val in param_vals:
                sp['model_params'][sy.sstr(key)] = val
            job = project.open_job(sp)
            job.init()
            job.data['J'] = np.array(J).astype(np.float64)
