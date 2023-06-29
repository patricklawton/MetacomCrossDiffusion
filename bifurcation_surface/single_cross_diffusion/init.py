import numpy as np
import sympy as sy
import signac
from itertools import product

# Initialize signac project (i.e. create workspace dir)
project = signac.init_project()

# Write adjacency matrices and lables to file
with signac.H5Store(project.fn('shared_data.h5')).open(mode='w') as shared_data:
    shared_data['adj_mats'] =  np.array([ 
        [[0.,1,0],[0,0,1],[0,0,0]],
        [[0,1,1],[0,0,0],[0,0,0]],
        [[0,0,1],[0,0,1],[0,0,0]],
        [[0,1,1],[0,0,1],[0,0,0]]
        ])
    shared_data['modules'] = ['chain', 'exploitative', 'apparent', 'omnivory']


# Define sympy symbols
u, w, v = sy.symbols("u w v")
r_u, r_v, K_u, K_v, A_uv, A_uw, A_vw, B_uv, B_uw, B_vw = sy.symbols('r_u, r_v, K_u, K_v, A_uv, A_uw, A_vw, B_uv, B_uw, B_vw')
k_uv, k_uw, k_vw, h_uv, h_uw, h_vw = sy.symbols('k_uv, k_uw, k_vw, h_uv, h_uw, h_vw') 

# Create non-spatial jacobians for each module
J_mats = []
modules = np.array(['chain', 'exploitative', 'apparent', 'omnivory'])
param_vals = [(r_u, 3.0),
     (K_u, 3.0),
     (A_uv, 1.0),
     (A_vw, 1.0),
     (B_uv, 0.25),
     (B_vw, 0.25),
     (k_uv, 6.0),
     (k_vw, 4.0),
     (h_uv, 1.0),
     (h_vw, 0.25),
     (r_v, 4.580904893992301),
     (K_v, 3.2645220255726133),
     (k_uw, 4.782821123187842),
     (h_uw, 0.5565011459148267),
     (A_uw, 3.2931039851054527),
     (B_uw, 4.9636504931459955)]
for module in modules:
    # Define nonlinear interaction terms
    'G_i -> prey i growth, F_ij -> gain of j from i, L_ij -> loss of i to j'
    G_u = r_u*(1 - u/K_u)
    G_v = r_v*(1 - v/K_v)
    F_uv = k_uv*(1 - (h_uv*v)/u)
    F_uw = k_uw*(1 - (h_uw*w)/u)
    F_vw = k_vw*(1 - (h_vw*w)/v)
    L_uv = v*(A_uv/(u+B_uv))
    L_uw = w*(A_uw/(u+B_uw))
    L_vw = w*(A_vw/(v+B_vw))
    if module == 'chain':
        f = u * (G_u - L_uv)
        g = v * (F_uv - L_vw)
        h = w * F_vw
    if module == 'exploitative':
        f = u * (G_u - L_uv - L_uw)
        g = v * F_uv
        h = w * F_uw
    if module == 'apparent':
        f = u * (G_u - L_uw)
        g = v * (G_v - L_vw)
        h = w * (F_uw + F_vw)
    if module == 'omnivory':
        f = u * (G_u - L_uv - L_uw)
        g = v * (F_uv - L_vw)
        h = w * (F_uw + F_vw)

    # Solve for the uniform steady state
    eq_vec = (sy.Eq(f.subs(param_vals),0), 
              sy.Eq(g.subs(param_vals),0), 
              sy.Eq(h.subs(param_vals),0))
    var_vec = (u, v, w)
    x0 = (1, 2.5, 10.2)
    u_0, v_0, w_0 = sy.nsolve(eq_vec, var_vec, x0)

    # Make non-spatial Jacobian
    J_star = sy.Matrix([[sy.diff(f, u), sy.diff(f, v), sy.diff(f, w)],
                   [sy.diff(g, u), sy.diff(g, v), sy.diff(g, w)],
                   [sy.diff(h, u), sy.diff(h, v), sy.diff(h, w)]])
    J_sub = J_star.subs(param_vals + [(u, u_0), (v, v_0), (w, w_0)])
    J_mats.append(np.array(J_sub).astype(np.float64))
J_mats = np.array(J_mats)

# Other statepoint parameters to combine 
C_offdiags = np.array([(0,1), (0,2),
                      (1,0), (1,2),
                      (2,0), (2,1)])
N_ns = [1e2] #Avg density of samples per 0-pi/2 interval

# Initialize workspace
for module, C_offdiag, N_n in product(modules, C_offdiags, N_ns):
    sp = {'module': module, 'C_offdiag': C_offdiag, 'N_n': N_n}
    job = project.open_job(sp)
    job.init()
    job.data['J_sub'] = J_mats[modules == module][0]
