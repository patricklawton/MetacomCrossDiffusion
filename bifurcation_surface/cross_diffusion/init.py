import numpy as np
import sympy as sy
import signac
from itertools import product, combinations

# Initialize signac project (i.e. create workspace dir)
project = signac.init_project()


# Write adjacency matrices and lables to file
modules = np.array(['chain', 'exploitative', 'apparent', 'omnivory'])
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
d_v, d_w, e_uv, e_uw, e_vw = sy.symbols('d_v, d_w, e_uv, e_uw, e_vw')

# Create non-spatial jacobians for each module
J_mats = []
param_vals = [(r_u, 5.387123939187507),
 (r_v, 4.715284021249707),
 (K_u, 1.9728605914743158),
 (K_v, 2.232308279445856),
 (A_uv, 1.751550913884543),
 (A_uw, 6.318075941167866),
 (A_vw, 5.589522483278595),
 (B_uv, 1.504426612266082),
 (B_uw, 6.829212231799925),
 (B_vw, 7.324566573298396),
 (d_v, 7.022845339163734),
 (d_w, 3.0663095303429166),
 (e_uv, 6.616029006415757),
 (e_uw, 9.276818518515862),
 (e_vw, 5.246212224679999)]
for module in modules:
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
    x0 = (1.28081606, 0.63280683, 4.32831638)
    u_0, v_0, w_0 = sy.nsolve(eq_vec, var_vec, x0)

    # Make non-spatial Jacobian
    J_star = sy.Matrix([[sy.diff(f, u), sy.diff(f, v), sy.diff(f, w)],
                   [sy.diff(g, u), sy.diff(g, v), sy.diff(g, w)],
                   [sy.diff(h, u), sy.diff(h, v), sy.diff(h, w)]])
    J_sub = J_star.subs(param_vals + [(u, u_0), (v, v_0), (w, w_0)])
    J_mats.append(np.array(J_sub).astype(np.float64))
J_mats = np.array(J_mats)

# Other statepoint parameters to combine 
N_ns = [1e2] #Avg density of samples per 0-pi/2 interval
C_offdiags = [(0,1), (0,2),
              (1,0), (1,2),
              (2,0), (2,1)]
#n_cross_arr = [i+1 for i in range(6)]
n_cross_arr = [1]

# Initialize workspace
for n_cross in n_cross_arr:
    cross_combs = [comb for comb in combinations(C_offdiags, n_cross)]
    for module, C_ijs, N_n in product(modules, cross_combs, N_ns):
        sp = {'module': module, 'C_ijs': C_ijs, 'N_n': N_n}
        job = project.open_job(sp)
        job.init()
        job.data['J_sub'] = J_mats[modules == module][0]
