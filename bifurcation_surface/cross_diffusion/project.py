import numpy as np
import sympy as sy
import signac as sg
from flow import FlowProject
from itertools import product

project = sg.get_project()

# Read in module info from shared data
sd_fn = project.fn('shared_data.h5')
with sg.H5Store(sd_fn).open(mode='r') as sd:
    adj_mats = np.array(sd['adj_mats'])
    modules = np.array([i.decode() for i in sd['modules']])

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

@FlowProject.post(lambda job: job.doc.get('surface_generated'))
@FlowProject.operation
def generate_surface(job):
    import timeit
    start = timeit.default_timer()
    sp = job.sp

    # Read in non-spatial jacobian from job data
    with job.data:
        J_sub = np.array(job.data['J_sub'])

    # Define sympy symbols
    kappa, lamda = sy.symbols("kappa lamda")
    J11, J12, J13, J21, J22, J23, J31, J32, J33 = sy.symbols('J11 J12 J13 J21 J22 J23 J31 J32 J33')
    C11, C12, C13, C21, C22, C23, C31, C32, C33 = sy.symbols('C11 C12 C13 C21 C22 C23 C31 C32 C33')

    # Construct metacommunity jacobian with symbolic dispersal entries
    J = sy.Matrix([[J11, J12, J13], 
                   [J21, J22, J23],
                   [J31, J32, J33]])
    C = sy.Matrix([[C11, C12, C13],
                   [C21, C22, C23],
                   [C31, C32, C33]])
    for i, j in product(range(C.shape[0]), repeat=2):
        if (i != j) and ([i,j] not in sp['C_ijs']):
            C[i,j] = 0.0
    M_sub = J_sub - kappa*C

    # Get oscillatory and stationary pattern-forming instability conditions
    M_star = sy.zeros(3,3)
    for i,j in product(range(C.shape[0]), repeat=2):
        if C[i,j] != 0:
            M_star[i,j] = J_sub[i,j] - C[i,j]*kappa
        else:
            M_star[i,j] = J_sub[i,j]
    p = M_star.charpoly(lamda) #characteristic polynomial
    p_coeffs = p.all_coeffs()
    I_wav = p_coeffs[3] - p_coeffs[1]*p_coeffs[2] #oscillatory
    I_st = p_coeffs[3] #stationary

    # Make lists for spatially dependent elements of C, J, and M
    C_k, M_sub_k, J_k, J_sub_k = ([],[],[],[])
    C_nonzero_indices = [list(C_ij) for C_ij in job.sp['C_ijs']] + [[i,i] for i in range(3)]
    for i, j in C_nonzero_indices:
            C_k.append(C[i,j])
            M_sub_k.append(M_sub[i,j])
            J_k.append(J[i,j])
            J_sub_k.append(J_sub[i,j])

    # Initialize data	
    data = {'kappa_cs': {'wav': [], 'st': []},
            'omega_integrand': {'wav': [], 'st': [], 'stab': []}}
    ang_coords = ['phi_{}'.format(i+1) for i in range(len(C_k)-1)]
    for ang_coord in ang_coords:
        data.update({ang_coord: []})
    cart_coords = [sy.sstr(C_ij) for C_ij in C_k]
    #for cart_coord in cart_coords:
    #    data.update({cart_coord: []})

    # Sample n dimensional dispersal parameter space in (n-1) spherical coordinates
    #while len(data[ang_coords[0]]) < (2**len(C_k))*sp.N_n:
    if len(sp['C_ijs']) == 0:
        total_samples = 3*sp.N_n
    else:
        total_samples = (2**(len(C_k)-3) + 3)*sp.N_n
    while len(data[ang_coords[0]]) < total_samples:
        # Sample and store directional data
        ang_coord_sample = []
        for i, ang_coord in enumerate(ang_coords):
            if len(sp['C_ijs']) == 0:
                if i < len(ang_coords) - 1:
                    phi_range = (np.pi/2, np.pi)
                else:
                    phi_range = (np.pi, 3*np.pi/2)
            else:
                if i <= len(sp['C_ijs']) - 1:
                    phi_range = (0, np.pi)
                elif i < len(ang_coords) - 1:
                    phi_range = (np.pi/2, np.pi)
                else:
                    phi_range = (np.pi, 3*np.pi/2)
            phi_sample = (phi_range[-1] - phi_range[0]) * np.random.sample() + phi_range[0]
            ang_coord_sample.append(phi_sample)
        for i, coord in enumerate(ang_coord_sample):
            data[ang_coords[i]].append(coord)
        # Convert spherical to cartesian coordinates
        cart_coord_sample = spherical_to_cartesian(ang_coord_sample)
        #for i, coord in enumerate(cart_coord_sample):
        #    data[cart_coords[i]].append(coord)

        # Find and store critical kappa values for this parameterization
        kappa_wavs, kappa_sts = ([], [])
        I_wav_sub = I_wav.subs([(C_k[i], cart_coord_sample[i]) for i in range(len(C_k))])
        I_st_sub = I_st.subs([(C_k[i], cart_coord_sample[i]) for i in range(len(C_k))])
        try:
            wav_sols = sy.solveset(I_wav_sub, kappa).args
            st_sols = sy.solveset(I_st_sub, kappa).args
        except:
            wav_sols, st_sols = ([np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan])
        for sol in wav_sols:
            try:
                kappa_wavs.append(float(sol))
            except:
                kappa_wavs.append(complex(sol))
        for sol in st_sols:
            try:
                kappa_sts.append(float(sol))
            except:
                kappa_sts.append(complex(sol))
        for cond, kappa_arr in zip(['wav', 'st'], [kappa_wavs, kappa_sts]):
            kappa_c_dir = []
            for kappa_c in kappa_arr:
                if np.isreal(kappa_c) and (kappa_c > 0):
                    kappa_c_dir.append(kappa_c)
                else:
                    kappa_c_dir.append(np.nan)
            data['kappa_cs'][cond].append(kappa_c_dir)

        # Store linear stability type (Omega integrand)
        #for i in range(len(data[ang_coords[0]])):
        wav_nonan = [val for val in data['kappa_cs']['wav'][-1] if not np.isnan(val)]
        st_nonan = [val for val in data['kappa_cs']['st'][-1] if not np.isnan(val)]
        wav = len(wav_nonan) > 0
        st = len(st_nonan) > 0
        if wav and st:
            wav_before_st = min(wav_nonan) < min(st_nonan)
            st_before_wav = min(st_nonan) < min(wav_nonan)
        else:
            wav_before_st = wav
            st_before_wav = st
        data['omega_integrand']['wav'].append(wav_before_st)
        data['omega_integrand']['st'].append(st_before_wav)
        data['omega_integrand']['stab'].append((wav == False) and (st == False))

    # Store all job data
    for key, item in data.items():
        if isinstance(item, dict):
            for sub_key, sub_item in item.items():
                job.data[key+'/'+sub_key] = np.array(sub_item)
        else:
            job.data[key] = np.array(item)
    job.doc['surface_generated'] = True
    stop = timeit.default_timer()
    #print('Time:', stop - start)

@FlowProject.pre(lambda job: job.doc.get('surface_generated'))
@FlowProject.post(lambda job: job.doc.get('data_processed'))
@FlowProject.operation
def process_data(job):
    with job.data:
        C_ijs = list(job.sp['C_ijs'])
        data_shape = [2 if i < (len(C_ijs) + 1) else 4 for i in range(len(C_ijs) + 2)]
        omega_sec_mat = np.zeros(data_shape)
        # Loop over parameter space sections
        for phi_lims in product(*[list(range(dim)) for dim in data_shape]):
            conds = []
            for lim_i, lim in enumerate(phi_lims):
                conds.append(np.array(job.data['phi_'+str(lim_i+1)]) > lim*np.pi/2)
                conds.append(np.array(job.data['phi_'+str(lim_i+1)]) < (lim + 1)*np.pi/2)
            # Find indices where angular coordinates are all within section ranges
            cond = np.all(conds, axis=0)
            if np.all(cond == False):
                omega_sec_mat[phi_lims] = np.nan
            else:
                omega_integrand_sec = np.array(job.data['omega_integrand/wav'])[cond] + np.array(job.data['omega_integrand/st'])[cond]
                omega_sec_mat[phi_lims] = sum(omega_integrand_sec) / sum(cond)
        job.data['omega_sec_mat'] = omega_sec_mat
        '''For now below only works for 1 cross element'''
        if len(C_ijs) != 0:
            adj = adj_mats[modules==job.sp.module][0]
            i, j = C_ijs[0]
            if adj[i,j] == 1.0:
                # j feeds on i -> C_ij > 0
                phi_lim_0 = 1
            elif adj[j,i] == 1.0:
                # i feeds on j -> C_ij < 0
                phi_lim_0 = 0
            else:
                # Some special cases where C_ij != 0 for indirect interactions
                if (job.sp.module == 'exploitative') and (C_ijs[0] in [[1,2], [2,1]]):
                    # Negative interaction btwn predators in exploitative -> C_ij > 0
                    phi_lim_0 = 1
                elif (job.sp.module == 'apparent') and (C_ijs[0] in [[0,1], [1,0]]):
                    # Negative interaction btwn prey in apparent -> C_ij > 0
                    '''Not really sure about this assumption'''
                    phi_lim_0 = 1
                # For all cases besides those above, C_ij = 0
                else:
                    phi_lim_0 = np.nan
            omega_constrained = omega_sec_mat[phi_lim_0,1,2] if not np.isnan(phi_lim_0) else np.nan
            job.doc['omega_constrained'] = omega_constrained 
            job.doc['omega_unconstrained'] = np.mean(omega_sec_mat, axis=0)[1,2]
    job.doc['data_processed'] = True

if __name__ == "__main__":
    FlowProject().main()
