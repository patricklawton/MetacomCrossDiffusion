import numpy as np
import sympy as sy
import signac as sg
from flow import FlowProject
from itertools import product

project = sg.get_project()

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
    while len(data[ang_coords[0]]) < (2**len(C_k))*sp.N_n:
        # Sample and store directional data
        ang_coord_sample = []
        for i, ang_coord in enumerate(ang_coords):
            if i < len(ang_coords) - 1:
                phi_range = (0.0, np.pi)
            else:
                phi_range = (0.0, 2*np.pi)
            phi_sample = phi_range[1] * np.random.sample()
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
        # Store critical kappas
        for cond, kappa_arr in zip(['wav', 'st'], [kappa_wavs, kappa_sts]):
            kappa_c_dir = []
            for kappa_c in kappa_arr:
                if np.isreal(kappa_c) and (kappa_c > 0):
                    kappa_c_dir.append(kappa_c)
                else:
                    kappa_c_dir.append(np.nan)
            data['kappa_cs'][cond].append(kappa_c_dir)

    # Store linear stability type (Omega integrand)
    for i in range(len(data[ang_coords[0]])):
        wav_nonan = [val for val in data['kappa_cs']['wav'][i] if not np.isnan(val)]
        st_nonan = [val for val in data['kappa_cs']['st'][i] if not np.isnan(val)]
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

    # Store job data
    for key, item in data.items():
        if isinstance(item, dict):
            for sub_key, sub_item in item.items():
                job.data[key+'/'+sub_key] = np.array(sub_item)
        else:
            job.data[key] = np.array(item)
    job.doc['surface_generated'] = True
    stop = timeit.default_timer()
    #print('Time:', stop - start)

if __name__ == "__main__":
    FlowProject().main()
