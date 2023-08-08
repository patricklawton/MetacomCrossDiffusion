import numpy as np
import sympy as sy
import signac as sg
import sys
from flow import FlowProject
from itertools import product, combinations, accumulate, groupby
import timeit 

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

# Open up signac project
project = sg.get_project()

# Read in some things from project data
sd_fn = project.fn('shared_data.h5')
with sg.H5Store(sd_fn).open(mode='r') as sd:
    adj_mats = np.array(sd['adj_mats']) #Adjacency matricies
    modules = np.array([i.decode() for i in sd['modules']]) #Interaction module labels
    cross_labels = np.array([i.decode() for i in sd['cross_labels']]) #Cross-diffusion scenario labels

# Define possible off diagonal C elements
C_offdiags = [(0,1), (0,2),
              (1,0), (1,2),
              (2,0), (2,1)]
n_cross_arr = [i for i in range(len(C_offdiags) + 1)] #Number of nonzero cross dispersal elements

@FlowProject.pre(lambda job: job.sp['local_stability'] == 'stable')
@FlowProject.post(lambda job: job.doc.get('surface_generated'))
@FlowProject.operation
def generate_surface(job):
    start = timeit.default_timer()
    sp = job.sp
    
    # Read in non-spatial jacobian from job data
    with job.data:
        J = np.array(job.data['J'])

    # Loop over all possible combinations of cross-dispersal elements 
    for n_cross in n_cross_arr:
        if n_cross == 0:
            cross_combs = [[]]
        else:
            cross_combs = [comb for comb in combinations(C_offdiags, n_cross)]
        for cross_comb in cross_combs:
            # Make list for nonzero elements of C
            '''The order of this list defines the map from n cartesian coordinates to (n-1) spherical coordinates.
               The last two spherical coordinates always map to the three diagonal C elements, such that
               phi_(n-2) -> C[0,0], phi_(n-1) -> (C[1,1], C[2,2])'''
            C_nonzero = [list(Cij) for Cij in cross_comb] + [[i,i] for i in range(3)]
            # Make label for nonzero offdiagonals to store in job data
            if len(cross_comb) == 0:
                cross_label = 'diag'
            else:
                cross_label = ','.join([str(c[0])+str(c[1]) for c in cross_comb])

            # Set up the critical kappa calculation for a given method 
            if sp['method'] == 'symbolic':
                # Initialize data for this cross-diffusive scenario
                data = {'kappa_cs': {'wav': [], 'st': []},
                        'omega_integrand': {'wav': [], 'st': [], 'stab': []}}
                # Define sympy symbols
                kappa, lamda = sy.symbols("kappa lamda")
                C11, C12, C13, C21, C22, C23, C31, C32, C33 = sy.symbols('C11 C12 C13 C21 C22 C23 C31 C32 C33')
                # Construct connectivity matrix 
                C = sy.Matrix([[C11, C12, C13],
                               [C21, C22, C23],
                               [C31, C32, C33]])
                for i, j in product(range(C.shape[0]), repeat=2):
                    if (i != j) and ([i,j] not in cross_comb):
                        C[i,j] = 0.0
                # Make list of nonzero C matrix symbols
                C_k = [C[i,j] for i, j in C_nonzero]
                # Construct metacommunity jacobian with symbolic dispersal entries
                M = sy.Matrix(J - kappa*C)
                # Get oscillatory and stationary pattern-forming instability conditions
                p = M.charpoly(lamda) #characteristic polynomial
                p_coeffs = p.all_coeffs()
                I_wav = p_coeffs[3] - p_coeffs[1]*p_coeffs[2] #oscillatory
                I_st = p_coeffs[3] #stationary
            elif sp['method'] == 'numeric':
                # Initialize data
                data = {'kappa_cs': [], 'omega_integrand': {'ddi': [], 'stab': []}}
                # Define array of kappa values to compute eigenvalues at
                #kappas = np.arange(1e3)
                kmax, step = (100, 0.1)
                kappas = np.arange(0, kmax+step, step)
            else:
                sys.exit('Invalid critical kappa computation method')


            # Initialize angular coordinates spanning dispersal coefficient space
            ang_coords = ['phi_{}'.format(i+1) for i in range(len(C_nonzero)-1)]
            for ang_coord in ang_coords:
                data.update({ang_coord: []})

            # Get the total number of dispersal parameterization samples
            'Use this if sampling all sign permutations`'
            #total_samples = 2**len(C_nonzero) * sp.N_n
            'Use this if constraining diagonal C elements to positive'
            if len(cross_comb) == 0:
                total_samples = 3*sp.N_n
            else:
                total_samples = (2**(len(C_nonzero)-3) + 3)*sp.N_n

            # Sample n dimensional cartesian dispersal parameter space in (n-1) spherical coordinates
            while len(data[ang_coords[0]]) < total_samples:
                # Draw the spherical coordinates randomly 
                ang_coord_sample = []
                for i, ang_coord in enumerate(ang_coords):
                    # Set the range of phi values for each coordinate
                    if len(cross_comb) == 0:
                        if i < len(ang_coords) - 1:
                            phi_range = (np.pi/2, np.pi)
                        else:
                            phi_range = (np.pi, 3*np.pi/2)
                    else:
                        if i <= len(cross_comb) - 1:
                            phi_range = (0, np.pi)
                        elif i < len(ang_coords) - 1:
                            phi_range = (np.pi/2, np.pi)
                        else:
                            phi_range = (np.pi, 3*np.pi/2)
                    phi_sample = (phi_range[-1] - phi_range[0]) * np.random.sample() + phi_range[0]
                    ang_coord_sample.append(phi_sample)
                # Store spherical coordinates in data
                for i, coord in enumerate(ang_coord_sample):
                    data[ang_coords[i]].append(coord)

                # Convert spherical to cartesian coordinates
                cart_coord_sample = spherical_to_cartesian(ang_coord_sample)
                
                # Solve for critical kappas (symbolically)
                if job.sp['method'] == 'symbolic':
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

                # Solve for critical kappas (numerically)
                if job.sp['method'] == 'numeric':
                    '''I think there can be at most 6 critical kappas for 3 species
                       3 roots for stationary condition, 3 roots for oscillatory condition'''
                    max_kap_crits = 6
                    # Construct connectivity matrix with random samples
                    C = np.zeros(J.shape)                     
                    for coord_idx, C_idx in enumerate(C_nonzero):
                        i, j = C_idx                          
                        C[i,j] = cart_coord_sample[coord_idx]   
                    M_vec = np.array([J - k*C for k in kappas])
                    re_evs = np.linalg.eigvals(M_vec).real
                    any_positive = np.any(re_evs > 0, axis=1)
                    critical_indices = list(accumulate(sum(1 for _ in g) for _,g in groupby(any_positive)))[:-1]
                    critical_kappas = kappas[critical_indices]
                    critical_kappas = np.pad(critical_kappas.astype(float),
                                            (0, max_kap_crits-len(critical_indices)), 
                                            mode='constant', constant_values=(np.nan,))
                    data['kappa_cs'].append(critical_kappas)
                    ddi = False
                    if len(critical_indices) != 0:
                        ddi = True
                    data['omega_integrand']['stab'].append(ddi == False)
                    data['omega_integrand']['ddi'].append(ddi)

            # Store all job data
            for key, item in data.items():
                if isinstance(item, dict):
                    for sub_key, sub_item in item.items():
                        job.data[cross_label+'/'+key+'/'+sub_key] = np.array(sub_item)
                else:
                    job.data[cross_label+'/'+key] = np.array(item)
    job.doc['surface_generated'] = True
    #stop = timeit.default_timer(); print('Time:', stop - start)

@FlowProject.pre(lambda job: job.doc.get('surface_generated') or (job.sp['local_stability'] == 'unstable'))
#@FlowProject.pre(lambda job: job.doc.get('surface_generated'))
@FlowProject.post(lambda job: 'omega_constrained' in job.doc)
@FlowProject.operation
def store_omega_in_doc(job):
    with job.data:
        # Add dictionaries to job document to store data in
        job.doc['omega_constrained'] = {}
        job.doc['omega_unconstrained'] = {}
        
        # Select adjacency matrix
        adj = adj_mats[modules==job.sp.module][0]

        # Loop over cross diffusion scenarios, stored as keys in job data
        drop = {'J'}
        for Cij_key in cross_labels:
            # Set constraints on the elements of C
            # Diagonal elements (phi_(n-2), phi_(n-1)) always > 0
            phi_diag_limits = [(np.pi/2, np.pi), (np.pi, 3*np.pi/2)]
            if Cij_key == 'diag':
                phi_cross_limits = []
            # Constrain off-diagonals to opposite sign of species interaction
            else:
                # Convert key to a list of off-diagonal C element indices
                Cij_arr = [(int(e[0]), int(e[1])) for e in Cij_key.split(',')]
                phi_cross_limits = []
                for Cij in Cij_arr:
                    i, j = Cij
                    # j feeds on i -> Cij > 0
                    if adj[i,j] == 1.0:
                        phi_lim = (np.pi/2, np.pi)
                    # i feeds on j -> Cij < 0
                    elif adj[j,i] == 1.0:
                        phi_lim = (0.0, np.pi/2)
                    # Some special cases where Cij != 0 for indirect interactions
                    else:
                        # Negative interaction btwn predators in exploitative -> Cij > 0
                        if (job.sp.module == 'exploitative') and (Cij in [(1,2), (2,1)]):
                            phi_lim = (np.pi/2, np.pi)
                        # Negative interaction btwn prey in apparent -> Cij > 0
                        #'''Not very confident about this assumption'''
                        elif (job.sp.module == 'apparent') and (Cij in [(0,1), (1,0)]):
                            phi_lim = (np.pi/2, np.pi)
                        # For all cases besides those specified below, i.e. Cuw and Cwu in chain module, Cij = 0
                        else:
                            phi_lim = (np.nan, np.nan)
                    phi_cross_limits.append(phi_lim)

            # Get the stored data across dispersal parameterizations
            if job.sp['local_stability'] != 'unstable':
                if job.sp['method'] == 'symbolic':
                    ddi = sum([np.array(job.data[Cij_key]['omega_integrand/'+key]) for key in ['wav', 'st']])
                elif job.sp['method'] == 'numeric':
                    ddi = np.array(job.data[Cij_key]['omega_integrand/ddi'])
                else: sys.exit('Invalid critical kappa computation method')

            # First get the constrained value of omega
            # Store nan for any cases with invalid constraints
            if (len(phi_cross_limits) != 0) and np.any(np.all(np.isnan(phi_cross_limits), axis=1)):
                omega_constrained = np.nan
            # Otherwise calculate omega_ddi with the constraints above
            else:
                if job.sp['local_stability'] == 'unstable':
                    omega_constrained = 0.0
                else:
                    # Combine all of the constraints
                    all_constraints = [] 
                    phi_limits = phi_cross_limits + phi_diag_limits
                    for i, limit in enumerate(phi_limits):
                        phi_i = np.array(job.data[Cij_key]['phi_'+str(i+1)])
                        all_constraints.append(phi_i > limit[0])
                        all_constraints.append(phi_i < limit[1])
                    constraint = np.all(all_constraints, axis=0)
                    # Get the mean value of the data within the constraints
                    if sum(constraint) != 0:
                        omega_constrained = np.mean(ddi[constraint])
                    # If there's no data within the contraints, something is wrong 
                    else:
                        sys.exit('No data found within the specified constraints')
            # Finally get the unconstrained value
            if job.sp['local_stability'] == 'unstable':
                omega_unconstrained = 0.0
            else:
                omega_unconstrained = np.mean(ddi)
            
            # Store in job document
            job.doc['omega_constrained'][Cij_key] = omega_constrained 
            job.doc['omega_unconstrained'][Cij_key] = omega_unconstrained 

if __name__ == "__main__":
    FlowProject().main()
