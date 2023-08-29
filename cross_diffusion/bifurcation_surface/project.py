import numpy as np
import sympy as sy
import signac as sg
from flow import FlowProject
from itertools import product, combinations, accumulate, groupby
import sys
import timeit 
from global_functions import spherical_to_cartesian, get_cross_limits

# Open up signac project
project = sg.get_project()

# Read in some things from project data
sd_fn = project.fn('shared_data.h5')
with sg.H5Store(sd_fn).open(mode='r') as sd:
    adj_mats = np.array(sd['adj_mats']) #Adjacency matricies
    modules = np.array([i.decode() for i in sd['modules']]) #Interaction module labels
    cross_labels = np.array([i.decode() for i in sd['cross_labels']]) #Cross-diffusion scenario labels
    C_offdiags = list(sd['C_offdiags'])
n_cross_arr = [i for i in range(len(C_offdiags) + 1)] #Number of nonzero cross dispersal elements
C_elements = np.array(C_offdiags + [[i,i] for i in range(3)])

def get_integrand_numeric(job, J, cross_label, cross_comb, C_offdiags, num_samples, sd_fn):
    # Define array of kappa values to compute eigenvalues at
    kmax, step = (100, 0.1)
    kappas = np.arange(0, kmax+step, step)
    max_kap_crits = 6 #3 roots for stationary condition, 3 roots for oscillatory condition
    kappa_vec = np.ones((3,3)) * np.reshape(kappas, (len(kappas),1,1))
    kappa_vec = np.broadcast_to(kappa_vec, (1,kappa_vec.shape[0],3,3))
    # Construct connectivity matricies from random samples
    C_nonzero = [list(Cij) for Cij in cross_comb] + [[i,i] for i in range(3)]
    indices = np.nonzero([list(ij) in C_nonzero for ij in C_elements])[0]
    C_vec = np.zeros((num_samples,1,3,3))
    with sg.H5Store(sd_fn).open(mode='r') as sd:
        cart_coord_samples = np.array(sd['cart_coord_samples'][indices, :num_samples])
    for idx, (i, j) in enumerate(C_nonzero):
        C_vec[:, 0, i, j] = cart_coord_samples[idx, :]
    # Construct array of community jacobians across dispersal parameterizations and kappa values
    M_vec = np.broadcast_to(J, (C_vec.shape[0],kappa_vec.shape[1],3,3)) - kappa_vec * C_vec
    # Find if and where critical kappas occur
    kappa_cs = []
    any_positive = np.any(np.linalg.eigvals(M_vec).real > 0, axis=2)
    for vec in any_positive:
        critical_indices = list(accumulate(sum(1 for _ in g) for _,g in groupby(vec)))[:-1]
        critical_kappas = kappas[critical_indices]
        critical_kappas = np.pad(critical_kappas.astype(float),
                                (0, max_kap_crits-len(critical_indices)), 
                                mode='constant', constant_values=(np.nan,))
        kappa_cs.append(critical_kappas)
    # Store data
    #job.data[cross_label + '/ddi'] = np.any(any_positive, axis=1)
    #job.data[cross_label + '/kappa_cs'] = np.array(kappa_cs)
    job.data['ddi/' + cross_label] = np.any(any_positive, axis=1)
    job.data['kappa_cs/' + cross_label] = np.array(kappa_cs)

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
        '''Fix: read in sample density from shared data'''
        num_samples = int((2**n_cross + 3)*1e2) if n_cross != 0 else int(3*1e2)
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
            
            # Generate surface numerically
            if sp['method'] == 'numeric':
                get_integrand_numeric(job, J, cross_label, cross_comb, C_offdiags, num_samples, sd_fn)
            # Generate surface symbolically 
            elif sp['method'] == 'symbolic':
                # Initialize data for this cross-diffusive scenario
                data = {'kappa_cs': {'wav': [], 'st': []},
                        'omega_integrand': {'wav': [], 'st': []}}
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
                C_nonzero = [list(Cij) for Cij in cross_comb] + [[i,i] for i in range(3)]
                C_k = [C[i,j] for i, j in C_nonzero]
                # Construct metacommunity jacobian with symbolic dispersal entries
                M = sy.Matrix(J - kappa*C)
                # Get oscillatory and stationary pattern-forming instability conditions
                p = M.charpoly(lamda) #characteristic polynomial
                p_coeffs = p.all_coeffs()
                I_wav = p_coeffs[3] - p_coeffs[1]*p_coeffs[2] #oscillatory
                I_st = p_coeffs[3] #stationary
                # Get dispersal samples 
                indices = np.nonzero([list(ij) in C_nonzero for ij in C_elements])[0]
                with sg.H5Store(sd_fn).open(mode='r') as sd:
                    cart_coord_samples = np.array(sd['cart_coord_samples'][indices, :num_samples])
                # Solve for critical kappas (symbolically) at each coordinate
                for cart_coord_sample in cart_coord_samples:
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
                # Store all job data
                for key, item in data.items():
                    if isinstance(item, dict):
                        for sub_key, sub_item in item.items():
                            job.data[key+'/'+sub_key+'/'+cross_label] = np.array(sub_item)
                    else:
                        job.data[key+'/'+cross_label] = np.array(item)
            else:
                sys.exit('Invalid critical kappa computation method')

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
        job.doc['stdev_constrained'] = {}
        job.doc['stdev_unconstrained'] = {}
        
        # Select adjacency matrix
        adj = adj_mats[modules==job.sp.module][0]

        # Loop over cross diffusion scenarios, stored as keys in job data
        for Cij_key in cross_labels:
            #print('Cij:', Cij_key)
            # Set constraints on the elements of C
            # Diagonal elements (phi_(n-2), phi_(n-1)) always > 0
            diag_limits = [(0,1), (0,1), (0,1)]
            if Cij_key == 'diag':
                C_nonzero = [[i,i] for i in range(3)]
                '''Fix: read in sample density from shared data'''
                num_samples = int(3*1e2)
                cross_limits = []
            # Constrain off-diagonals to opposite sign of species interaction
            else:
                # Convert key to a list of off-diagonal C element indices
                Cij_arr = [(int(e[0]), int(e[1])) for e in Cij_key.split(',')]
                C_nonzero = [list(Cij) for Cij in Cij_arr] + [[i,i] for i in range(3)]
                cross_limits = [get_cross_limits(ij, job.sp.module, adj) for ij in Cij_arr]
                '''Fix: read in sample density from shared data'''
                num_samples = int((2**len(Cij_arr) + 3)*1e2) 

            # Get the stored data across dispersal parameterizations
            if job.sp['local_stability'] != 'unstable':
                if job.sp['method'] == 'symbolic':
                    ddi = sum([np.array(job.data[key][Cij_key]) for key in ['wav', 'st']])
                elif job.sp['method'] == 'numeric':
                    ddi = np.array(job.data['ddi'][Cij_key])
                else: sys.exit('Invalid critical kappa computation method')

            # First get the constrained value of omega
            # Store nan for any cases with invalid constraints
            if (len(cross_limits) != 0) and np.any(np.all(np.isnan(cross_limits), axis=1)):
                omega_constrained = np.nan
                stdev_constrained = np.nan
            # Otherwise calculate omega_ddi with the constraints above
            else:
                if job.sp['local_stability'] == 'unstable':
                    omega_constrained = 0.0
                    stdev_constrained = 0.0
                else:
                    # Combine all of the constraints
                    with sg.H5Store(sd_fn).open(mode='r') as sd:
                        all_constraints = [] 
                        limits = cross_limits + diag_limits
                        for i, limit in enumerate(limits):
                            coord_idx = np.nonzero([list(ij) == C_nonzero[i] for ij in C_elements])[0]
                            coord_i = np.array(sd['cart_coord_samples'][coord_idx, :num_samples])
                            all_constraints.append(coord_i > limit[0])
                            all_constraints.append(coord_i < limit[1])
                        #print(all_constraints)
                        constraint = np.all(all_constraints, axis=0)[0]
                        #print(constraint)
                        # Get the mean value of the data within the constraints
                        if sum(constraint) != 0:
                            omega_constrained = np.mean(ddi[constraint])
                            squared_avg = sum(ddi[constraint]) / len(ddi[constraint])
                            avg_squared = sum(ddi[constraint])**2 / len(ddi[constraint])**2  
                            stdev_constrained = (1/np.sqrt(len(ddi[constraint]))) * np.sqrt(squared_avg - avg_squared) 
                            #sys.exit()
                        # If there's no data within the contraints, something is wrong 
                        else:
                            sys.exit('No data found within the specified constraints')
            # Finally get the unconstrained value
            if job.sp['local_stability'] == 'unstable':
                omega_unconstrained = 0.0
                stdev_unconstrained = 0.0
            else:
                omega_unconstrained = np.mean(ddi)
                squared_avg = sum(ddi) / len(ddi)
                avg_squared = sum(ddi) / len(ddi)**2  
                stdev_unconstrained = (1/np.sqrt(len(ddi))) * np.sqrt(squared_avg - avg_squared) 
            
            # Store in job document
            job.doc['omega_constrained'][Cij_key] = omega_constrained 
            job.doc['stdev_constrained'][Cij_key] = stdev_constrained
            job.doc['omega_unconstrained'][Cij_key] = omega_unconstrained 
            job.doc['stdev_unconstrained'][Cij_key] = stdev_unconstrained

if __name__ == "__main__":
    FlowProject().main()
