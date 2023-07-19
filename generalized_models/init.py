import signac
import numpy as np
from scipy.linalg import solve
from tqdm import tqdm
from tqdm.contrib.itertools import product
import timeit
start = timeit.default_timer()

# Function to get trophic levels (for alpha calculation)
def getTrophicLevels(adj):
    flws = adj.copy()
    for to in range(len(flws)):
        totflow=sum(flws[:,to])
        if totflow > 0:
            flws[:,to]=flws[:,to]/totflow
        flws[to,to]=-1
    return solve(flws.T,-1.*np.ones(len(flws)))-1.

# Initialize signac project (i.e. create workspace dir)
project = signac.init_project()

# Ranges for randomly sampled parameters
prngs = {
    'gamma': np.array([0.5,1.5]),
    'mu': np.array([1.,1.15]),
    'phi': np.array([0.,1.]),
    'psi': np.array([0.5,1.5]),
    'sigma': np.array([0.5,1.]),
    'chi_high': np.array([0.01,0.99]), # Gain from higher index prey
    'beta_low': np.array([0.01,0.99]) # Loss to higher index predator
    }

# Constants across all statepoints
kappa_max = 100 #Max Laplacian eigenvalue used for MSF
kappa_step = 1
reps = 1e3 #Number of samples within each non-random param combination

# Parameters to combine
R_arr = np.array([1/0.025])  #Average predator/prey body mass ratios
adj_mats = np.array([    # Adjacency matrices (i.e. web structures)
    [[0.,1,1],[0,0,1],[0,0,0]],
    [[0,1,1],[0,0,0],[0,0,0]],
    [[0,1,0],[0,0,1],[0,0,0]],
    [[0,0,1],[0,0,1],[0,0,0]]
    ])
web_indices = np.arange(len(adj_mats)) #Index for each Niche web structure
d_indices = np.arange(len(adj_mats[0])) #Indices for dispersing species
deltas = np.array([1.]) #Dispersal 'rates' for dispersing species

# Store shared data in a project level file
with signac.H5Store(project.fn('shared_data.h5')).open(mode='w') as shared_data:
    for key, val in prngs.items():
        h5_key = 'prngs/' + key
        shared_data[h5_key] = val
    shared_data['kappa'] = np.arange(0, float(kappa_max)+kappa_step, kappa_step)
    shared_data['adj_mats'] = adj_mats
    shared_data['webnms'] = ["Omnivory","Exploitative","Chain","Apparent"]
    shared_data['specnms'] = ['r','n','p']

# Loop over combs of params not being randomly sampled
for params in product(R_arr, web_indices, d_indices, deltas):
    # Create/reassign dictionary for statepoints
    R, wi, di, delta = params
    adj = adj_mats[wi]
    sp = {'R': R, 'wi': wi, 'di': di, 'delta': delta}

    # Get some metadata for this statepoint
    toppred = [i for i in range(len(adj)) if sum(adj[i,:])==adj[i,i]]
    alpha = R**(-getTrophicLevels(adj)/4) #M=R^TL -> alp=M^(-1/4)=R^(-TL/4)  
    rho = np.array([0 if sum(adj[:,i])==0 else 1 for i in range(len(adj))])
    lambdas = np.ones((len(adj), len(adj)))

    # Get the number of samples to generate
    find_jobs = project.find_jobs({'R': float(R), 'wi': int(wi), 
                                   'di': int(di), 'delta': float(delta)})
    reps_left = max([0, reps - len(find_jobs)]) 

    # Create unique statepoint for each sample
    for r in tqdm(range(int(reps_left))):
        # Sample each parameter, for each species as needed
        for p, rng in prngs.items():
            if (p == 'beta_low') or (p == 'chi_high'):
                sp[p] = np.random.uniform(min(rng), max(rng))
                # Create beta/chi matrix to write to job data
                pmat = adj.transpose().copy()
                for i in range(len(adj)):
                    # Adjust chi if multiple prey for 1 pred
                    if (sum(adj[:,i]) > 1) and (p == 'chi_high'):
                        prey_i = [k for k in range(len(adj)) if adj[k][i]!=0]
                        pmat[i][min(prey_i)] = 1 - sp[p] # For lower index prey
                        pmat[i][max(prey_i)] = sp[p]     # For higher index prey
                    # Adjust beta if multiple pred for 1 prey
                    if (sum(adj[i,:]) > 1) and (p == 'beta_low'):
                        pred_i = [k for k in range(len(adj)) if adj[i][k]!=0]
                        pmat[min(pred_i)][i] = 1 - sp[p] # For lower index pred
                        pmat[max(pred_i)][i] = sp[p]    # For higher index pred
                if p == 'beta_low':
                    beta = pmat
                    if np.any([sum(adj[i]) > 1 for i in range(len(adj))]) == False:
                        sp[p] = None
                else:
                    chi = pmat
                    if np.any([sum(adj[:,i]) > 1 for i in range(len(adj))]) == False:
                        sp[p] = None
            else:
                p_dict = {}
                for spec_index in range(len(adj)):
                    if (p=='sigma') and (spec_index in toppred):
                        p_dict[str(spec_index)] = 0.
                    else:
                        p_dict[str(spec_index)] = np.random.uniform(min(rng), max(rng))
                sp[p] = p_dict

        # Initialize job with statepoint
        job = project.open_job(sp)
        job.init()

        # Store some data we will read in later
        job.data['alpha'] = alpha
        job.data['rho'] = rho
        job.data['beta'] = beta
        job.data['chi'] = chi
        job.data['lambdas'] = lambdas

stop = timeit.default_timer()
print('Time: ', stop - start, 's')
