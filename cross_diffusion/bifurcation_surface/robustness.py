import numpy as np
import signac as sg
from itertools import product
from scipy.stats import skew
from global_functions import get_cross_limits, get_num_spatials
import os
from tqdm import tqdm
import sys

project = sg.get_project()

# Constants 
sd_fn = project.fn('shared_data.h5')
with sg.H5Store(sd_fn).open(mode='r') as sd:
    adj_mats = np.array(sd['adj_mats'])
    modules = np.array([i.decode() for i in sd['modules']])
    cross_labels = np.array([i.decode() for i in sd['cross_labels']])
    C_offdiags = list(sd['C_offdiags'])
    N_n = int(sd['N_n'])
n_cross_arr = np.arange(0, len(C_offdiags)+1)
# Define q values which yield mean (expected) values for each possible number of nonzero cross diffusive elements
q_expected_arr = np.linspace(0, 1, len(C_offdiags)+1)
C_elements = np.array(C_offdiags + [[i,i] for i in range(3)])
constraint_keys = ['unconstrained', 'constrained']
rob_keys = ['local', 'spatial', 'total']
stat_keys = ['robustness', 'variance', 'skewness']
outfn = 'robustness.h5'
overwrite = True

# Decide whether or not to run
file_check = os.path.isfile(outfn)
if os.path.isfile(outfn) and (overwrite == False):
    sys.exit('Delete saved data or set overwrite to True')
with sg.H5Store(outfn).open(mode='w') as output:
    # Get robustness data for each module
    for module_idx, module in enumerate(modules):
        # Initialize data
        data = {}
        for stat_key in stat_keys:
            data[stat_key] = {}
            for rob_key in rob_keys:
                if stat_key == 'robustness':
                    data[stat_key][rob_key] = {key: [[] for n in n_cross_arr] for key in constraint_keys}  
                elif (stat_key == 'variance') and (rob_key in ['local', 'spatial']):
                    data[stat_key][rob_key] = {key: [] for key in constraint_keys}
                elif (stat_key == 'variance') and (rob_key == 'total'):
                    ...
                elif (stat_key == 'skewness') and (rob_key in ['local', 'spatial']):
                    data[stat_key][rob_key] = {key: [] for key in constraint_keys}
        # Select adjacency matrix
        print('module', module)
        adj = adj_mats[modules==module][0]
        # Filter for feasible trophic parameterizations in desired module
        all_locals = project.find_jobs({'local_stability': 'stable', 'module': module})
        num_local = len(all_locals)
        for n_cross in tqdm(n_cross_arr):
            # Get the number of spatial parameterizations
            num_spatial = get_num_spatials(n_cross, sample_density=N_n)
            # Get the labels for each cross scenario
            cross_labels_q = []
            for cross_idx, label in enumerate(cross_labels):
                if (label == 'diag') and (n_cross == 0):
                    cross_labels_q.append(label)
                elif (label != 'diag') and (len(label.split(',')) == n_cross):
                    cross_labels_q.append(label)
            # Loop over cross diffusive scenarios
            for cross_idx, cross_label in tqdm(enumerate(cross_labels_q)):
                print('cross_label', cross_label)
                # Get the spatial parameter limits for the relevant constraints
                if n_cross != 0:
                    cross_arr = [(int(e[0]), int(e[1])) for e in cross_label.split(',')]
                else:
                    cross_arr = []
                C_nonzero = [list(ij) for ij in cross_arr] + [[i,i] for i in range(3)]
                cross_limits = [get_cross_limits(ij, adj) for ij in cross_arr]
                # Skip any cross scenarios with invalid constraints
                if (len(cross_limits) != 0) and np.any(np.all(np.isnan(cross_limits), axis=1)):
                    continue
                diag_limits = [(0,1), (0,1), (0,1)]
                limits = cross_limits + diag_limits

                # Find spatial samples satisfying constraints
                all_constraints = []
                with sg.H5Store(sd_fn).open(mode='r') as sd:
                    for i, limit in enumerate(limits):
                        coord_idx = np.nonzero([list(ij) == C_nonzero[i] for ij in C_elements])[0]
                        coord_i = np.array(sd['cart_coord_samples'][coord_idx, :num_spatial])
                        all_constraints.append(coord_i > limit[0])
                        all_constraints.append(coord_i < limit[1])
                constraint = np.all(all_constraints, axis=0)[0]
                spatial_indices = np.nonzero(constraint)[0]

                # Initialize empty matrix of dim (spatial samples X feasible local samples) for phase data;
                # boolean because only two linearly independent phases
                phase_data = np.zeros((num_local, num_spatial))
                for local_idx, local in tqdm(enumerate(all_locals)):
                    with local.data:
                        # Get the phase data at this trophic sample
                        if local.sp.local_stability == 'stable':
                            phase_data[local_idx][:] = np.array(local.data['ddi'][cross_label])
                        elif local.sp.local_stability == 'unstable':
                            phase_data[local_idx][:] = np.zeros(num_spatial)

                # Calculate local and spatial robustness
                for rob_idx, rob_key in enumerate(['local', 'spatial']):
                    data['robustness'][rob_key]['unconstrained'][n_cross].extend(np.mean(phase_data, axis=rob_idx))
                    data['robustness'][rob_key]['constrained'][n_cross].extend(np.mean(phase_data[:,spatial_indices], axis=rob_idx))
                data['robustness']['total']['unconstrained'][n_cross].append(np.mean(phase_data))
                data['robustness']['total']['constrained'][n_cross].append(np.mean(phase_data[:,spatial_indices]))
            # Store robustness distributions
            for rob_key, cons_key in product(['local', 'spatial'], constraint_keys):
                data_key = '{}/robustness/{}/{}/{}'.format(module, rob_key, cons_key, str(n_cross))
                output[data_key] = np.array(data['robustness'][rob_key][cons_key][n_cross])
            # Take the average over cross scenarios for the total robustness
            for cons_key in constraint_keys: 
                data['robustness']['total'][cons_key][n_cross] = np.mean(data['robustness']['total'][cons_key][n_cross])

            # Calculate variance and skewness
            for rob_key, cons_key in product(['local', 'spatial'], constraint_keys):
                variance = np.var(data['robustness'][rob_key][cons_key][n_cross])
                data['variance'][rob_key][cons_key].append(variance) 
                skewness = skew(data['robustness'][rob_key][cons_key][n_cross], nan_policy='raise')
                data['skewness'][rob_key][cons_key].append(skewness) 

        for stat_key, cons_key in product(stat_keys, constraint_keys):
            if stat_key == 'robustness':
                data_key = '{}/{}/total/{}'.format(module, stat_key, cons_key)
                output[data_key] = np.array(data[stat_key]['total'][cons_key])
            elif stat_key == 'variance':
                for rob_key in ['local', 'spatial']:
                    data_key = '{}/{}/{}/{}'.format(module, stat_key, rob_key, cons_key)
                    output[data_key] = np.array(data[stat_key][rob_key][cons_key])
            elif stat_key == 'skewness':
                for rob_key in ['local', 'spatial']:
                    data_key = '{}/{}/{}/{}'.format(module, stat_key, rob_key, cons_key)
                    output[data_key] = np.array(data[stat_key][rob_key][cons_key])

## Now write the data over all modules
#with sg.H5Store(outfn).open(mode='r+') as output:
#    ...
