import numpy as np
import signac as sg
from itertools import product
from global_functions import get_cross_limits
import os
from tqdm import tqdm

project = sg.get_project()

# Read in shared data 
sd_fn = project.fn('shared_data.h5')
with sg.H5Store(sd_fn).open(mode='r') as sd:
    adj_mats = np.array(sd['adj_mats']) #Adjacency matricies
    modules = np.array([i.decode() for i in sd['modules']])
    C_offdiags = list(sd['C_offdiags'])
    cross_labels = np.array([i.decode() for i in sd['cross_labels']])
n_cross_arr = np.arange(0, len(C_offdiags)+1)
C_elements = np.array(C_offdiags + [[i,i] for i in range(3)])

# Initialize data
out_fn = 'local_robustness.h5'
constraint_keys = ['unconstrained', 'constrained']
robustness_data = {}
for module in modules:
    robustness_data[module] = {key: {} for key in constraint_keys}

# Separate data by motif
for module_idx, constraint_key in tqdm(product(range(len(modules)), constraint_keys)):
    adj = adj_mats[module_idx]
    # First filter for the module
    jobs_filter = {'module': modules[module_idx]}
    num_jobs = len(project.find_jobs(jobs_filter))
    # Get the fraction of unstable parameterizations
    jobs_filter.update({'local_stability': 'unstable'})
    fraction_unstable = len(project.find_jobs(jobs_filter)) / num_jobs
    robustness_data[modules[module_idx]].update({'fraction_unstable': fraction_unstable})
    # Filter out any infeasible parameterizations
    jobs_filter.update({'local_stability': {'$ne': 'infeasible'}})
    jobs = project.find_jobs(jobs_filter)

    # Store non-spatial robustness for each value of n_cross
    omega_module = {'value': [], 'stdev': []}
    for n_cross in tqdm(n_cross_arr):
        # Get the number of spatial parameterizations
        num_samples = int((2**n_cross + 3)*1e2) if n_cross != 0 else int(3*1e2)
        # Get the labels for cross diffusive scenarios 
        if n_cross == 0:
            label_indices = [label == 'diag' for label in cross_labels]
        else:
            label_indices = [(len(label.split(',')) == n_cross) and (label != 'diag') for label in cross_labels]
        labels = cross_labels[np.nonzero(label_indices)[0]]
        # Get the average for each cross diffusive scenario
        omega_n_cross = {'value': [], 'stdev': []}
        for cross_label in labels:
            if (constraint_key == 'constrained') and (n_cross != 0):
                # Get the cross diffusive limits for the constrained case
                Cij_arr = [(int(e[0]), int(e[1])) for e in cross_label.split(',')]
                #C_nonzero = [list(Cij) for Cij in Cij_arr]
                cross_limits = [get_cross_limits(ij, modules[module_idx], adj) for ij in Cij_arr]
            # Get omega for each spatial (dispersal) sample
            omega_cross_scenario = {'value': [], 'stdev': []}
            for spatial_sample_idx in tqdm(range(num_samples)): 
                # If constraining, check that spatial sample within constraints
                if (constraint_key == 'constrained') and (n_cross != 0):
                    with sg.H5Store(sd_fn).open(mode='r') as sd:
                        all_constraints = [] 
                        for i, limit in enumerate(cross_limits):
                            coord_idx = np.nonzero([list(ij) == list(Cij_arr[i]) for ij in C_elements])[0]
                            coord_i = float(sd['cart_coord_samples'][coord_idx, spatial_sample_idx])
                            all_constraints.append(coord_i > limit[0])
                            all_constraints.append(coord_i < limit[1])
                    # If constraints are not met, skip this sample
                    if not np.all(all_constraints):
                        continue
                # Check phase data at each local (trophic) parameterization
                omega_spatial_sample = np.zeros(len(jobs))
                for job_idx, job in enumerate(jobs):
                    with job.data:
                        # If locally stable, check for ddi
                        if job.sp['local_stability'] == 'stable':
                            omega_spatial_sample[job_idx] = job.data['ddi'][cross_label][spatial_sample_idx]
                        # If locally unstable, no ddi 
                        elif job.sp['local_stability'] == 'unstable':
                            omega_spatial_sample[job_idx] = False
                omega = np.mean(omega_spatial_sample)
                omega_cross_scenario['value'].append(omega)
                squared_avg = sum(omega_spatial_sample) / len(omega_spatial_sample)
                avg_squared = sum(omega_spatial_sample) / len(omega_spatial_sample)**2  
                stdev = (1/np.sqrt(len(omega_spatial_sample))) * np.sqrt(squared_avg - avg_squared) 
                omega_cross_scenario['stdev'].append(stdev)
            omega_n_cross['value'].append(np.mean(omega_cross_scenario['value']))
            stdev = (1/len(omega_cross_scenario['value'])) * np.sqrt(sum(np.square(omega_cross_scenario['stdev'])))
            omega_n_cross['stdev'].append(stdev)
        omega_module['value'].append(np.mean(omega_n_cross['value']))
        stdev = (1/len(omega_n_cross['value'])) * np.sqrt(sum(np.square(omega_n_cross['stdev'])))
        omega_module['stdev'].append(stdev)
    robustness_data[modules[module_idx]][constraint_key]['mean'] = omega_module['value']
    robustness_data[modules[module_idx]][constraint_key]['stdev'] = omega_module['stdev']

if not os.path.isfile(out_fn):
    with sg.H5Store(out_fn).open(mode='w') as robustness_file:
        for module in modules:
            robustness_file[module+'/fraction_unstable'] = robustness_data[module]['fraction_unstable']
            for key in constraint_keys:
                robustness_file[module+'/'+key] = np.array(robustness_data[module][key]['mean'])
                robustness_file[module+'/'+key] = np.array(robustness_data[module][key]['stdev'])
