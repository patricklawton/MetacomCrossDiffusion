import numpy as np
import sympy as sy
import signac
from itertools import product

# Initialize signac project (i.e. create workspace dir)
project = signac.init_project()

# Parameters to combine for different statepoints
J_mats = np.array([
    [
        [0.471, -0.813, 0.0], 
        [5.552, 0.962, -0.911], 
        [0.0, 16.0, -4.0]
    ]
])
J_indices = np.array([0])
C_offdiags = np.array([(0,1), (0,2),
                      (1,0), (1,2),
                      (2,0), (2,1)])
N_ns = [1e2] #Avg density of samples per 0-pi/2 interval

# Store shared data in a project level file
with signac.H5Store(project.fn('shared_data.h5')).open(mode='w') as shared_data:
    shared_data['J_mats'] = J_mats
    shared_data['J_labels'] = ['hata_example_eco']

# Initialize workspace
for J_i, C_offdiag, N_n in product(J_indices, C_offdiags, N_ns):
    sp = {'J_i': J_i, 'C_offdiag': C_offdiag, 'N_n': N_n}
    job = project.open_job(sp)
    job.init()
