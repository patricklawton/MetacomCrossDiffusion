import signac
import numpy as np

project = signac.get_project()
#
#test = project.find_jobs({'R': 40, 'wi': 0, 'di': 0, 'delta':1})
#print(test)
#print(len(test))

with signac.H5Store(project.fn('shared_data.h5')).open(mode='r') as shared_data:
    print('shared_data')
    print(list(shared_data.keys()))
    print(np.array(shared_data['adj_mats']))
    print(np.array(shared_data['kappa']))
    print(list(shared_data.prngs.keys()))
    for prng in list(shared_data.prngs.keys()):
        print (shared_data.prngs[prng])
        print (np.array(shared_data.prngs[prng]))
    #print(np.array(share
