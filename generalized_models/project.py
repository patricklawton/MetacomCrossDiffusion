import signac
from flow import FlowProject
import numpy as np

project = signac.get_project()
msf_group = FlowProject.make_group(name='msf')

@FlowProject.label
def P_calculated(job):
    return job.isfile('P_matrix.h5')

@msf_group
@FlowProject.post(P_calculated)
@FlowProject.operation
def calc_P(job):
    # Read in params from statepoint
    sp = job.sp
    gamma = list(sp['gamma'].values())
    mu = list(sp['mu'].values())
    phi = list(sp['phi'].values())
    psi = list(sp['psi'].values())
    sigma = list(sp['sigma'].values())
    
    # Read in other params from job data
    with job.data:
        alpha = np.array(job.data['alpha'])
        rho = np.array(job.data['rho'])
        beta = np.array(job.data['beta'])
        chi = np.array(job.data['chi'])
        lambdas = np.array(job.data['lambdas'])
    
    # Get the number of species from the adjacency matrix
    sd_fn = project.fn('shared_data.h5')
    with signac.H5Store(sd_fn).open(mode='r') as sd:
        adj = np.array(sd['adj_mats'])[sp['wi']]
    N = len(adj)

    # Compute the patches' Jacobian matrix P
    P = np.zeros((N,N))
    for n in range(N):
        for i in range(N):
            dsum = 0; #For calculating the sum in the loss by predation part for mutualistic effects
            if (i != n): #Off diagonal
                for m in range (N):
                    dsum +=  beta[m][n] * lambdas[m][i] * (gamma[m] - 1.) * chi[m][i]
                # Gain by predation -  loss by predation
                P[n][i] = alpha[n]*( rho[n]*gamma[n]*chi[n][i]*lambdas[n][i] - sigma[n]*(beta[i][n]*psi[i] + dsum) )
            
            else: #Diagonal
                for m in range(N):
                    dsum  += beta[m][i] * lambdas[m][i] * ( (gamma[m] - 1.) * chi[m][i]  + 1. )
                # Primary production + gain by predation  - mortality - loss by predation
                P[i][i] = alpha[i]*( (1.-rho[i])*phi[i] + rho[i]*(gamma[i]*chi[i][i]*lambdas[i][i] + psi[i]) - (1.-sigma[i])*mu[i] - sigma[i]*(beta[i][i]*psi[i] + dsum) )

    # Write local Jacobian to file
    with job.stores.P_matrix as P_matrix:
            P_matrix['P'] = P

@FlowProject.label
def msf_calculated(job):
    return job.isfile('msf_array.h5')

@msf_group
@FlowProject.pre.after(calc_P)
@FlowProject.post(msf_calculated)
@FlowProject.operation
def calc_msf(job):
    # Load sp plus kappa and adj from shared data
    sp = job.sp
    sd_fn = project.fn('shared_data.h5')
    with signac.H5Store(sd_fn).open(mode='r') as sd:
        kap_arr = np.array(sd['kappa'])
        adj = np.array(sd['adj_mats'])[sp['wi']]
    
    # Load P matrix
    with job.stores.P_matrix:
        P = np.array(job.stores.P_matrix.P)
    
    # Make C matrix
    C = np.zeros(adj.shape)
    C[sp.di][sp.di] = sp.delta

    # Construct msf and write to file
    msf = np.empty(len(kap_arr))
    for k, kap in enumerate(kap_arr):
        J = P - kap*C
        J_evals, Jevecs = np.linalg.eig(J)
        msf[k] = max(J_evals.real)
    with job.stores.msf_array as msf_array:
        msf_array['msf'] = msf

    # Write msf info to job doc
    job.doc['nonspatial_stable'] = msf[0] < 0
    job.doc['msf_crosseszero'] = np.all([np.any(msf<0), np.any(msf>0)])

@FlowProject.pre.after(calc_msf)
@FlowProject.post(lambda job: 'gamma_diff' in list(job.doc.keys()))
@FlowProject.operation
def store_gamma_diff_in_doc(job):
    # Load sp plus adj from shared data
    sp = job.sp
    sd_fn = project.fn('shared_data.h5')
    with signac.H5Store(sd_fn).open(mode='r') as sd:
        adj = np.array(sd['adj_mats'])[sp['wi']]

    # Get difference, zero if n is producer (apparent web)
    if (sum(adj[:,1])>0) and (sum(adj[:,2])>0):
        gamma_diff = abs(sp['gamma']['2'] - sp['gamma']['1'])
    else:
        gamma_diff = 0.

    job.doc['gamma_diff'] = gamma_diff

if __name__ == "__main__":
    FlowProject().main()
