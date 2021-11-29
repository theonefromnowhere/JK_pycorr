import os
import tempfile
from pycorr import TwoPointCounter
import numpy as np
import math
from pycorr import TwoPointCounter
import h5py
from tqdm import trange
from nbodykit.lab import *
from mpi4py import MPI
import time
import configparser
import argparse


def formatBcast(arr):
    temp = []
    for i in range(len(arr)):
        temp.extend(arr[i])
    return temp



def CountPairs(dset1,dset2,edges,weights1=None,weights2=None,nthreads=32):
    #print('Computing counts')
    D1D2 = TwoPointCounter('smu', edges, positions1=dset1.T,positions2=dset2.T,weights1=weights1,weights2=weights2,
                       engine='corrfunc', nthreads=nthreads)
    return D1D2.wcounts



if __name__ == "__main__":

    desc = "Analysis of the PS"
    parser = argparse.ArgumentParser(description=desc)

    h = 'conf file'
    parser.add_argument('k', type=str, help=h)
    ns=parser.parse_args()
    config = ns.k
    conf = configparser.ConfigParser()
    conf.read(config)
    data = (conf.get('Input','data'))
    randoms = (conf.get('Input','randoms'))
    output = conf.get('Output','output')
    omega_m = float(conf.get('Input','omega_m'))
    h = float(conf.get('Input','h'))
    sigma8 = float(conf.get('Input','sigma8'))
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size_comms = comm.size
    
    min_s = float(conf.get('Input','min_s'))
    max_s = float(conf.get('Input','max_s'))
    num_s = int(conf.get('Input','num_s'))
    
    num_mu = int(conf.get('Input','num_mu'))
    
    lab_ra1 = conf.get('Input','RA1')
    lab_dec1 = conf.get('Input','DEC1')
    lab_z1 = conf.get('Input','Z1')
    lab_w1 =  conf.get('Input','weights1')
    lab_jk1 =  conf.get('Input','reg_ind1')
    
    lab_ra2 = conf.get('Input','RA2')
    lab_dec2 = conf.get('Input','DEC2')
    lab_z2 = conf.get('Input','Z2')
    lab_w2 =  conf.get('Input','weights2')
    lab_jk2 =  conf.get('Input','reg_ind2')

    print('Setting up cosmology')

    if (rank==0):
        temp_time = time.time()

    cosmo = cosmology.Cosmology(sigma8=sigma8,h=h).match(Omega0_m=omega_m)

    os.environ['NUMEXPR_MAX_THREADS'] = '32'

    nthreads=32

    print('Reading datasets')

    f = h5py.File(data, "r")
    z=f[lab_z1][...]
    ra = f[lab_ra1][...]
    dec = f[lab_dec1][...]
    w_fkp = f[lab_w1][...]
    jk_ind = f[lab_jk1][...]
    f.close()

    print('Data length ', len(z))


    f = h5py.File(randoms, "r")
    z_r=f[lab_z2][...]
    ra_r = f[lab_ra2][...]
    dec_r = f[lab_dec2][...]
    w_fkp_r = f[lab_w2][...]
    jk_ind_r = f[lab_jk2][...]
    f.close()


    

    data_pos = transform.SkyToCartesian(ra, dec, z, cosmo=cosmo)
    rand_pos = transform.SkyToCartesian(ra_r, dec_r, z_r, cosmo=cosmo)



    edges = (np.linspace(min_s, max_s, num_s), np.linspace(0, 1., num_mu))
    shp = [len(edges[0])-1,len(edges[1])-1]
    jk_max = np.max(jk_ind)+1
    corrs_jk=[]
    delta_jk = jk_max/size_comms
    



    print('Starting counts from ',math.ceil(delta_jk*rank),' to ',math.ceil((delta_jk)*(rank+1))-1)
    DhDn_s = []
    DhDh_s =[]
    RhRn_s = []
    RhRh_s = []
    DnRh_s = []
    DhRn_s = []
    DhRh_s = []



    for i in range(math.ceil(delta_jk*rank),math.ceil((delta_jk)*(rank+1))):
        if(i>=jk_max):
            continue
        if(i<0):
            continue
        dt_wth_hole = data_pos[jk_ind!=i]
        rt_wth_hole = rand_pos[jk_ind_r!=i]
    
        dt_hole = data_pos[jk_ind==i]
        rt_hole = rand_pos[jk_ind_r==i]
    
        dw_hole = w_fkp[jk_ind==i]
        rw_hole = w_fkp_r[jk_ind_r==i]
    
        dw_wth_hole = w_fkp[jk_ind!=i]
        rw_wth_hole = w_fkp_r[jk_ind_r!=i]
    
        DhDn_s.append(CountPairs(dt_hole,dt_wth_hole,edges,dw_hole,dw_wth_hole,nthreads=nthreads))
        DhDh_s.append(CountPairs(dt_hole,dt_hole,edges,dw_hole,dw_hole,nthreads=nthreads))
    
        RhRn_s.append(CountPairs(rt_hole,rt_wth_hole,edges,rw_hole,rw_wth_hole,nthreads=nthreads))
        RhRh_s.append(CountPairs(rt_hole,rt_hole,edges,rw_hole,rw_hole,nthreads=nthreads))
    
        DnRh_s.append(CountPairs(dt_wth_hole,rt_hole,edges,dw_wth_hole,rw_hole,nthreads=nthreads))
        DhRn_s.append(CountPairs(dt_hole,rt_wth_hole,edges,dw_hole,rw_wth_hole,nthreads=nthreads))
        DhRh_s.append(CountPairs(dt_hole,rt_hole,edges,dw_hole,rw_hole,nthreads=nthreads))
        print(i," done by ", rank)

    DhDn_s = comm.gather(DhDn_s,root=0)
    DhDh_s = comm.gather(DhDh_s,root=0)

    RhRn_s = comm.gather(RhRn_s,root=0)
    RhRh_s = comm.gather(RhRh_s,root=0)

    DnRh_s = comm.gather(DnRh_s,root=0)
    DhRn_s = comm.gather(DhRn_s,root=0)
    DhRh_s = comm.gather(DhRh_s,root=0)

    if(rank==0):
        np.save(output+'_edges',edges)
        DhDn_s = formatBcast(DhDn_s)
        DhDh_s = formatBcast(DhDh_s)

        RhRn_s = formatBcast(RhRn_s)
        RhRh_s = formatBcast(RhRh_s)

        DhRn_s = formatBcast(DhRn_s)
        DhRh_s = formatBcast(DhRh_s)
        DnRh_s = formatBcast(DnRh_s)
        print('Computing DD,DR,RR ')
    
        DD = np.zeros(np.shape(DhDh_s[0]))
        DR = np.zeros(np.shape(DhDh_s[0]))
        RR = np.zeros(np.shape(DhDh_s[0]))
        DD = np.sum(DhDh_s+DhDn_s,axis=0)
        RR = np.sum(RhRh_s+RhRn_s,axis=0)
        DR = np.sum(DnRh_s+DhRh_s,axis=0)
        
        print('Computing jackknife ')
    
        for i in trange(jk_max):
            dt_wth_hole = data_pos[jk_ind!=i]
            rt_wth_hole = rand_pos[jk_ind_r!=i]
    
            dt_hole = data_pos[jk_ind==i]
            rt_hole = rand_pos[jk_ind_r==i]
    
            dw_hole = w_fkp[jk_ind==i]
            rw_hole = w_fkp_r[jk_ind_r==i]
    
            dw_wth_hole = w_fkp[jk_ind!=i]
            rw_wth_hole = w_fkp_r[jk_ind_r!=i]
        
            DhDn=DhDn_s[i]
            DhDh =DhDh_s[i]
        
            DnRh=DnRh_s[i]
            DhRn=DhRn_s[i]
            DhRh = DhRh_s[i]
        
            RhRn=RhRn_s[i]
            RhRh=RhRh_s[i]
        
            DD_jk = (DD-2*DhDn-DhDh)/(np.sum(dw_wth_hole))/(np.sum(dw_wth_hole)+1)
            DR_jk = (DR-DnRh - DhRn- DhRh)/(np.sum(rw_wth_hole))/(np.sum(dw_wth_hole))
            RR_jk = (RR-2*RhRn-RhRh)/(np.sum(rw_wth_hole))/(np.sum(rw_wth_hole)+1)
            cf = ((DD_jk-2*DR_jk+RR_jk)/RR_jk)
        
            np.save(output+'_'+str(i),cf)
    
        DD = DD /np.sum(w_fkp)/(np.sum(w_fkp)+1)
        DR = DR /np.sum(w_fkp)/(np.sum(w_fkp_r))
        RR = RR /np.sum(w_fkp_r)/(np.sum(w_fkp_r)+1)
    
        cf = ((DD-2*DR+RR)/RR)
        np.save(output+'_cf',cf)
        
    
        print('Finished in ', time.time()-temp_time)
        
    
    


    
    