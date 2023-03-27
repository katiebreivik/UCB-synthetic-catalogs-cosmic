import numpy as np
import pandas as pd
from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.evolve import Evolve

initC = pd.read_csv('initial_grid_parameters.txt', delimiter=',', header=1, names=['mass_1', 'mass_2', 'porb'])

initBin = InitialBinaryTable.InitialBinaries(
    m1=initC.mass_1.values, m2=initC.mass_2.values, porb=initC.porb.values, 
    ecc=np.zeros_like(initC.mass_1.values), tphysf=13700.0 * np.ones_like(initC.mass_1.values), 
    kstar1=1 * np.ones_like(initC.mass_1.values), kstar2=1 * np.ones_like(initC.mass_1.values), 
    metallicity=0.02 * np.ones_like(initC.mass_1.values))
    
# no magnetic braking, no wind accretion, no aml from winds, no tides, jeans mode mass loss
BSEDict = {'xi': 1.0, 'bhflag': 1, 'neta': 0.5, 'windflag': 3, 'wdflag': 1, 'alpha1': 1.0, 'pts1': 0.05, 'pts3': 0.02, 'pts2': 0.01, 
           'epsnov': 0.001, 'hewind': 0.5, 'ck': 1000, 'bwind': 0.0, 'lambdaf': 0.0, 'mxns': 3.0, 'beta': -1.0, 'tflag': 0, 'acc2': 0.0, 
           'grflag' : 1, 'remnantflag': 4, 'ceflag': 0, 'eddfac': 1.0, 'ifflag': 0, 'bconst': 3000, 'sigma': 265.0, 'gamma': 0.0, 
           'pisn': 45.0, 'natal_kick_array' : [[-100.0,-100.0,-100.0,-100.0,0.0], [-100.0,-100.0,-100.0,-100.0,0.0]], 
           'bhsigmafrac' : 1.0, 'polar_kick_angle' : 90, 'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 
           'cekickflag' : 2, 'cehestarflag' : 0, 'cemergeflag' : 0, 'ecsn' : 2.25, 'ecsn_mlow' : 1.6, 'aic' : 1, 'ussn' : 0, 
           'sigmadiv' :-20.0, 'qcflag' : 1, 'eddlimflag' : 0, 
           'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,
                             2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0], 
           'bhspinflag' : 0, 'bhspinmag' : 0.0, 'rejuv_fac' : 1.0, 'rejuvflag' : 0, 'htpmb' : 1, 'ST_cr' : 1, 'ST_tide' : 0, 
           'bdecayfac' : 1, 'rembar_massloss' : 0.5, 'kickflag' : 0, 'zsun' : 0.014, 'bhms_coll_flag' : 0, 'don_lim' : -1, 
           'acc_lim' : -4}
           
bpp, bcm, initC, kick_info = Evolve.evolve(initialbinarytable=initBin, BSEDict=BSEDict)

bpp.to_hdf('binary_grid_out.h5', key='bpp')