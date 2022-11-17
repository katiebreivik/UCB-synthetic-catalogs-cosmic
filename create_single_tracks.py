from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.evolve import Evolve

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# set up the BSEDicts for standard winds and no winds
BSEDict_standard_winds = {'xi': 1.0, 'bhflag': 1, 'neta': 0.5, 'windflag': 3, 'wdflag': 1, 'alpha1': 1.0, 'pts1': 0.001, 'pts3': 0.02, 'pts2': 0.01, 'epsnov': 0.001, 'hewind': 0.5, 'ck': 1000, 'bwind': 0.0, 'lambdaf': 0.0, 'mxns': 3.0, 'beta': -1.0, 'tflag': 1, 'acc2': 1.5, 'grflag' : 1, 'remnantflag': 4, 'ceflag': 0, 'eddfac': 1.0, 'ifflag': 0, 'bconst': 3000, 'sigma': 265.0, 'gamma': -2.0, 'pisn': 45.0, 'natal_kick_array' : [[-100.0,-100.0,-100.0,-100.0,0.0], [-100.0,-100.0,-100.0,-100.0,0.0]], 'bhsigmafrac' : 1.0, 'polar_kick_angle' : 90, 'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 'cekickflag' : 2, 'cehestarflag' : 0, 'cemergeflag' : 0, 'ecsn' : 2.25, 'ecsn_mlow' : 1.6, 'aic' : 1, 'ussn' : 0, 'sigmadiv' :-20.0, 'qcflag' : 1, 'eddlimflag' : 0, 'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0], 'bhspinflag' : 0, 'bhspinmag' : 0.0, 'rejuv_fac' : 1.0, 'rejuvflag' : 0, 'htpmb' : 1, 'ST_cr' : 1, 'ST_tide' : 1, 'bdecayfac' : 1, 'rembar_massloss' : 0.5, 'kickflag' : 0, 'zsun' : 0.02, 'bhms_coll_flag' : 0, 'don_lim' : -1, 'acc_lim' : -1}

BSEDict_no_winds = {'xi': 1.0, 'bhflag': 1, 'neta': 0.5, 'windflag': 0, 'wdflag': 1, 'alpha1': 1.0, 'pts1': 0.001, 'pts3': 0.02, 'pts2': 0.01, 'epsnov': 0.001, 'hewind': 0.5, 'ck': 1000, 'bwind': 0.0, 'lambdaf': 0.0, 'mxns': 3.0, 'beta': -1.0, 'tflag': 1, 'acc2': 1.5, 'grflag' : 1, 'remnantflag': 4, 'ceflag': 0, 'eddfac': 1.0, 'ifflag': 0, 'bconst': 3000, 'sigma': 265.0, 'gamma': -2.0, 'pisn': 45.0, 'natal_kick_array' : [[-100.0,-100.0,-100.0,-100.0,0.0], [-100.0,-100.0,-100.0,-100.0,0.0]], 'bhsigmafrac' : 1.0, 'polar_kick_angle' : 90, 'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 'cekickflag' : 2, 'cehestarflag' : 0, 'cemergeflag' : 0, 'ecsn' : 2.25, 'ecsn_mlow' : 1.6, 'aic' : 1, 'ussn' : 0, 'sigmadiv' :-20.0, 'qcflag' : 1, 'eddlimflag' : 0, 'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0], 'bhspinflag' : 0, 'bhspinmag' : 0.0, 'rejuv_fac' : 1.0, 'rejuvflag' : 0, 'htpmb' : 1, 'ST_cr' : 1, 'ST_tide' : 1, 'bdecayfac' : 1, 'rembar_massloss' : 0.5, 'kickflag' : 0, 'zsun' : 0.02, 'bhms_coll_flag' : 0, 'don_lim' : -1, 'acc_lim' : -1}


m1 = np.arange(0.7, 10.0, 0.05)
m2 = np.zeros_like(m1)
porb = np.zeros_like(m1)
ecc = np.zeros_like(m1)
tphysf = np.ones_like(m1) * 13700.0
kstar = np.ones_like(m1)
Z_list = [0.02, 0.002, 0.0002]


for Z in Z_list:
    Zs = np.ones_like(m1) * Z
    init = InitialBinaryTable.InitialBinaries(m1=m1, m2=m2, porb=porb, ecc=ecc, tphysf=tphysf, kstar1=kstar, kstar2=kstar, metallicity=Zs)
    bpp_winds, bcm_winds, initC_winds, kick_info_winds = Evolve.evolve(initialbinarytable=init, BSEDict=BSEDict_standard_winds, dtp = 0.0)
    bpp_no_winds, bcm_no_winds, initC_no_winds, kick_info_no_winds = Evolve.evolve(initialbinarytable=init, BSEDict=BSEDict_no_winds, dtp = 0.0)
    
    for bpp, bcm, winds in zip([bpp_winds, bpp_no_winds], [bcm_winds, bcm_no_winds], ['winds_on', 'winds_off']):
        #select all the stages of interest
        bpp_HG = bpp.loc[bpp.kstar_1 == 2].groupby('bin_num', as_index=False).first()
        bpp_GB = bpp.loc[bpp.kstar_1 == 3].groupby('bin_num', as_index=False).first()
        bpp_WD = bpp.loc[bpp.kstar_1.isin([10,11,12])].groupby('bin_num', as_index=False).first()
        
        #Fill in stages which aren't hit with zeros
        m_HG = bpp_HG.mass_1.values
        t_HG = bpp_HG.tphys.values
        rad_max_HG = bcm.loc[bcm.kstar_1 == 2].groupby('bin_num').rad_1.max()
        m_HG = np.append(m_HG, np.zeros(len(m1) - len(m_HG)))
        t_HG = np.append(t_HG, np.zeros(len(m1) - len(t_HG)))              
        rad_max_HG = np.append(rad_max_HG, np.zeros(len(m1) - len(rad_max_HG)))              
        
        m_GB = bpp_GB.mass_1.values
        t_GB = bpp_GB.tphys.values
        rad_max_GB = bcm.loc[bcm.kstar_1 == 3].groupby('bin_num').rad_1.max()
        m_GB = np.append(m_GB, np.zeros(len(m1) - len(m_GB)))
        t_GB = np.append(t_GB, np.zeros(len(m1) - len(t_GB)))              
        rad_max_GB = np.append(rad_max_GB, np.zeros(len(m1) - len(rad_max_GB)))    
        
        m_WD = bpp_WD.mass_1.values
        t_WD = bpp_WD.tphys.values
        r_WD = bpp_WD.rad_1.values
        m_WD = np.append(m_WD, np.zeros(len(m1) - len(m_WD)))
        t_WD = np.append(t_WD, np.zeros(len(m1) - len(t_WD)))              
        r_WD = np.append(r_WD, np.zeros(len(m1) - len(r_WD)))              
        
        #Z, m_ZAMS, m_HG, m_GB, m_WD, t_HG_form, t_GB_form, t_WD_form, R_HG_max, R_GB_max, R_WD
        dat = np.vstack([Zs, m1, m_HG, m_GB, m_WD, t_HG, t_GB, t_WD, rad_max_HG, rad_max_GB, r_WD])
        
        #save the data        
        np.savetxt(f'single_tracks_cosmic_{winds}_{Z}.csv', dat, delimiter=', ', header="run by Katie Breivik\n Z, m_ZAMS, m_HG, m_GB, m_WD, t_HG_form, t_GB_form, t_WD_form, R_HG_max, R_GB_max, R_WD")

                     
