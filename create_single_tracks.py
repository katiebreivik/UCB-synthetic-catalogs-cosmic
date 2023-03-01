from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.evolve import Evolve

import cosmic
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# set up the BSEDicts for standard winds and no winds
BSEDict_HTP = {'xi': 0.5, 'bhflag': 1, 'neta': 0.5, 'windflag': 0, 'wdflag': 1, 'alpha1': 1.0, 'pts1': 0.05, 'pts3': 0.02, 'pts2': 0.01, 'epsnov': 0.001, 'hewind': 0.5, 'ck': 1000, 'bwind': 0.0, 'lambdaf': 0.0, 'mxns': 3.0, 'beta': -1.0, 'tflag': 1, 'acc2': 1.5, 'grflag' : 1, 'remnantflag': 4, 'ceflag': 0, 'eddfac': 1.0, 'ifflag': 0, 'bconst': 3000, 'sigma': 265.0, 'gamma': -2.0, 'pisn': 45.0, 'natal_kick_array' : [[-100.0,-100.0,-100.0,-100.0,0.0], [-100.0,-100.0,-100.0,-100.0,0.0]], 'bhsigmafrac' : 1.0, 'polar_kick_angle' : 90, 'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 'cekickflag' : 2, 'cehestarflag' : 0, 'cemergeflag' : 0, 'ecsn' : 2.25, 'ecsn_mlow' : 1.6, 'aic' : 1, 'ussn' : 0, 'sigmadiv' :-20.0, 'qcflag' : 1, 'eddlimflag' : 0, 'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0], 'bhspinflag' : 0, 'bhspinmag' : 0.0, 'rejuv_fac' : 1.0, 'rejuvflag' : 0, 'htpmb' : 1, 'ST_cr' : 0, 'ST_tide' : 0, 'bdecayfac' : 1, 'rembar_massloss' : 0.5, 'kickflag' : 0, 'zsun' : 0.02, 'bhms_coll_flag' : 0, 'don_lim' : -1, 'acc_lim' : -1}


BSEDict_HTP_014 = {'xi': 0.5, 'bhflag': 1, 'neta': 0.5, 'windflag': 0, 'wdflag': 1, 'alpha1': 1.0, 'pts1': 0.05, 'pts3': 0.02, 'pts2': 0.01, 'epsnov': 0.001, 'hewind': 0.5, 'ck': 1000, 'bwind': 0.0, 'lambdaf': 0.0, 'mxns': 3.0, 'beta': -1.0, 'tflag': 1, 'acc2': 1.5, 'grflag' : 1, 'remnantflag': 4, 'ceflag': 0, 'eddfac': 1.0, 'ifflag': 0, 'bconst': 3000, 'sigma': 265.0, 'gamma': -2.0, 'pisn': 45.0, 'natal_kick_array' : [[-100.0,-100.0,-100.0,-100.0,0.0], [-100.0,-100.0,-100.0,-100.0,0.0]], 'bhsigmafrac' : 1.0, 'polar_kick_angle' : 90, 'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 'cekickflag' : 2, 'cehestarflag' : 0, 'cemergeflag' : 0, 'ecsn' : 2.25, 'ecsn_mlow' : 1.6, 'aic' : 1, 'ussn' : 0, 'sigmadiv' :-20.0, 'qcflag' : 1, 'eddlimflag' : 0, 'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0], 'bhspinflag' : 0, 'bhspinmag' : 0.0, 'rejuv_fac' : 1.0, 'rejuvflag' : 0, 'htpmb' : 1, 'ST_cr' : 0, 'ST_tide' : 0, 'bdecayfac' : 1, 'rembar_massloss' : 0.5, 'kickflag' : 0, 'zsun' : 0.014, 'bhms_coll_flag' : 0, 'don_lim' : -1, 'acc_lim' : -1}


BSEDict_HPE = {'xi': 0.5, 'bhflag': 1, 'neta': 0.5, 'windflag': 0, 'wdflag': 1, 'alpha1': 1.0, 'pts1': 0.05, 'pts3': 0.02, 'pts2': 0.01, 'epsnov': 0.001, 'hewind': 0.5, 'ck': 1000, 'bwind': 0.0, 'lambdaf': 0.0, 'mxns': 3.0, 'beta': -1.0, 'tflag': 1, 'acc2': 1.5, 'grflag' : 1, 'remnantflag': 4, 'ceflag': 0, 'eddfac': 1.0, 'ifflag': 1, 'bconst': 3000, 'sigma': 265.0, 'gamma': -2.0, 'pisn': 45.0, 'natal_kick_array' : [[-100.0,-100.0,-100.0,-100.0,0.0], [-100.0,-100.0,-100.0,-100.0,0.0]], 'bhsigmafrac' : 1.0, 'polar_kick_angle' : 90, 'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 'cekickflag' : 2, 'cehestarflag' : 0, 'cemergeflag' : 0, 'ecsn' : 2.25, 'ecsn_mlow' : 1.6, 'aic' : 1, 'ussn' : 0, 'sigmadiv' :-20.0, 'qcflag' : 1, 'eddlimflag' : 1, 'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0], 'bhspinflag' : 0, 'bhspinmag' : 0.0, 'rejuv_fac' : 1.0, 'rejuvflag' : 0, 'htpmb' : 1, 'ST_cr' : 0, 'ST_tide' : 0, 'bdecayfac' : 1, 'rembar_massloss' : 0.5, 'kickflag' : 0, 'zsun' : 0.02, 'bhms_coll_flag' : 0, 'don_lim' : -1, 'acc_lim' : -1}

BSEDict_HPE_014 = {'xi': 0.5, 'bhflag': 1, 'neta': 0.5, 'windflag': 0, 'wdflag': 1, 'alpha1': 1.0, 'pts1': 0.05, 'pts3': 0.02, 'pts2': 0.01, 'epsnov': 0.001, 'hewind': 0.5, 'ck': 1000, 'bwind': 0.0, 'lambdaf': 0.0, 'mxns': 3.0, 'beta': -1.0, 'tflag': 1, 'acc2': 1.5, 'grflag' : 1, 'remnantflag': 4, 'ceflag': 0, 'eddfac': 1.0, 'ifflag': 1, 'bconst': 3000, 'sigma': 265.0, 'gamma': -2.0, 'pisn': 45.0, 'natal_kick_array' : [[-100.0,-100.0,-100.0,-100.0,0.0], [-100.0,-100.0,-100.0,-100.0,0.0]], 'bhsigmafrac' : 1.0, 'polar_kick_angle' : 90, 'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 'cekickflag' : 2, 'cehestarflag' : 0, 'cemergeflag' : 0, 'ecsn' : 2.25, 'ecsn_mlow' : 1.6, 'aic' : 1, 'ussn' : 0, 'sigmadiv' :-20.0, 'qcflag' : 1, 'eddlimflag' : 1, 'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0], 'bhspinflag' : 0, 'bhspinmag' : 0.0, 'rejuv_fac' : 1.0, 'rejuvflag' : 0, 'htpmb' : 1, 'ST_cr' : 0, 'ST_tide' : 0, 'bdecayfac' : 1, 'rembar_massloss' : 0.5, 'kickflag' : 0, 'zsun' : 0.014, 'bhms_coll_flag' : 0, 'don_lim' : -1, 'acc_lim' : -1}


BSEDict_MIST_014 = {'xi': 0.5, 'bhflag': 1, 'neta': 0.5, 'windflag': 4, 'wdflag': 1, 'alpha1': 1.0, 'pts1': 0.05, 'pts3': 0.02, 'pts2': 0.01, 'epsnov': 0.001, 'hewind': 0.5, 'ck': 1000, 'bwind': 0.0, 'lambdaf': 0.0, 'mxns': 3.0, 'beta': -1.0, 'tflag': 1, 'acc2': 1.5, 'grflag' : 1, 'remnantflag': 4, 'ceflag': 0, 'eddfac': 1.0, 'ifflag': 0, 'bconst': 3000, 'sigma': 265.0, 'gamma': -2.0, 'pisn': 45.0, 'natal_kick_array' : [[-100.0,-100.0,-100.0,-100.0,0.0], [-100.0,-100.0,-100.0,-100.0,0.0]], 'bhsigmafrac' : 1.0, 'polar_kick_angle' : 90, 'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 'cekickflag' : 2, 'cehestarflag' : 0, 'cemergeflag' : 0, 'ecsn' : 2.25, 'ecsn_mlow' : 1.6, 'aic' : 1, 'ussn' : 0, 'sigmadiv' :-20.0, 'qcflag' : 1, 'eddlimflag' : 1, 'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0], 'bhspinflag' : 0, 'bhspinmag' : 0.0, 'rejuv_fac' : 1.0, 'rejuvflag' : 0, 'htpmb' : 1, 'ST_cr' : 0, 'ST_tide' : 0, 'bdecayfac' : 1, 'rembar_massloss' : 0.5, 'kickflag' : 0, 'zsun' : 0.014, 'bhms_coll_flag' : 0, 'don_lim' : -1, 'acc_lim' : -1}


m1 = np.arange(0.7, 10.0, 0.05)
m2 = np.zeros_like(m1)
porb = np.zeros_like(m1)
ecc = np.zeros_like(m1)
tphysf = np.ones_like(m1) * 50000.0
kstar = np.ones_like(m1)
kstar2 = np.zeros_like(m1)
Z_list = [0.014, 0.002, 0.0002]


for Z in Z_list:
    Zs = np.ones_like(m1) * Z
    init = InitialBinaryTable.InitialBinaries(m1=m1, m2=m2, porb=porb, ecc=ecc, tphysf=tphysf, kstar1=kstar, kstar2=kstar2, metallicity=Zs)
    bpp, bcm, initC, kick_info = Evolve.evolve(initialbinarytable=init, BSEDict=BSEDict_HTP_014, dtp = 0.0)
    bpp_HPE, bcm_HPE, initC_HPE, kick_info_HPE = Evolve.evolve(initialbinarytable=init, BSEDict=BSEDict_HPE_014, dtp = 0.0)
    bpp_MIST, bcm_MIST, initC_MIST, kick_info_MIST = Evolve.evolve(initialbinarytable=init, BSEDict=BSEDict_MIST_014, dtp = 0.0)
    
    for bpp, bcm, winds in zip([bpp, bpp_HPE, bpp_MIST], [bcm, bcm_HPE, bcm_MIST], ['HTP', 'HPE', 'MIST']):
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
        np.savetxt(f'single_tracks_cosmic_{winds}_{Z}_014.csv', dat, delimiter=', ', header="run by Katie Breivik\n Z, m_ZAMS, m_HG, m_GB, m_WD, t_HG_form, t_GB_form, t_WD_form, R_HG_max, R_GB_max, R_WD")

                     
