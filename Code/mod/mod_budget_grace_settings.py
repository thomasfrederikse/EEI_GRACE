# All settings
import numpy as np
from netCDF4 import Dataset
import os

def init():
    global settings
    settings = {}
    # ------------
    # Run settings
    # ------------
    settings['startyear'] = 2003
    settings['stopyear']  = 2019
    settings['ens_start'] = 0
    settings['ens_stop']  = 5000
    settings['ens_range'] = np.arange(settings['ens_start'],settings['ens_stop'])
    settings['num_ens']   = len(settings['ens_range']) # Number of Monte Carlo samples
    settings['steric_products'] = ['EN4_l09','EN4_g10','I17','CZ16','WOA','SIO','JAMSTEC','BOA']
    settings['trend_eras'] = [[2005,2019],[2003,2019],[2005,2015],[2010,2018]] # Eras over which trends are determined

    # -----------------------------
    # Number of available processes
    # -----------------------------
    if os.uname().nodename == 'MT-110180': settings['nproc'] = 4 # JPL MB
    elif os.uname().nodename == 'debian': settings['nproc']  = 4 # Delft 4790k
    else: settings['nproc'] = 48 # Gattaca

    # -----------
    # Directories
    # -----------
    settings['dir_data']    = os.getenv('HOME') + '/Data/'
    settings['dir_budget'] = settings['dir_data'] + 'Budget_GRACE/'
    settings['dir_gmt']    = os.getenv('HOME') + '/Scripts/GMT/Papers/GRACE_budget/'
    settings['dir_glacier_zemp'] =  settings['dir_data'] + 'Glaciers/Zemp_2019/'
    settings['dir_save_ens']  = settings['dir_data'] + 'Budget_GRACE/mass_ens/'

    # ---------
    # Filenames
    # ---------

    # Read GRACE/GIA/masks/altimetry
    settings['fn_grace']       = settings['dir_data'] + 'GRACE/JPL_mascon/JPL_mascon_RL06v02_CRI_noGIA_noEQ_noseas.nc'
    settings['fn_mscn_coords'] = settings['dir_data'] + 'GRACE/JPL_mascon/mascon_coords.npy'
    settings['fn_mask']        = settings['dir_data'] + 'GRACE/JPL_mascon/mask.npy'
    settings['fn_gia_ens_rad'] = settings['dir_data'] + 'GIA/Caron/Ensemble/rad_ens_05.nc'
    settings['fn_gia_ens_rsl'] = settings['dir_data'] + 'GIA/Caron/Ensemble/rsl_ens_05.nc'
    settings['fn_gia_ens_ewh'] = settings['dir_data'] + 'GIA/Caron/Ensemble/ewh_ens_05.nc'
    settings['fn_love']        = settings['dir_data'] + 'Budget_20c/grd_prep/love.npy'
    settings['fn_altimetry']   = settings['dir_data'] + 'Altimetry/CDS/CDS_monthly_2003_2019.nc'

    # Read Steric data
    settings['fn_EN4_l09'] = settings['dir_data'] + 'Steric/EN4/EN4_L09_1950_2020.nc'
    settings['fn_EN4_g10'] = settings['dir_data'] + 'Steric/EN4/EN4_G10_1950_2020.nc'
    settings['fn_I17']     = settings['dir_data'] + 'Steric/I17/I17_1955_2019.nc'
    settings['fn_CZ16']    = settings['dir_data'] + 'Steric/Cheng/Cheng_1940_2020.nc'
    settings['fn_WOA']     = settings['dir_data'] + 'Steric/Levitus/Levitus_2005_2019.nc'
    settings['fn_SIO']     = settings['dir_data'] + 'Steric/SIO/SIO_2004_2020.nc'
    settings['fn_JAMSTEC'] = settings['dir_data'] + 'Steric/JAMSTEC/JAMSTEC_2001_2020.nc'
    settings['fn_BOA']     = settings['dir_data'] + 'Steric/BOA/BOA_2004_2019.nc'

    # Read/write EEI/non-ocean terms
    settings['fn_EEI_GCOS']     = settings['dir_data'] + 'Steric/GCOS/GCOS_all_heat_content_1960-2018_ZJ_v22062020.nc'
    settings['fn_ceres']        = settings['dir_data'] + 'Steric/CERES/CERES_EBAF-TOA_Ed4.1_Subset_200003-202004.nc'
    settings['fn_other_terms'] = settings['dir_budget'] + 'stats/other_terms.npy'

    # Read/write ensembles
    settings['fn_steric_ensemble'] = settings['dir_budget'] + 'steric/steric_ensemble.npy'
    settings['fn_steric_indiv'] = settings['dir_budget'] + 'steric/steric_indiv.npy'
    settings['fn_OHC_ensemble'] = settings['dir_budget'] + 'steric/OHC_ensemble.npy'

    settings['fn_efficiency_deep']  = settings['dir_budget'] + 'efficiency/ECCO_eff_trends5_deep.mat'
    settings['fn_efficiency_total'] = settings['dir_budget'] + 'efficiency/ECCO_eff_trends5_full.mat'
    settings['fn_efficiency']       = settings['dir_budget'] + 'efficiency/efficiency_combined.npy'

    # Read/write statistics
    settings['fn_sealevel_stats'] = settings['dir_budget'] + 'stats/sealevel_stats.npy'
    settings['fn_OHC_stats'] = settings['dir_budget'] + 'stats/OHC_stats.npy'
    settings['fn_OHU_stats'] = settings['dir_budget'] + 'stats/OHU_stats.npy'
    settings['fn_EEI_stats'] = settings['dir_budget'] + 'stats/EEI_stats.npy'
    settings['fn_corr_gia_stats'] = settings['dir_budget'] + 'stats/corr_gia_stats.npy'
    settings['fn_corr_grd_stats'] = settings['dir_budget'] + 'stats/corr_grd_stats.npy'
    settings['fn_corr_tot_stats'] = settings['dir_budget'] + 'stats/corr_tot_stats.npy'

    # Read/write gridded stats
    settings['fn_grid_trends']        = settings['dir_budget'] + 'stats/grid_trends.nc'
    settings['fn_grid_stats']        = settings['dir_budget'] + 'stats/grid_stats.nc'

    # Time/lat/lon/basins
    settings['ntime'] = len(np.zeros(12 * (settings['stopyear'] - settings['startyear']+1)))
    year = (np.floor(np.arange(settings['ntime']) / 12) + settings['startyear'])
    settings['time'] = np.around(year + (np.arange(settings['ntime'])/12 + 1/24) - (year - settings['startyear']),4)
    settings['time_mask_grace'] = np.ones(len(settings['time']),dtype=bool)
    settings['time_mask_grace'][174:185] = False

    settings['lat']  = np.arange(-89.75,90.25,0.5)
    settings['lon']  = np.arange(0.25,360.25,0.5)
    settings['probability'] = Dataset(settings['dir_data'] + 'GIA/Caron/Ensemble/rad_ens_05.nc').variables['probability'][:]._get_data()
    settings['region_names'] = ['Subpolar North Atlantic','Indian Ocean-South Pacific','Subtropical North Atlantic','East Pacific','South Atlantic','Northwest Pacific']
    return
