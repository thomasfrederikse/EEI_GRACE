# --------------------------------------
# Compute time series and trends for all
# ocean mass and observed sea-level time
# series on basin, altimetry and global
# --------------------------------------
import numpy as np
from netCDF4 import Dataset
import multiprocessing as mp
import ctypes as ct
import mod_gentools as gentools

def main():
    print('Computing ocean mass and altimetry statistics...')
    from mod_budget_grace_settings import settings
    global settings,dt_start
    sealevel_stats = {}

    # Ensembles and statistics for sea level
    rsl_ensemble,sealevel_stats['rsl']                    = compute_rsl()
    gsl_ensemble,sealevel_stats['gsl']                    = compute_gsl()
    mass_ensemble,sealevel_stats['mass']                  = compute_mass()
    steric_ensemble,sealevel_stats['steric']              = compute_steric()
    rsl_min_mass_ensemble, sealevel_stats['rsl_min_mass'] = compute_rsl_min_mass(rsl_ensemble,mass_ensemble)
    budget_ensemble, sealevel_stats['budget']             = compute_budget(mass_ensemble,steric_ensemble)
    diff_ensemble, sealevel_stats['diff']        = compute_diff(rsl_ensemble,budget_ensemble)
    sealevel_stats['mass_ctb']                   = compute_ctb_mass()

    # Ensembles and statistics for OHC
    OHC_stats = {}
    OHC_hydrography_ensemble, OHC_stats['hydrography'] = compute_OHC_hydrography()
    OHC_from_steric_ensemble, OHC_stats['hydrography_steric'] = compute_OHC_from_steric_hydrography()
    OHC_altmass_total_ens, OHC_stats['altmass_total']  = compute_OHC_altmass_total(rsl_min_mass_ensemble)
    OHC_altmass_deep_ens, OHC_stats['altmass_deep']    = compute_OHC_altmass_deep(OHC_hydrography_ensemble,OHC_altmass_total_ens)

    # Statistics for annual-mean ocean heat uptake (OHU)
    OHU_stats = {}
    OHU_stats['hydrography'] = ensemble_ohu(OHC_hydrography_ensemble['altimetry'],is_grace=False)
    OHU_stats['hydrography_steric'] = ensemble_ohu(OHC_from_steric_ensemble['altimetry'],is_grace=False)
    OHU_stats['altmass']     = ensemble_ohu(OHC_altmass_total_ens['altimetry'],is_grace=True)

    # Statistics for EEI trends (OHC trend plus non-OHC trends)
    EEI_stats = {}
    EEI_stats['hydrography'] = ensemble_eei(OHC_hydrography_ensemble['altimetry'], add_land=True, add_deep=True)
    EEI_stats['altmass']     = ensemble_eei(OHC_altmass_total_ens['altimetry'], add_land=True, add_deep=False)

    # Corrections for GIA and GRD
    corr_gia_stats, corr_grd_stats,corr_tot_stats = compute_corr()
    # Save data
    print('   Saving data...')
    np.save(settings['fn_sealevel_stats'],sealevel_stats)
    np.save(settings['fn_OHC_stats'],OHC_stats)
    np.save(settings['fn_OHU_stats'],OHU_stats)
    np.save(settings['fn_EEI_stats'],EEI_stats)
    np.save(settings['fn_corr_gia_stats'],corr_gia_stats)
    np.save(settings['fn_corr_grd_stats'],corr_grd_stats)
    np.save(settings['fn_corr_tot_stats'],corr_tot_stats)
    return

# Sea level
def compute_gsl():
    print('   Geocentric sea level...')
    global settings, gsl_ensemble
    # Read data
    gsl_ensemble = {}
    gsl_ensemble['altimetry'] = mp_empty_float([settings['num_ens'],settings['ntime']])
    gsl_ensemble['basin'] = mp_empty_float([settings['num_ens'],6,settings['ntime']])
    pool = mp.Pool(settings['nproc'])
    out  = pool.map(read_gsl_ens, settings['ens_range'])
    gsl_ensemble['altimetry'][:, ~settings['time_mask_grace']] = np.nan
    gsl_ensemble['basin'][:, :, ~settings['time_mask_grace']] = np.nan
    gsl_stats = {}
    gsl_stats['basin'] = np.zeros(6,dtype=object)
    gsl_stats['altimetry'] = ensemble_stats(gsl_ensemble['altimetry'])
    for basin in range(6): gsl_stats['basin'][basin] = ensemble_stats(gsl_ensemble['basin'][:,basin,:])
    return(gsl_ensemble,gsl_stats)

def read_gsl_ens(ens):
    global settings, gsl_ensemble
    fname = settings['dir_save_ens'] + 'mass_rsl_ens_' + str(ens).zfill(4) + '.nc'
    file_handle = Dataset(fname, 'r')
    file_handle.set_auto_mask(False)
    gsl_ensemble['altimetry'][ens, :] = file_handle.variables['obs_gsl_altimetry'][:]
    gsl_ensemble['basin'][ens, :,:] = file_handle.variables['obs_gsl_basin'][:]
    file_handle.close()
    return

def compute_rsl_min_mass(rsl_ensemble,mass_ensemble):
    print('   Difference...')
    global settings
    rsl_min_mass_ensemble = {}
    rsl_min_mass_ensemble['global']    = rsl_ensemble['altimetry'] - mass_ensemble['global']
    rsl_min_mass_ensemble['altimetry'] = rsl_ensemble['altimetry'] - mass_ensemble['altimetry']
    rsl_min_mass_ensemble['basin']     = rsl_ensemble['basin']     - mass_ensemble['basin']

    rsl_min_mass_stats = {}
    rsl_min_mass_stats['global']    = ensemble_stats(rsl_min_mass_ensemble['global'])
    rsl_min_mass_stats['altimetry'] = ensemble_stats(rsl_min_mass_ensemble['altimetry'])
    rsl_min_mass_stats['basin'] = np.zeros(6, dtype=object)
    for basin in range(6): rsl_min_mass_stats['basin'][basin] = ensemble_stats(rsl_min_mass_ensemble['basin'][:, basin, :])
    return(rsl_min_mass_ensemble, rsl_min_mass_stats)

def compute_steric():
    print('   Steric...')
    global settings
    steric_ensemble = np.load(settings['fn_steric_ensemble'],allow_pickle=True).all()
    steric_stats = {}
    steric_stats['basin']     = np.zeros(6,dtype=object)
    steric_stats['global']    = ensemble_stats(steric_ensemble['global'][:settings['num_ens'],:])
    steric_stats['altimetry'] = ensemble_stats(steric_ensemble['altimetry'][:settings['num_ens'],:])
    for basin in range(6): steric_stats['basin'][basin] = ensemble_stats(steric_ensemble['basin'][:settings['num_ens'],basin,:])
    return(steric_ensemble,steric_stats)

def compute_ctb_mass():
    print('   Mass contributors...')
    # Compute land mass changes from indiv contributors
    global settings, mass_ctb_ens, ctb
    mass_ctb_list = ['mass_land','mass_GrIS','mass_AIS','mass_glac','mass_tws']
    mass_ctb_stats = {}
    for ctb in mass_ctb_list:
        print('     '+ctb)
        mass_ctb_ens = mp_empty_float([settings['num_ens'],settings['ntime']])
        pool = mp.Pool(settings['nproc'])
        out = pool.map(read_ctb_mass_ens, settings['ens_range'])
        mass_ctb_ens[:, ~settings['time_mask_grace']] = np.nan
        mass_ctb_stats[ctb] = ensemble_stats(mass_ctb_ens)
    return(mass_ctb_stats)

def read_ctb_mass_ens(ens):
    global settings, mass_ctb_ens, ctb
    fname = settings['dir_save_ens'] + 'mass_rsl_ens_' + str(ens).zfill(4) + '.nc'
    mass_ctb_ens[ens, :] = Dataset(fname, 'r').variables[ctb][:]._get_data()
    return

def compute_mass():
    print('   Ocean mass...')
    global settings, mass_ensemble
    # Read data
    mass_ensemble = {}
    mass_ensemble['global'] = mp_empty_float([settings['num_ens'],settings['ntime']])
    mass_ensemble['altimetry'] = mp_empty_float([settings['num_ens'],settings['ntime']])
    mass_ensemble['basin'] = mp_empty_float([settings['num_ens'],6,settings['ntime']])
    pool = mp.Pool(settings['nproc'])
    out  = pool.map(read_mass_ens, settings['ens_range'])
    mass_ensemble['global'][:, ~settings['time_mask_grace']] = np.nan
    mass_ensemble['altimetry'][:, ~settings['time_mask_grace']] = np.nan
    mass_ensemble['basin'][:, :, ~settings['time_mask_grace']] = np.nan
    mass_stats = {}
    mass_stats['basin'] = np.zeros(6,dtype=object)
    mass_stats['global']    = ensemble_stats(mass_ensemble['global'])
    mass_stats['altimetry'] = ensemble_stats(mass_ensemble['altimetry'])
    for basin in range(6): mass_stats['basin'][basin] = ensemble_stats(mass_ensemble['basin'][:,basin,:])
    return(mass_ensemble,mass_stats)

def read_mass_ens(ens):
    global settings, mass_ensemble
    fname = settings['dir_save_ens'] + 'mass_rsl_ens_' + str(ens).zfill(4) + '.nc'
    file_handle = Dataset(fname, 'r')
    file_handle.set_auto_mask(False)
    mass_ensemble['global'][ens, :]    = file_handle.variables['mass_ocean'][:]
    mass_ensemble['altimetry'][ens, :] = file_handle.variables['mass_altimetry'][:]
    mass_ensemble['basin'][ens, :, ]   = file_handle.variables['mass_basins'][:]
    file_handle.close()
    return

def compute_rsl():
    print('   Sea level...')
    global settings, rsl_ensemble
    # Read data
    rsl_ensemble = {}
    rsl_ensemble['altimetry'] = mp_empty_float([settings['num_ens'],settings['ntime']])
    rsl_ensemble['basin'] = mp_empty_float([settings['num_ens'],6,settings['ntime']])
    pool = mp.Pool(settings['nproc'])
    out  = pool.map(read_rsl_ens, settings['ens_range'])
    rsl_ensemble['altimetry'][:, ~settings['time_mask_grace']] = np.nan
    rsl_ensemble['basin'][:, :, ~settings['time_mask_grace']] = np.nan
    rsl_stats = {}
    rsl_stats['basin'] = np.zeros(6,dtype=object)
    rsl_stats['altimetry'] = ensemble_stats(rsl_ensemble['altimetry'])
    for basin in range(6): rsl_stats['basin'][basin] = ensemble_stats(rsl_ensemble['basin'][:,basin,:])
    return(rsl_ensemble,rsl_stats)

def read_rsl_ens(ens):
    global settings, rsl_ensemble
    fname = settings['dir_save_ens'] + 'mass_rsl_ens_' + str(ens).zfill(4) + '.nc'
    file_handle = Dataset(fname, 'r')
    file_handle.set_auto_mask(False)
    rsl_ensemble['altimetry'][ens, :] = file_handle.variables['obs_rsl_altimetry'][:]
    rsl_ensemble['basin'][ens, :, ] = file_handle.variables['obs_rsl_basin'][:]
    file_handle.close()
    return

def compute_budget(mass_ensemble,steric_ensemble):
    print('   Budget...')
    global settings
    budget_ensemble = {}
    budget_ensemble['global'] = mass_ensemble['global'] + steric_ensemble['global'][:settings['num_ens'],:]
    budget_ensemble['altimetry'] = mass_ensemble['altimetry'] + steric_ensemble['altimetry'][:settings['num_ens'],:]
    budget_ensemble['basin'] = mass_ensemble['basin'] + steric_ensemble['basin'][:settings['num_ens'],:,:]

    budget_stats = {}
    budget_stats['basin'] = np.zeros(6,dtype=object)
    budget_stats['global'] = ensemble_stats(budget_ensemble['global'])
    budget_stats['altimetry'] = ensemble_stats(budget_ensemble['altimetry'])
    for basin in range(6):
        budget_stats['basin'][basin] = ensemble_stats(budget_ensemble['basin'][:,basin,:])
    return(budget_ensemble, budget_stats)

def compute_diff(rsl_ensemble,budget_ensemble):
    print('   Difference...')
    global settings
    diff_ensemble = {}
    diff_ensemble['altimetry'] = rsl_ensemble['altimetry'] - budget_ensemble['altimetry']
    diff_ensemble['basin']     = rsl_ensemble['basin']     - budget_ensemble['basin']

    diff_stats = {}
    diff_stats['basin'] = np.zeros(6, dtype=object)
    diff_stats['altimetry'] = ensemble_stats(diff_ensemble['altimetry'])
    for basin in range(6): diff_stats['basin'][basin] = ensemble_stats(diff_ensemble['basin'][:, basin, :])
    return(diff_ensemble,diff_stats)

# ---
# OHC
# ---
def compute_OHC_hydrography():
    print('   OHC from hydrography...')
    global settings
    OHC_ensemble = np.load(settings['fn_OHC_ensemble'],allow_pickle=True).all()
    OHC_stats = {}
    OHC_stats['global'] = ensemble_stats(OHC_ensemble['global'][:settings['num_ens'],:])
    OHC_stats['altimetry'] = ensemble_stats(OHC_ensemble['altimetry'][:settings['num_ens'],:])
    return(OHC_ensemble,OHC_stats)

def compute_OHC_from_steric_hydrography():
    print('   OHC from steric hydrography...')
    global settings
    efficiency = np.load(settings['fn_efficiency'], allow_pickle=True).all()
    efficiency_total_ens = 1. / np.random.normal(loc=efficiency['shallow'][0], scale=efficiency['shallow'][1], size=settings['num_ens'])
    steric_ensemble = np.load(settings['fn_steric_ensemble'], allow_pickle=True).all()

    OHC_from_steric_ensemble = {}
    OHC_from_steric_ensemble['global'] = efficiency_total_ens[:, np.newaxis] * steric_ensemble['global'][:settings['num_ens'], :]
    OHC_from_steric_ensemble['altimetry'] = efficiency_total_ens[:, np.newaxis] * steric_ensemble['altimetry'][:settings['num_ens'], :]

    OHC_from_steric_stats = {}
    OHC_from_steric_stats['global'] = ensemble_stats(OHC_from_steric_ensemble['global'])
    OHC_from_steric_stats['altimetry'] = ensemble_stats(OHC_from_steric_ensemble['altimetry'])
    return(OHC_from_steric_ensemble,OHC_from_steric_stats)


def compute_OHC_altmass_total(rsl_min_mass_ensemble):
    # From residual to OHC
    print('   OHC from altimetry minus mass...')
    global settings
    efficiency = np.load(settings['fn_efficiency'],allow_pickle=True).all()
    efficiency_total_ens = 1./np.random.normal(loc=efficiency['full'][0],scale=efficiency['full'][1],size=settings['num_ens'])

    OHC_altmass_total_ens = {}
    OHC_altmass_total_ens['altimetry'] = rsl_min_mass_ensemble['altimetry'] * efficiency_total_ens[:,np.newaxis]

    OHC_altmass_total_stats  = {}
    OHC_altmass_total_stats['altimetry'] = ensemble_stats(OHC_altmass_total_ens['altimetry'])
    return(OHC_altmass_total_ens, OHC_altmass_total_stats)

def compute_OHC_altmass_deep(OHC_hydrography_ensemble,OHC_altmass_total_ens):
    # ---------------------------------------------------
    # Compute deep OHC from subtracting geodetic OHC from
    # hydrographic OHC.
    # ---------------------------------------------------
    global settings
    print('   Deep OHC from altimetry minus mass minus upper-ocean steric...')
    OHC_altmass_deep_ens = {}
    OHC_altmass_deep_ens['altimetry']  = OHC_altmass_total_ens['altimetry'] - OHC_hydrography_ensemble['altimetry'][:settings['num_ens'],...]
    OHC_altmass_deep_stats  = {}
    OHC_altmass_deep_stats['altimetry']  = ensemble_stats(OHC_altmass_deep_ens['altimetry'])
    return(OHC_altmass_deep_ens,OHC_altmass_deep_stats)

def ensemble_stats(ensemble):
    global settings
    # ---------------------------------------------------------------------
    # Compute time series, trend, and acceleration statistics from ensemble
    #  - Mean, and 5-95 CI
    # ---------------------------------------------------------------------
    probability = read_probability()
    stats = {}
    # Remove baseline (2005)
    acc_baseline  = (settings['time']>2019) & (settings['time']<2020)
    ensemble-= ensemble[:,acc_baseline].mean(axis=1)[:,np.newaxis]
    # Time series
    stats['tseries'] = np.zeros([len(settings['time']), 3])*np.nan
    for t in range(len(settings['time'])):
        t_acc = np.isfinite(ensemble[:, t])
        if t_acc.sum()>0:
            sort_idx = np.argsort(ensemble[t_acc, t])
            sort_cdf = np.cumsum((probability[t_acc]/probability[t_acc].sum())[sort_idx])
            stats['tseries'][t, 0] = ensemble[t_acc, t][sort_idx][np.argmin(np.abs(sort_cdf - 0.05))]
            stats['tseries'][t, 1] = (probability[t_acc]/probability[t_acc].sum() * ensemble[t_acc, t]).sum()
            stats['tseries'][t, 2] = ensemble[t_acc, t][sort_idx][np.argmin(np.abs(sort_cdf - 0.95))]
    # Trends and acceleration
    stats['trend'] = {}
    stats['accel'] = {}
    for era in range(len(settings['trend_eras'])):
        # Only select ensemble members that have full coverage over the period
        time_acc = ((settings['time'] > settings['trend_eras'][era][0]) & (settings['time'] < settings['trend_eras'][era][1]+1))
        ens_acc  = np.isfinite(ensemble[:,time_acc]).sum(axis=1) == (np.isfinite(ensemble[:,time_acc]).sum(axis=1)).max()
        ens_era = ensemble[ens_acc,:]
        time_acc[np.isnan(ens_era[0,:])] = False
        ens_era = ens_era[:,time_acc]

        tname = str(settings['trend_eras'][era][0])+'-'+str(settings['trend_eras'][era][1])
        stats['trend'][tname] = np.zeros(3)
        stats['accel'][tname] = np.zeros(3)
        ens_trend_era = np.zeros(ens_era.shape[0])
        ens_accel_era = np.zeros(ens_era.shape[0])
        # Design matrix
        amat = np.ones([len(settings['time'][time_acc]),3])
        amat[:,1] = settings['time'][time_acc] - settings['time'][time_acc].mean()
        amat[:,2] = 0.5*(settings['time'][time_acc] - settings['time'][time_acc].mean())**2
        amat_T  = amat.T
        amat_sq = np.linalg.inv(np.dot(amat_T, amat))
        # Loop over all ensemble members
        for i in range(ens_era.shape[0]):
            sol = np.dot(amat_sq, np.dot(amat_T, ens_era[i,:]))
            ens_trend_era[i] = sol[1]
            ens_accel_era[i] = sol[2]
        # Statistics from ensemble
        sort_idx = np.argsort(ens_trend_era)
        sort_cdf = np.cumsum((probability[ens_acc] / probability[ens_acc].sum())[sort_idx])
        stats['trend'][tname][0] = ens_trend_era[sort_idx][np.argmin(np.abs(sort_cdf - 0.05))]
        stats['trend'][tname][1] = ((probability[ens_acc] / probability[ens_acc].sum()) * ens_trend_era).sum()
        stats['trend'][tname][2] = ens_trend_era[sort_idx][np.argmin(np.abs(sort_cdf - 0.95))]
        sort_idx = np.argsort(ens_accel_era)
        sort_cdf = np.cumsum((probability[ens_acc] / probability[ens_acc].sum())[sort_idx])
        stats['accel'][tname][0] = ens_accel_era[sort_idx][np.argmin(np.abs(sort_cdf - 0.05))]
        stats['accel'][tname][1] = ((probability[ens_acc] / probability[ens_acc].sum()) * ens_accel_era).sum()
        stats['accel'][tname][2] = ens_accel_era[sort_idx][np.argmin(np.abs(sort_cdf - 0.95))]
    return(stats)

def ensemble_ohu(ensemble,is_grace=False):
    global settings

    # Only process set number of ensembles
    ensemble = ensemble[:settings['num_ens'],:]

    # Compute a OHU ensemble from OHC estimates
    stats = {}
    probability = read_probability()
    ohc_years = np.unique(np.floor(settings['time'])).astype(int)

    # Compute the OHC gradient to generate OHU time series ensemble
    ensemble_gradient = np.zeros(ensemble.shape)*np.nan
    for ens in range(settings['num_ens']):
        acc_idx = np.isfinite(ensemble[ens,:])
        ensemble_gradient[ens,acc_idx] = np.gradient(ensemble[ens,acc_idx])/(4*np.pi*6371000**2)/(3600*24*30.5)
    ensemble_gradient_annual = np.zeros([settings['num_ens'],len(ohc_years)]) * np.nan
    for idx, year in enumerate(ohc_years):
        acc_idx = (settings['time']>=year) & (settings['time']<year+1)
        ensemble_gradient_annual[:,idx] = np.nanmean(ensemble_gradient[:,acc_idx],axis=1)

    if is_grace:
        no_grace_idx = (ohc_years > 2016) & (ohc_years < 2019)
        ensemble_gradient_annual[:,no_grace_idx] = np.nan


    # Time series
    stats['years'] = ohc_years
    stats['tseries'] = np.zeros([len(stats['years']),3]) * np.nan
    for t in range(len(stats['years'])):
        t_acc = np.isfinite(ensemble_gradient_annual[:, t])
        if t_acc.sum()>0:
            sort_idx = np.argsort(ensemble_gradient_annual[t_acc, t])
            sort_cdf = np.cumsum((probability[t_acc]/probability[t_acc].sum())[sort_idx])
            stats['tseries'][t, 0] = ensemble_gradient_annual[t_acc, t][sort_idx][np.argmin(np.abs(sort_cdf - 0.05))]
            stats['tseries'][t, 1] = (probability[t_acc]/probability[t_acc].sum() * ensemble_gradient_annual[t_acc, t]).sum()
            stats['tseries'][t, 2] = ensemble_gradient_annual[t_acc, t][sort_idx][np.argmin(np.abs(sort_cdf - 0.95))]

    # Time series with mean removed
    stats['tseries_nomean'] = np.zeros([len(stats['years']),3]) * np.nan
    mean_years = (stats['years']>2004.5) & (stats['years']<2018.5)
    ensemble_gradient_annual_nomean = ensemble_gradient_annual - ensemble_gradient_annual[:,mean_years].mean(axis=1)[:,np.newaxis]
    for t in range(len(stats['years'])):
        t_acc = np.isfinite(ensemble_gradient_annual[:, t])
        if t_acc.sum()>0:
            sort_idx = np.argsort(ensemble_gradient_annual[t_acc, t])
            sort_cdf = np.cumsum((probability[t_acc]/probability[t_acc].sum())[sort_idx])
            stats['tseries_nomean'][t, 0] = ensemble_gradient_annual_nomean[t_acc, t][sort_idx][np.argmin(np.abs(sort_cdf - 0.05))]
            stats['tseries_nomean'][t, 1] = (probability[t_acc]/probability[t_acc].sum() * ensemble_gradient_annual_nomean[t_acc, t]).sum()
            stats['tseries_nomean'][t, 2] = ensemble_gradient_annual_nomean[t_acc, t][sort_idx][np.argmin(np.abs(sort_cdf - 0.95))]

    # Trends
    stats['trend'] = {}
    for era in range(len(settings['trend_eras'])):
        # Only select ensemble members that have full coverage over the period
        time_acc = ((settings['time'] > settings['trend_eras'][era][0]) & (settings['time'] < settings['trend_eras'][era][1]+1))
        ens_acc  = np.isfinite(ensemble_gradient[:,time_acc]).sum(axis=1) == (np.isfinite(ensemble_gradient[:,time_acc]).sum(axis=1)).max()
        ens_era = ensemble_gradient[ens_acc,:]
        time_acc[np.isnan(ens_era[0,:])] = False
        ens_era = ens_era[:,time_acc]

        tname = str(settings['trend_eras'][era][0])+'-'+str(settings['trend_eras'][era][1])
        stats['trend'][tname] = np.zeros(3)
        ens_trend_era = np.zeros(ens_era.shape[0])
        # Design matrix
        amat = np.ones([len(settings['time'][time_acc]),2])
        amat[:,1] = settings['time'][time_acc] - settings['time'][time_acc].mean()
        amat_T  = amat.T
        amat_sq = np.linalg.inv(np.dot(amat_T, amat))
        # Loop over all ensemble members
        for i in range(ens_era.shape[0]):
            sol = np.dot(amat_sq, np.dot(amat_T, ens_era[i,:]))
            ens_trend_era[i] = sol[1]
        # Statistics from ensemble
        sort_idx = np.argsort(ens_trend_era)
        sort_cdf = np.cumsum((probability[ens_acc] / probability[ens_acc].sum())[sort_idx])
        stats['trend'][tname][0] = ens_trend_era[sort_idx][np.argmin(np.abs(sort_cdf - 0.05))]
        stats['trend'][tname][1] = ((probability[ens_acc] / probability[ens_acc].sum()) * ens_trend_era).sum()
        stats['trend'][tname][2] = ens_trend_era[sort_idx][np.argmin(np.abs(sort_cdf - 0.95))]
    return(stats)

def ensemble_eei(ensemble,add_land=False,add_deep=False):
    # Use OHC ensemble to compute EEI trends
    global settings

    # Only process set number of ensembles
    ensemble = ensemble[:settings['num_ens'],:]
    probability = read_probability()
    other_terms = np.load(settings['fn_other_terms'],allow_pickle=True).all()

    # Non-ocean EEI trends (GCOS estimates)
    non_ocean_terms = np.zeros(ensemble.shape)
    if add_land: non_ocean_terms+=other_terms['non_ocean']['ts_ens'][:settings['num_ens'],:]
    if add_deep: non_ocean_terms+=other_terms['deep_ocean']['ohc_ts_ens'][:settings['num_ens'],:]

    ensemble+=non_ocean_terms

    # Scale ensemble from ZJ/yr to W/m^2
    ensemble_scaled = ensemble/(3600*24*365.25*4*np.pi*6371000**2)

    stats = {}
    stats['trend'] = {}
    stats['accel'] = {}
    for era in range(len(settings['trend_eras'])):
        # Only select ensemble members that have full coverage over the period
        time_acc = ((settings['time'] > settings['trend_eras'][era][0]) & (settings['time'] < settings['trend_eras'][era][1]+1))
        ens_acc  = np.isfinite(ensemble_scaled[:,time_acc]).sum(axis=1) == (np.isfinite(ensemble_scaled[:,time_acc]).sum(axis=1)).max()
        ens_era = ensemble_scaled[ens_acc,:]
        time_acc[np.isnan(ens_era[0,:])] = False
        ens_era = ens_era[:,time_acc]

        tname = str(settings['trend_eras'][era][0])+'-'+str(settings['trend_eras'][era][1])
        stats['trend'][tname] = np.zeros(3)
        stats['accel'][tname] = np.zeros(3)
        ens_trend_era = np.zeros(ens_era.shape[0])
        ens_accel_era = np.zeros(ens_era.shape[0])
        # Design matrix
        amat = np.ones([len(settings['time'][time_acc]),3])
        amat[:,1] = settings['time'][time_acc] - settings['time'][time_acc].mean()
        amat[:,2] = 0.5*(settings['time'][time_acc] - settings['time'][time_acc].mean())**2
        amat_T  = amat.T
        amat_sq = np.linalg.inv(np.dot(amat_T, amat))
        # Loop over all ensemble members
        for i in range(ens_era.shape[0]):
            sol = np.dot(amat_sq, np.dot(amat_T, ens_era[i,:]))
            ens_trend_era[i] = sol[1]
            ens_accel_era[i] = sol[2]

        # Statistics from ensemble
        sort_idx = np.argsort(ens_trend_era)
        sort_cdf = np.cumsum((probability[ens_acc] / probability[ens_acc].sum())[sort_idx])
        stats['trend'][tname][0] = ens_trend_era[sort_idx][np.argmin(np.abs(sort_cdf - 0.05))]
        stats['trend'][tname][1] = ((probability[ens_acc] / probability[ens_acc].sum()) * ens_trend_era).sum()
        stats['trend'][tname][2] = ens_trend_era[sort_idx][np.argmin(np.abs(sort_cdf - 0.95))]
        sort_idx = np.argsort(ens_accel_era)
        sort_cdf = np.cumsum((probability[ens_acc] / probability[ens_acc].sum())[sort_idx])
        stats['accel'][tname][0] = ens_accel_era[sort_idx][np.argmin(np.abs(sort_cdf - 0.05))]
        stats['accel'][tname][1] = ((probability[ens_acc] / probability[ens_acc].sum()) * ens_accel_era).sum()
        stats['accel'][tname][2] = ens_accel_era[sort_idx][np.argmin(np.abs(sort_cdf - 0.95))]
    return(stats)

# Sea level
def compute_corr():
    print('   Corrections from GSL to RSL...')
    global settings, corr_grd_ensemble, corr_gia_ensemble
    # Read data
    corr_grd_ensemble = {}
    corr_grd_ensemble['altimetry'] = mp_empty_float([settings['num_ens'], settings['ntime']])
    corr_gia_ensemble = {}
    corr_gia_ensemble['altimetry'] = mp_empty_float([settings['num_ens'], settings['ntime']])

    pool = mp.Pool(settings['nproc'])
    out = pool.map(read_corr_ens, settings['ens_range'])
    corr_grd_ensemble['altimetry'][:, ~settings['time_mask_grace']] = np.nan
    corr_gia_ensemble['altimetry'][:, ~settings['time_mask_grace']] = np.nan

    corr_gia_stats = {}
    corr_grd_stats = {}
    corr_tot_stats = {}
    corr_gia_stats['altimetry'] = ensemble_stats(corr_gia_ensemble['altimetry'])
    corr_grd_stats['altimetry'] = ensemble_stats(corr_grd_ensemble['altimetry'])
    corr_tot_stats['altimetry'] = ensemble_stats(corr_grd_ensemble['altimetry']+corr_gia_ensemble['altimetry'])
    return(corr_gia_stats, corr_grd_stats,corr_tot_stats)

def read_corr_ens(ens):
    global settings, corr_grd_ensemble, corr_gia_ensemble
    fname = settings['dir_save_ens'] + 'mass_rsl_ens_' + str(ens).zfill(4) + '.nc'
    file_handle = Dataset(fname, 'r')
    file_handle.set_auto_mask(False)
    corr_grd_ensemble['altimetry'][ens, :] = file_handle.variables['corr_grd_altimetry'][:]
    corr_gia_ensemble['altimetry'][ens, :] = file_handle.variables['corr_gia_altimetry'][:]
    file_handle.close()
    return

###### HELPER FUNCS
def read_probability():
    global settings
    probability = Dataset(settings['fn_gia_ens_rad'],'r').variables['probability'][settings['ens_range']]._get_data()
    probability /= probability.sum()
    return(probability)

def prepare_mask():
    # Read masks and glacier data for separation of land mass
    # changes into individual components
    print('  Reading mask data...')
    global settings, mask
    mask = {}
    mask = np.load(settings['fn_mask'],allow_pickle=True).all()
    # Ocean basin area
    mask['area'] = gentools.grid_area(settings['lat'],settings['lon'])
    mask['ocean_basin_area'] = np.zeros(6)
    for basin in range(6): mask['ocean_basin_area'][basin] = (mask['area']*(mask['basin'] == basin)).sum()
    return

# Parallel processing routines
def mp_empty_float(shape):
    shared_array_base = mp.RawArray(ct.c_float, int(np.prod(shape)))
    shared_array = np.ctypeslib.as_array(shared_array_base).reshape(*shape)
    return shared_array

def mp_empty_int(shape):
    shared_array_base = mp.RawArray(ct.c_int, int(np.prod(shape)))
    shared_array = np.ctypeslib.as_array(shared_array_base).reshape(*shape)
    return shared_array

def mp_empty_bool(shape):
    shared_array_base = mp.RawArray(ct.c_bool, int(np.prod(shape)))
    shared_array = np.ctypeslib.as_array(shared_array_base).reshape(*shape)
    return shared_array

def mp_filled_float(input_array):
    shape = input_array.shape
    shared_array_base = mp.RawArray(ct.c_float, input_array.flatten())
    shared_array = np.ctypeslib.as_array(shared_array_base).reshape(*shape)
    return shared_array

def mp_filled_int(input_array):
    shape = input_array.shape
    shared_array_base = mp.RawArray(ct.c_int, input_array.flatten())
    shared_array = np.ctypeslib.as_array(shared_array_base).reshape(*shape)
    return shared_array

def mp_filled_bool(input_array):
    shape = input_array.shape
    shared_array_base = mp.RawArray(ct.c_bool, input_array.flatten())
    shared_array = np.ctypeslib.as_array(shared_array_base).reshape(*shape)
    return shared_array