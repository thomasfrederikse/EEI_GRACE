# ----------------------------------
# Create save_data_gmt input for budget_ohc_ts
# - Budget
#  * RSL
#  * GRACE
#  * Argo
#  * Alt-GRACE
#  * Residual
# - OHC
#  * CERES
#  * Argo
#  * Alt-GRCAE
#  * Alt-GRACE-Argo
# ---------------------------------
import numpy as np
import mod_budget_grace_settings

def main():
    mod_budget_grace_settings.init()
    from mod_budget_grace_settings import settings
    global settings
    sealevel_stats = np.load(settings['fn_sealevel_stats'],allow_pickle=True).all()
    corr_gia_stats = np.load(settings['fn_corr_gia_stats'], allow_pickle=True).all()
    corr_grd_stats = np.load(settings['fn_corr_grd_stats'], allow_pickle=True).all()
    corr_tot_stats = np.load(settings['fn_corr_tot_stats'], allow_pickle=True).all()

    OHC_stats = np.load(settings['fn_OHC_stats'],allow_pickle=True).all()
    OHU_stats = np.load(settings['fn_OHU_stats'],allow_pickle=True).all()
    other_terms = np.load(settings['fn_other_terms'],allow_pickle=True).all()

    # Budget time series
    gmt_save_tseries_poly(settings['dir_gmt']+'budget_ohc_ts/','budget_rsl',settings['time'],sealevel_stats['rsl']['altimetry']['tseries'])
    gmt_save_tseries_poly(settings['dir_gmt']+'budget_ohc_ts/','budget_mass',settings['time'],sealevel_stats['mass']['altimetry']['tseries'])
    gmt_save_tseries_poly(settings['dir_gmt']+'budget_ohc_ts/','budget_steric',settings['time'],sealevel_stats['steric']['altimetry']['tseries'])
    gmt_save_tseries_poly(settings['dir_gmt']+'budget_ohc_ts/','budget_rsl_min_mass',settings['time'],sealevel_stats['rsl_min_mass']['altimetry']['tseries'])
    gmt_save_tseries_poly(settings['dir_gmt']+'budget_ohc_ts/','budget_diff',settings['time'],sealevel_stats['diff']['altimetry']['tseries'])
    gmt_save_tseries_poly(settings['dir_gmt']+'budget_ohc_ts/','budget_desbruyeres',settings['time'],other_terms['deep_ocean']['steric_ts'])

    gmt_save_trend_ci(settings['dir_gmt']+'budget_ohc_ts/','budget_rsl', 0.5, sealevel_stats['rsl']['altimetry']['trend']['2005-2019'])
    gmt_save_trend_ci(settings['dir_gmt']+'budget_ohc_ts/','budget_mass', 1.5, sealevel_stats['mass']['altimetry']['trend']['2005-2019'])
    gmt_save_trend_ci(settings['dir_gmt']+'budget_ohc_ts/','budget_rsl_min_mass', 2.5, sealevel_stats['rsl_min_mass']['altimetry']['trend']['2005-2019'])
    gmt_save_trend_ci(settings['dir_gmt']+'budget_ohc_ts/','budget_steric', 3.5, sealevel_stats['steric']['altimetry']['trend']['2005-2019'])
    gmt_save_trend_ci(settings['dir_gmt']+'budget_ohc_ts/','budget_diff', 4.5, sealevel_stats['diff']['altimetry']['trend']['2005-2019'])
    gmt_save_trend_ci(settings['dir_gmt']+'budget_ohc_ts/','budget_desbruyeres', 5.5, other_terms['deep_ocean']['steric_trend'])

    # OHC time series
    gmt_save_tseries_poly(settings['dir_gmt']+'budget_ohc_ts/','ohc_shallow',settings['time'],OHC_stats['hydrography']['altimetry']['tseries']/1e21)
    gmt_save_tseries_poly(settings['dir_gmt']+'budget_ohc_ts/','ohc_full',settings['time'],OHC_stats['altmass_total']['altimetry']['tseries']/1e21)
    gmt_save_tseries_poly(settings['dir_gmt']+'budget_ohc_ts/','ohc_deep',settings['time'],OHC_stats['altmass_deep']['altimetry']['tseries']/1e21)
    gmt_save_tseries_poly(settings['dir_gmt']+'budget_ohc_ts/','ohc_desbruyeres',settings['time'],other_terms['deep_ocean']['ohc_ts']/1e21)

    gmt_save_trend_ci(settings['dir_gmt']+'budget_ohc_ts/','ohc_shallow', 1.5, OHC_stats['hydrography']['altimetry']['trend']['2005-2019']/1e21)
    gmt_save_trend_ci(settings['dir_gmt']+'budget_ohc_ts/','ohc_full', 0.5, OHC_stats['altmass_total']['altimetry']['trend']['2005-2019']/1e21)
    gmt_save_trend_ci(settings['dir_gmt']+'budget_ohc_ts/','ohc_deep', 2.5, OHC_stats['altmass_deep']['altimetry']['trend']['2005-2019']/1e21)
    gmt_save_trend_ci(settings['dir_gmt']+'budget_ohc_ts/','ohc_desbruyeres', 3.5, other_terms['deep_ocean']['ohc_trend']/1e21)

    # OHU time series
    gmt_save_tseries_poly(settings['dir_gmt']+'budget_ohc_ts/','ohu_hydro',OHU_stats['hydrography']['years'][:],OHU_stats['hydrography']['tseries'][:,:])
    gmt_save_tseries_poly(settings['dir_gmt']+'budget_ohc_ts/','ohu_altmass',OHU_stats['altmass']['years'][:],OHU_stats['altmass']['tseries'][:,:])
    gmt_save_tseries(settings['dir_gmt']+'budget_ohc_ts/','ohu_ceres',other_terms['CERES']['years'],other_terms['CERES']['eei']-0.08)

    # Deep OHU
    OHU_deep = other_terms['deep_ocean']['ohc_trend'] / (4 * np.pi * 6371000 ** 2) / (3600*24*365.25)
    gmt_save_tseries_poly(settings['dir_gmt']+'budget_ohc_ts/','ohu_deep_desbruyeres',other_terms['CERES']['years'],OHU_deep[np.newaxis,:]*np.ones([len(other_terms['CERES']['years']),3]))

    # OHU trends
    gmt_save_trend_ci(settings['dir_gmt']+'budget_ohc_ts/','ohu_hydro', 1.5, OHU_stats['hydrography']['trend']['2005-2019'])
    gmt_save_trend_ci(settings['dir_gmt']+'budget_ohc_ts/','ohu_altmass', 0.5, OHU_stats['altmass']['trend']['2005-2019'])
    gmt_save_trend(settings['dir_gmt']+'budget_ohc_ts/','ohu_ceres', 2.5, other_terms['CERES']['trend']['2005-2019'])
    return

def gmt_save_tseries_poly(gmt_dir,fname,time,tseries):
    acc_idx = np.isfinite(tseries[:, 0])
    # Save mean and confidence intervals
    save_array_ci  = (np.array([time, tseries[:, 1],tseries[:, 0],tseries[:, 2]])).T
    if (~acc_idx).sum()>0: arr_list = np.split(save_array_ci, np.where(~acc_idx)[0])
    else: arr_list = [save_array_ci]
    n=0
    for lst in arr_list:
        if lst.shape[0]>1:
            acc_idx2 = np.isfinite(lst[:,1])
            np.savetxt(gmt_dir + fname + '_'+str(n)+'.txt', lst[acc_idx2,:], fmt='%4.3f;%4.3f;%4.3f;%4.3f')
            n+=1
    return

def gmt_save_tseries(gmt_dir, fname, time, tseries):
    save_array = (np.array([time, tseries])).T
    np.savetxt(gmt_dir + fname + '_ts.txt', save_array, fmt='%4.3f;%4.3f')
    return

def gmt_save_trend(gmt_dir,fname,position,trend):
    save_array = np.array([position,trend])
    np.savetxt(gmt_dir + fname + '_trend.txt', save_array[np.newaxis,:], fmt='%4.3f;%4.3f')
    return

def gmt_save_trend_ci(gmt_dir,fname,position,trend):
    save_array = np.array([position,trend[1],trend[0]-trend[1],trend[2]-trend[1]])
    np.savetxt(gmt_dir + fname + '_trend.txt', save_array[np.newaxis,:], fmt='%4.3f;%4.3f;%4.3f;%4.3f')
    return