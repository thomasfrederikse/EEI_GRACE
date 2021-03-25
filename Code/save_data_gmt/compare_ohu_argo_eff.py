# -------------------------------------------------------------------
# Compare OHC OHU estimates derived from Argo using:
# 1. OHC directly derived from temperature and specific heat capacity
# 2. OHC from steric and a prescribed expansion efficiency
# Write the files for GMT plotting
# -------------------------------------------------------------------
import numpy as np
import mod_budget_grace_settings

def main():
    mod_budget_grace_settings.init()
    from mod_budget_grace_settings import settings
    global settings

    OHC_stats = np.load(settings['fn_OHC_stats'],allow_pickle=True).all()
    OHU_stats = np.load(settings['fn_OHU_stats'],allow_pickle=True).all()

    # Save data
    gmt_save_tseries_poly(settings['dir_gmt']+'compare_ohu_argo_eff/','ohu_spc',OHU_stats['hydrography']['years'],OHU_stats['hydrography']['tseries'])
    gmt_save_tseries_poly(settings['dir_gmt']+'compare_ohu_argo_eff/','ohu_eff',OHU_stats['hydrography_steric']['years'],OHU_stats['hydrography_steric']['tseries'])

    gmt_save_tseries_poly(settings['dir_gmt']+'compare_ohu_argo_eff/','ohc_spc',settings['time'],OHC_stats['hydrography']['altimetry']['tseries']/1e21)
    gmt_save_tseries_poly(settings['dir_gmt']+'compare_ohu_argo_eff/','ohc_eff',settings['time'],OHC_stats['hydrography_steric']['altimetry']['tseries']/1e21)
    return

def gmt_save_tseries(gmt_dir, fname, time, tseries):
    save_array = (np.array([time, tseries])).T
    np.savetxt(gmt_dir + fname + '_ts.txt', save_array, fmt='%4.3f;%4.3f')
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
