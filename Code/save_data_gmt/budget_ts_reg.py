# Write GMT scripts for budget_ts_reg.pdf
import numpy as np
import os
import mod_budget_grace_settings
from importlib import *
def main():
    mod_budget_grace_settings.init()
    from mod_budget_grace_settings import settings
    global settings

    # Load global budget data
    global_basin_stats = np.load(settings['fn_sealevel_stats'],allow_pickle=True).all()
    for basin in range(len(settings['region_names'])):
        gmt_save_tseries_poly(settings['dir_gmt']+'budget_ts_reg/','rsl_'+str(basin),settings['time'],global_basin_stats['rsl']['basin'][basin]['tseries'])
        gmt_save_tseries_poly(settings['dir_gmt']+'budget_ts_reg/','mass_'+str(basin),settings['time'],global_basin_stats['mass']['basin'][basin]['tseries'])
        gmt_save_tseries_poly(settings['dir_gmt']+'budget_ts_reg/','steric_'+str(basin),settings['time'],global_basin_stats['steric']['basin'][basin]['tseries'])
        gmt_save_tseries_poly(settings['dir_gmt']+'budget_ts_reg/','rsl_min_mass_'+str(basin),settings['time'],global_basin_stats['rsl_min_mass']['basin'][basin]['tseries'])
        gmt_save_tseries_poly(settings['dir_gmt']+'budget_ts_reg/','diff_'+str(basin),settings['time'],global_basin_stats['diff']['basin'][basin]['tseries'])

        # Trends
        gmt_save_trend_ci(settings['dir_gmt']+'budget_ts_reg/','rsl_'+str(basin),0.5,global_basin_stats['rsl']['basin'][basin]['trend']['2005-2019'])
        gmt_save_trend_ci(settings['dir_gmt']+'budget_ts_reg/','mass_'+str(basin),1.5,global_basin_stats['mass']['basin'][basin]['trend']['2005-2019'])
        gmt_save_trend_ci(settings['dir_gmt']+'budget_ts_reg/','rsl_min_mass_'+str(basin),2.5,global_basin_stats['rsl_min_mass']['basin'][basin]['trend']['2005-2019'])
        gmt_save_trend_ci(settings['dir_gmt']+'budget_ts_reg/','steric_'+str(basin),3.5,global_basin_stats['steric']['basin'][basin]['trend']['2005-2019'])
        gmt_save_trend_ci(settings['dir_gmt']+'budget_ts_reg/','diff_'+str(basin),4.5,global_basin_stats['diff']['basin'][basin]['trend']['2005-2019'])
    return


def gmt_save_tseries_poly(gmt_dir,fname,time,tseries):
    acc_idx = np.isfinite(tseries[:, 0])
    # Save mean and confidence intervals
    save_array_ci  = (np.array([time, tseries[:, 1],tseries[:, 0],tseries[:, 2]])).T
    save_array_mean = np.array([time, tseries[:, 1]]).T
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
    # Save mean and confidence intervals
    save_array = (np.array([time, tseries])).T
    np.savetxt(gmt_dir + fname + '_ts.txt', save_array, fmt='%4.3f;%4.3f')
    return

def gmt_save_trend(gmt_dir,fname,position,trend):
    save_array = np.array([position,trend])
    np.savetxt(gmt_dir + fname + '_trend.txt', save_array[np.newaxis,:], fmt='%4.3f;%4.3f')
    return


def gmt_save_trend_ci(gmt_dir,fname,position,trend):
    save_array = np.array([trend[1],position,trend[0]-trend[1],trend[2]-trend[1]])
    np.savetxt(gmt_dir + fname + '_trend.txt', save_array[np.newaxis,:], fmt='%4.3f;%4.3f;%4.3f;%4.3f')
    return


