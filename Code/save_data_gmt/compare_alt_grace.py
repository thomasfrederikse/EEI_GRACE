# -------------------------------------------
# Write GMT scripts for compare_alt_grace.pdf
# ---
# Altimetry estimates
# - NASA measures (Our estimate)
# - AVISO
# - Colorado
# - CSIRO
# - ESA CCI
# - NOAA
# ---
# GRACE estimates
# - JPL mascons
# - CSR mascons
# - GSFC mascons
# - JPL/CSR/GFZ SPH solutions
# - SWARM
# -------------------------------------------
import numpy as np
import os
import mod_budget_grace_settings
from importlib import *
import mod_gentools as gentools
from netCDF4 import Dataset
import h5py

def main():
    mod_budget_grace_settings.init()
    from mod_budget_grace_settings import settings
    global settings
    global_basin_stats = np.load(settings['fn_sealevel_stats'],allow_pickle=True).all()
    altimetry = read_altimetry_gmsl()
    ocean_mass = read_ocean_mass(global_basin_stats)

    # Save altimetry
    for idx, alt in enumerate(altimetry):
        if alt == 'AVISO':
            gmt_save_tseries_poly(settings['dir_gmt']+'mass_alt_comp/', alt, settings['time'], altimetry[alt]['tseries'])
            gmt_save_trend_ci(settings['dir_gmt']+'mass_alt_comp/', alt, altimetry[alt]['trend_2005_2019'], idx+0.5)

        else:
            gmt_save_tseries(settings['dir_gmt']+'mass_alt_comp/', alt, settings['time'], altimetry[alt]['tseries'])
            gmt_save_trend(settings['dir_gmt']+'mass_alt_comp/', alt, altimetry[alt]['trend_2005_2019'], idx+0.5)

    for mass in ocean_mass:
        if mass == 'JPL':
            gmt_save_tseries_poly(settings['dir_gmt']+'mass_alt_comp/', mass, settings['time'], ocean_mass[mass])
        elif mass == 'Bernd':
            for n in range(ocean_mass['Bernd'].shape[1]):
                gmt_save_tseries(settings['dir_gmt']+'mass_alt_comp/', mass+'_'+str(n), settings['time'], ocean_mass[mass][:,n])
        else:
            gmt_save_tseries(settings['dir_gmt']+'mass_alt_comp/', mass, settings['time'], ocean_mass[mass])
    return

def read_altimetry_gmsl():
    global settings
    altimetry = {}
    alt_list = ['AVISO','NOAA','CSIRO','MEASURES']
    for i in alt_list: altimetry[i] = {}


    # GSFC
    gsfc = np.loadtxt(settings['dir_data']+'Altimetry/MEASURES/GMSL_TPJAOS_4.2_199209_202003.txt',comments='HDR',usecols=(2,7))
    gsfc[:,1] = gentools.remove_seasonal(gsfc[:,0],gsfc[:,1])
    hgt_int = np.interp(settings['time'],gsfc[:,0],gsfc[:,1],right=gsfc[-1,1])
    hgt_int-=hgt_int[-12:].mean()
    altimetry['MEASURES']['tseries'] = hgt_int

    # CSIRO
    fh = Dataset(settings['dir_data']+'Altimetry/CSIRO/jb_iby_sry_gtn_gin.nc','r')
    fh.set_auto_mask(False)
    time = fh.variables['time'][:]/ 365.25 + 1990
    hgt  = fh.variables['gmsl'][:]
    fh.close()
    hgt_int = np.interp(settings['time'],time,hgt,right=np.nan)
    hgt_int-=hgt_int[-12:].mean()
    altimetry['CSIRO']['tseries'] = hgt_int

    # NOAA
    missions = ['tx','e2','g1','j1','n1','j2','j3']
    hgt = np.zeros([settings['ntime'],len(missions)])*np.nan
    fh = Dataset(settings['dir_data']+'Altimetry/GMSL/slr_sla_gbl_free_all_66.nc','r')
    fh.set_auto_mask(False)
    for idx, m in enumerate(missions):
        time = fh.variables['time_'+m][:]
        hgt_lcl  = fh.variables['sla_'+m][:]
        hgt[:,idx] = np.interp(settings['time'], time, hgt_lcl, right=np.nan,left=np.nan)
    fh.close()
    hgt_sum = np.nanmean(hgt,axis=1)
    hgt_sum-=hgt_sum[-12:].mean()
    altimetry['NOAA']['tseries'] = hgt_sum

    # AVISO
    global_basin_stats = np.load(settings['fn_sealevel_stats'],allow_pickle=True).all()
    altimetry['AVISO']['tseries'] = global_basin_stats['gsl']['altimetry']['tseries']
    altimetry['AVISO']['trend_2005_2019'] = global_basin_stats['gsl']['altimetry']['trend']['2005-2019']
    # Trends
    for alt in alt_list:
        if alt != 'AVISO':
            altimetry[alt]['trend_2005_2019'] = gentools.lsqtrend(settings['time'][24:],altimetry[alt]['tseries'][24:])
    return(altimetry)

def read_ocean_mass(global_basin_stats):
    global settings
    ocean_mass = {}

    # JPL mascons
    ocean_mass['JPL'] = global_basin_stats['mass']['altimetry']['tseries'][:]

    # CSR mascons
    fh = Dataset(settings['dir_data']+'/GRACE/CSR_mascon/CSR_GRACE_GRACE-FO_RL06_Mascons_all-corrections_v02.nc','r')
    fh.set_auto_mask(False)
    time = 2002 + fh.variables['time'][:]/365.25
    lat = fh.variables['lat'][:]
    lon = fh.variables['lon'][:]
    ewh = fh.variables['lwe_thickness'][:] * 10
    fh.close()
    mask = Dataset(settings['dir_data'] + '/GRACE/CSR_mascon/CSR_GRACE_RL06_Mascons_v02_OceanMask.nc', 'r').variables['LO_val'][:]._get_data()
    area = gentools.grid_area(lat,lon)
    ocean_mass_raw = -((area*(1-mask))[np.newaxis,...] * ewh).sum(axis=(1,2))/(area*mask).sum()
    ocean_mass_raw = gentools.remove_seasonal(time,ocean_mass_raw)
    ocean_mass_int = np.interp(settings['time'],time,ocean_mass_raw,right=np.nan,left=np.nan)
    ocean_mass_int[~settings['time_mask_grace']] = np.nan
    ocean_mass['CSR'] = ocean_mass_int - np.nanmean(ocean_mass_int[-12:])

    # GSFC mascons
    gsfc = {}
    fh = h5py.File(settings['dir_data']+'GRACE/GSFC_mascon/GSFC.glb.200301_201607_v02.4-ICE6G.h5', 'r')
    gsfc['time'] = fh['/time/yyyy_doy_yrplot_middle'][2, :]
    gsfc['location'] = fh['/mascon/location'][0, :]
    gsfc['basin'] = fh['/mascon/basin'][0, :]
    gsfc['area'] = 1e6 * fh['/mascon/area_km2'][0, :]
    gsfc['lon_center'] = fh['/mascon/lon_center'][0, :]
    gsfc['lat_center'] = fh['/mascon/lat_center'][0, :]
    gsfc['lon_span'] = fh['/mascon/lon_span'][0, :]
    gsfc['lat_span'] = fh['/mascon/lat_span'][0, :]
    gsfc['ewh'] = 10 * fh['solution/cmwe'][:]
    fh.close()
    loc_land = [1, 3, 5, 80]
    loc_ocn = [90]
    basin_caspiansea = [4]
    gsfc['mask_land'] = (np.in1d(gsfc['location'], loc_land)) | ((np.in1d(gsfc['location'], loc_ocn)) & (gsfc['basin'] == basin_caspiansea))
    gsfc['ts_ocn'] = np.zeros(len(gsfc['time']))
    for mscn in range(len(gsfc['location'])):
        if gsfc['mask_land'][mscn]:
            gsfc['ts_ocn'] += gsfc['area'][mscn] * gsfc['ewh'][mscn, :]
    gsfc['ts_ocn']/=-gsfc['area'][~gsfc['mask_land']].sum()
    ocean_mass_raw = gentools.remove_seasonal(gsfc['time'],gsfc['ts_ocn'])
    ocean_mass_int = np.interp(settings['time'],gsfc['time'],ocean_mass_raw,right=np.nan,left=np.nan)
    ocean_mass['GSFC'] =ocean_mass_int - np.nanmean(ocean_mass_int[-48:-36]) + np.nanmean(ocean_mass['JPL'][-48:-36,1])

    # # Uebbing
    # dir_bernd = os.getenv('HOME') + '/Data/GRACE/Ocean_mass_Bernd/'
    # list_files = os.listdir(dir_bernd)
    # gia_model = ['A2013', 'Caron2018', 'ICE6GC', 'ICE6GD', 'Klemann2009', 'Paulson2007', 'Rietbroek2016']
    # proc_ctr = ['CSR', 'GFZ', 'JPL', 'ITSG18']
    # oc_mass = np.zeros([len(settings['time']),len(gia_model)*len(proc_ctr)])
    # n=0
    # for i in proc_ctr:
    #     for j in gia_model:
    #         fn = dir_bernd+i+'_GRACE_RL06_ocean_mass_gia_'+j+'.txt'
    #         raw_data = np.loadtxt(fn)
    #         ocean_mass_raw = np.interp(settings['time'], raw_data[:, 0], gentools.remove_seasonal(raw_data[:, 0], raw_data[:, 1]), right=np.nan, left=np.nan)
    #         ocean_mass_raw -= ocean_mass_raw[24:36].mean()
    #         oc_mass[:,n] = ocean_mass_raw
    #         n+=1
    # ocean_mass['Bernd'] = oc_mass

    # SWARM (Christina Lueck)
    raw_data = np.loadtxt(settings['dir_data']+'GRACE/Ocean_mass_Bernd/oceanMassSwarm.txt')
    ocean_mass_raw = np.interp(settings['time'], raw_data[:, 0], gentools.remove_seasonal(raw_data[:, 0], raw_data[:, 1]), right=np.nan, left=np.nan)
    acc_idx = np.isfinite(ocean_mass['JPL'][:,1]) & np.isfinite(ocean_mass_raw)

    ocean_mass_raw = ocean_mass_raw - ocean_mass_raw[acc_idx].mean() + ocean_mass['JPL'][acc_idx,1].mean()
    ocean_mass['SWARM'] = ocean_mass_raw

    # Read Felix'
    data_raw = np.genfromtxt(settings['dir_data']+'GRACE/SPH/OcM_FWL.txt',skip_header=1,delimiter=';')
    ocean_mass['JPL_sph'] = np.zeros(len(settings['time']))*np.nan
    ocean_mass['CSR_sph'] = np.zeros(len(settings['time']))*np.nan

    ocean_mass['JPL_sph'][np.isfinite(ocean_mass['JPL'][:,0])] = np.interp(settings['time'][np.isfinite(ocean_mass['JPL'][:,0])],data_raw[:,0], gentools.remove_seasonal(data_raw[:,0],data_raw[:,2]))
    ocean_mass['CSR_sph'][np.isfinite(ocean_mass['JPL'][:,0])] = np.interp(settings['time'][np.isfinite(ocean_mass['JPL'][:,0])],data_raw[:,0], gentools.remove_seasonal(data_raw[:,0],data_raw[:,3]))

    ocean_mass['JPL_sph']-=np.nanmean(ocean_mass['JPL_sph'][-12:])
    ocean_mass['CSR_sph']-=np.nanmean(ocean_mass['CSR_sph'][-12:])
    return(ocean_mass)

def gmt_save_tseries(gmt_dir, fname, time, tseries):
    # Save mean and confidence intervals
    save_array = (np.array([time, tseries])).T
    np.savetxt(gmt_dir + fname + '_ts.txt', save_array, fmt='%4.3f;%4.3f')
    return

def gmt_save_trend(gmt_dir, fname, trend, position):
    save_array = np.array([position,trend])
    np.savetxt(gmt_dir + fname + '_trend.txt', save_array[np.newaxis,:], fmt='%4.3f;%4.3f')
    return

def gmt_save_trend_ci(gmt_dir,fname,trend,position):
    save_array = np.array([position,trend[1],trend[0]-trend[1],trend[2]-trend[1]])
    np.savetxt(gmt_dir + fname + '_trend.txt', save_array[np.newaxis,:], fmt='%4.3f;%4.3f;%4.3f;%4.3f')
    return


def gmt_save_tseries_poly(gmt_dir, fname, time, tseries):
    acc_idx = np.isfinite(tseries[:, 0])
    # Save mean and confidence intervals
    save_array_ci = (np.array([time, tseries[:, 1], tseries[:, 0], tseries[:, 2]])).T
    save_array_mean = np.array([time, tseries[:, 1]]).T
    if (~acc_idx).sum() > 0:
        arr_list = np.split(save_array_ci, np.where(~acc_idx)[0])
    else:
        arr_list = [save_array_ci]
    n = 0
    for lst in arr_list:
        if lst.shape[0] > 1:
            acc_idx2 = np.isfinite(lst[:, 1])
            np.savetxt(gmt_dir + fname + '_' + str(n) + '.txt', lst[acc_idx2, :], fmt='%4.3f;%4.3f;%4.3f;%4.3f')
            n += 1
    return
