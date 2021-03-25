# Write GMT scripts for mass_ctb.pdf
import numpy as np
import os
import mod_budget_grace_settings
from importlib import *
import mod_gentools as gentools
from scipy.interpolate import interp1d
from netCDF4 import Dataset
def main():
    mod_budget_grace_settings.init()
    from mod_budget_grace_settings import settings
    global settings
    this_study()
    zemp()
    bamber()
    imbie()
    mouginot()
    return

def this_study():
    global settings
    global_basin_stats = np.load(settings['fn_sealevel_stats'],allow_pickle=True).all()
    gmt_save_tseries_poly(settings['dir_gmt']+'mass_ctb/','glac',settings['time'],-global_basin_stats['mass_ctb']['mass_glac']['tseries'])
    gmt_save_tseries_poly(settings['dir_gmt']+'mass_ctb/','GrIS',settings['time'],-global_basin_stats['mass_ctb']['mass_GrIS']['tseries'])
    gmt_save_tseries_poly(settings['dir_gmt']+'mass_ctb/','AIS',settings['time'],-global_basin_stats['mass_ctb']['mass_AIS']['tseries'])
    gmt_save_tseries_poly(settings['dir_gmt']+'mass_ctb/','tws',settings['time'],-global_basin_stats['mass_ctb']['mass_tws']['tseries'])
    return

def zemp():
    global_basin_stats = np.load(settings['fn_sealevel_stats'],allow_pickle=True).all()
    Zemp_dir = settings['dir_data']+'Glaciers/Zemp_2019/'
    zemp_flist = os.listdir(Zemp_dir)
    acc_regions = [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18]
    zemp_rate  = np.zeros([len(acc_regions),17])
    zemp_sterr = np.zeros([len(acc_regions),17])
    for idx,num in enumerate(acc_regions):
        fname = Zemp_dir+[i for i in zemp_flist if '_'+str(num)+'_' in i][0]
        raw_data = np.loadtxt(fname,skiprows=28,delimiter=',')
        years      = raw_data[:,0]
        acc_idx = np.in1d(years,np.arange(2000,2017))
        zemp_rate[idx,:] = raw_data[acc_idx,10] # Rate in gigatons
        zemp_sterr[idx,:] = raw_data[acc_idx,18]/1.96 # Uncertainty in gigatons
    zemp_rate_tot = np.sum(zemp_rate,axis=0)
    zemp_sterr_tot = np.sqrt(np.sum(zemp_sterr**2,axis=0))
    random_mb = np.cumsum(zemp_rate_tot[:,np.newaxis] + np.random.randn(zemp_rate_tot.shape[0],5000)*zemp_sterr_tot[:,np.newaxis],axis=0)
    zemp_t   = np.arange(2001, 2018)+0.5
    random_mb_interp = interp1d(zemp_t,random_mb,axis=0,kind='linear',fill_value='extrapolate')(settings['time'][:-36])
    random_mb_interp = (random_mb_interp - np.mean(random_mb_interp[-12:,:],axis=0)[np.newaxis,:])/-362 - global_basin_stats['mass_ctb']['mass_glac']['tseries'][:,1][-48:-36].mean()
    tseries = np.zeros([settings['ntime'],3]) * np.nan
    t_acc = settings['time'] < 2017
    tseries[t_acc,0] = np.percentile(random_mb_interp,5,axis=1)
    tseries[t_acc,1] = np.mean(random_mb_interp,axis=1)
    tseries[t_acc,2] = np.percentile(random_mb_interp,95,axis=1)

    gmt_save_tseries_poly(settings['dir_gmt']+'mass_ctb/','zemp', settings['time'], tseries)
    return

def bamber():
    global settings
    global_basin_stats = np.load(settings['fn_sealevel_stats'],allow_pickle=True).all()

    bamber_raw = np.loadtxt(os.getenv('HOME')+'/Data/IceSheets/Bamber_2018/Bamber-etal_2018.tab',delimiter='	',skiprows=21)
    t_acc = settings['time'] < 2017
    random_mb_GrIS = np.cumsum(bamber_raw[:,1][:,np.newaxis] + np.random.randn(bamber_raw[:,1].shape[0],5000)*bamber_raw[:,2][:,np.newaxis],axis=0)
    random_mb_AIS = np.cumsum(bamber_raw[:,3][:,np.newaxis] + bamber_raw[:,5][:,np.newaxis] + np.random.randn(bamber_raw[:,1].shape[0],5000)*bamber_raw[:,4][:,np.newaxis] + np.random.randn(bamber_raw[:,1].shape[0],5000)*bamber_raw[:,6][:,np.newaxis],axis=0)
    random_mb_glac = np.cumsum(bamber_raw[:,7][:,np.newaxis] + np.random.randn(bamber_raw[:,8].shape[0],5000)*bamber_raw[:,8][:,np.newaxis],axis=0)
    time_cum = bamber_raw[:,0] + 0.5
    random_mb_GrIS = interp1d(time_cum,random_mb_GrIS,axis=0,kind='linear',fill_value='extrapolate')(settings['time'][t_acc])
    random_mb_AIS = interp1d(time_cum,random_mb_AIS,axis=0,kind='linear',fill_value='extrapolate')(settings['time'][t_acc])
    random_mb_glac = interp1d(time_cum,random_mb_glac,axis=0,kind='linear',fill_value='extrapolate')(settings['time'][t_acc])
    random_mb_GrIS = (random_mb_GrIS - np.mean(random_mb_GrIS[-12:,:],axis=0)[np.newaxis,:])/-362  - global_basin_stats['mass_ctb']['mass_GrIS']['tseries'][:,1][-48:-36].mean()
    random_mb_AIS = (random_mb_AIS - np.mean(random_mb_AIS[-12:,:],axis=0)[np.newaxis,:])/-362     - global_basin_stats['mass_ctb']['mass_AIS']['tseries'][:,1][-48:-36].mean()
    random_mb_glac = (random_mb_glac - np.mean(random_mb_glac[-12:,:],axis=0)[np.newaxis,:])/-362  - global_basin_stats['mass_ctb']['mass_glac']['tseries'][:,1][-48:-36].mean()
    glac = np.zeros([settings['ntime'],3]) * np.nan
    GrIS = np.zeros([settings['ntime'],3]) * np.nan
    AIS = np.zeros([settings['ntime'],3]) * np.nan

    glac[t_acc,0] = np.percentile(random_mb_glac,5,axis=1)
    glac[t_acc,1] = np.mean(random_mb_glac,axis=1)
    glac[t_acc,2] = np.percentile(random_mb_glac,95,axis=1)

    GrIS[t_acc,0] = np.percentile(random_mb_GrIS,5,axis=1)
    GrIS[t_acc,1] = np.mean(random_mb_GrIS,axis=1)
    GrIS[t_acc,2] = np.percentile(random_mb_GrIS,95,axis=1)

    AIS[t_acc,0] = np.percentile(random_mb_AIS,5,axis=1)
    AIS[t_acc,1] = np.mean(random_mb_AIS,axis=1)
    AIS[t_acc,2] = np.percentile(random_mb_AIS,95,axis=1)

    gmt_save_tseries_poly(settings['dir_gmt']+'mass_ctb/','bamber_glac', settings['time'], glac)
    gmt_save_tseries_poly(settings['dir_gmt']+'mass_ctb/','bamber_GrIS', settings['time'], GrIS)
    gmt_save_tseries_poly(settings['dir_gmt']+'mass_ctb/','bamber_AIS', settings['time'], AIS)
    return

def mouginot():
    global settings
    global_basin_stats = np.load(settings['fn_sealevel_stats'],allow_pickle=True).all()
    t_acc = settings['time'] < 2019
    raw_data  = np.loadtxt(settings['dir_data']+'IceSheets/Mouginot_2019/GrIS_total.csv',delimiter=';')
    random_mb = np.cumsum(raw_data[:,1][:,np.newaxis] + np.random.randn(raw_data.shape[0],5000)*raw_data[:,2][:,np.newaxis],axis=0)
    time_cum = raw_data[:,0] + 0.5
    random_mb_interp = interp1d(time_cum,random_mb,axis=0,kind='linear',fill_value='extrapolate')(settings['time'][t_acc])
    random_mb_interp = (random_mb_interp - np.mean(random_mb_interp[-12:,:],axis=0)[np.newaxis,:])/-362 - np.nanmean(global_basin_stats['mass_ctb']['mass_GrIS']['tseries'][:,1][-24:-12])
    GrIS = np.zeros([settings['ntime'],3]) * np.nan
    GrIS[t_acc,0] = np.percentile(random_mb_interp,5,axis=1)
    GrIS[t_acc,1] = np.mean(random_mb_interp,axis=1)
    GrIS[t_acc,2] = np.percentile(random_mb_interp,95,axis=1)
    gmt_save_tseries_poly(settings['dir_gmt']+'mass_ctb/','mouginot', settings['time'], GrIS)
    return

def imbie():
    global settings
    global_basin_stats = np.load(settings['fn_sealevel_stats'],allow_pickle=True).all()
    t_acc = settings['time'] < 2017.5
    raw_data = np.loadtxt(settings['dir_data']+'IceSheets/IMBIE/IMBIE2.csv',delimiter=';')
    acc_idx = (raw_data[:,0]>=2002.99)
    t_imbie = raw_data[acc_idx,0]
    m_imbie = raw_data[acc_idx,1]
    s_imbie = raw_data[acc_idx,2]
    m_imbie = m_imbie - np.mean(m_imbie[-30:]) - np.nanmean(global_basin_stats['mass_ctb']['mass_AIS']['tseries'][:,1][-60:-30])
    AIS = np.zeros([settings['ntime'],3]) * np.nan
    AIS[t_acc,0] = m_imbie-s_imbie
    AIS[t_acc,1] = m_imbie
    AIS[t_acc,2] = m_imbie+s_imbie
    gmt_save_tseries_poly(settings['dir_gmt']+'mass_ctb/','IMBIE_AIS', settings['time'], AIS)
    return

def humphrey():
    global settings
    global_basin_stats = np.load(settings['fn_sealevel_stats'],allow_pickle=True).all()
    # Read Vincents ERA5 GSFC/JPL data
    fn = settings['dir_data'] + 'GRACE/JPL_mascon/mask.nc'
    fh = Dataset(fn, 'r')
    fh.set_auto_mask(False)
    mask_tws = fh.variables['land'][:] - fh.variables['GrIS'][:] - fh.variables['AIS'][:]
    mask_ocn = 1-fh.variables['land'][:]
    fh.close()
    area = gentools.grid_area(settings['lat'],settings['lon'])
    ewh_jpl = np.zeros([100,settings['ntime']])

    for ens in range(100):
        print(ens)
        fn = settings['dir_data'] + 'Hydrology/Humphrey/02_monthly_grids_ensemble_JPL_ERA5_1979-201907/GRACE_REC_v03_JPL_ERA5_monthly_ens'+str(ens+1).zfill(3)+'.nc'
        fh = Dataset(fn,'r')
        fh.set_auto_mask(False)
        ewh = fh.variables['rec_ensemble_member'][288:,...].astype(float)
        ewh[ewh==-32768] = np.nan
        ewh = np.dstack([ewh[:,:,360:],ewh[:,:,:360]])
        time =  fh.variables['time'][288:]/365.24+1901
        fh.close()
        ewh_glb = -np.nansum(ewh*(mask_tws*area)[np.newaxis,:,:],axis=(1,2)) / (area*mask_ocn).sum()
        ewh_glb = gentools.remove_seasonal(time, ewh_glb)
        ewh_glb = np.interp(settings['time'], time, ewh_glb, right=np.nan, left=np.nan)
        ewh_jpl[ens,:] = ewh_glb - ewh_glb[-24:-12].mean()  - np.nanmean(global_basin_stats['mass_ctb']['mass_tws']['tseries'][:,1][-24:-12])

    ewh_gsfc = np.zeros([100,settings['ntime']])
    for ens in range(100):
        print(ens)
        fn = settings['dir_data'] + 'Hydrology/Humphrey/02_monthly_grids_ensemble_GSFC_ERA5_1979-201907/GRACE_REC_v03_GSFC_ERA5_monthly_ens'+str(ens+1).zfill(3)+'.nc'
        fh = Dataset(fn,'r')
        fh.set_auto_mask(False)
        ewh = fh.variables['rec_ensemble_member'][288:,...].astype(float)
        ewh[ewh==-32768] = np.nan
        ewh = np.dstack([ewh[:,:,360:],ewh[:,:,:360]])
        time =  fh.variables['time'][288:]/365.24+1901
        fh.close()
        ewh_glb = -np.nansum(ewh*(mask_tws*area)[np.newaxis,:,:],axis=(1,2)) / (area*mask_ocn).sum()
        ewh_glb = gentools.remove_seasonal(time, ewh_glb)
        ewh_glb = np.interp(settings['time'], time, ewh_glb, right=np.nan, left=np.nan)
        ewh_gsfc[ens,:] = ewh_glb - ewh_glb[-24:-12].mean() - np.nanmean(global_basin_stats['mass_ctb']['mass_tws']['tseries'][:,1][-24:-12])

    # Stats
    ewh_humphrey = {}
    ewh_humphrey['jpl'] = np.percentile(ewh_jpl,[5,50,95],axis=0).T
    ewh_humphrey['gsfc'] = np.percentile(ewh_gsfc,[5,50,95],axis=0).T
    gmt_save_tseries_poly(settings['dir_gmt']+'mass_ctb/','Humphrey_JPL', settings['time'], ewh_humphrey['jpl'])
    gmt_save_tseries_poly(settings['dir_gmt']+'mass_ctb/','Humphrey_GSFC', settings['time'], ewh_humphrey['gsfc'])
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



