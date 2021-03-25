# --------------------------------------------------
# Estimate expansion efficiency from ECCO and ORAS5
# Obtain spread by computing 10-year efficiencies
# - Deep and shallow, averaged over altimetry domain
# - Compare to Argo
# --------------------------------------------------
import numpy as np
from netCDF4 import Dataset
import os
from scipy.interpolate import interp2d
import mod_gentools as gentools
import matplotlib.pyplot as plt

def main():
    settings = def_settings()

    # Read steric and OHC
    oras5 = read_oras5(settings)
    ecco  = read_ecco(settings)

    oras5 = comp_efficiency(oras5, settings)
    ecco  = comp_efficiency(ecco, settings)

    efficiency_obs,efficiency_ts = compute_obs_eff_indiv(settings)

    # Save data for processing
    eff_shallow_mean   = np.hstack([oras5['shallow']['alt']['eff_rolling'], ecco['shallow']['alt']['eff_rolling']]).mean()
    eff_shallow_sterr  = np.hstack([oras5['shallow']['alt']['eff_rolling'], ecco['shallow']['alt']['eff_rolling']]).std()
    eff_full_mean   = np.hstack([oras5['full']['alt']['eff_rolling'], ecco['full']['alt']['eff_rolling']]).mean()
    eff_full_sterr  = np.hstack([oras5['full']['alt']['eff_rolling'], ecco['full']['alt']['eff_rolling']]).std()

    # Save all efficiency data
    efficiency = {}
    efficiency['shallow'] = np.array([eff_shallow_mean,eff_shallow_sterr])
    efficiency['full']     = np.array([eff_full_mean,eff_full_sterr])
    efficiency['ecco'] = ecco
    efficiency['oras5'] = oras5
    efficiency['obs'] = {}
    efficiency['obs']['models'] = settings['steric_products']
    efficiency['obs']['trend_glb'] = efficiency_obs[:,0]
    efficiency['obs']['trend_alt'] = efficiency_obs[:,1]
    efficiency['obs']['ts_glb'] = efficiency_ts
    np.save(settings['fn_efficiency'],efficiency)

    # Plots
    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(-0.2,ecco['shallow']['alt']['eff_rolling'].mean(),label='Shallow',yerr=ecco['shallow']['alt']['eff_rolling'].std(),color='C0',width=0.3)
    ax.bar(0.2,ecco['full']['alt']['eff_rolling'].mean(),label='Full',yerr=ecco['full']['alt']['eff_rolling'].std(),color='C1',width=0.3)
    ax.bar(0.8,oras5['shallow']['alt']['eff_rolling'].mean(),yerr=oras5['shallow']['alt']['eff_rolling'].std(),color='C0',width=0.3)
    ax.bar(1.2,oras5['full']['alt']['eff_rolling'].mean(),yerr=oras5['full']['alt']['eff_rolling'].std(),color='C1',width=0.3)
    for i in range(len(efficiency_obs)):
        ax.bar(i+2, efficiency_obs[i,0],color='C0',width=0.3)
    ax.bar(10,efficiency_obs[:,0].mean(),yerr=efficiency_obs[:,0].std(),color='C0',width=0.5)
    ax.bar(10.8,eff_shallow_mean,yerr=eff_shallow_sterr,color='C0',width=0.4)
    ax.bar(11.2,eff_full_mean,yerr=eff_full_sterr,color='C1',width=0.4)
    ax.set_xticks(np.arange(0,12))
    ax.set_xticklabels(['ECCO','ORAS5']+settings['steric_products']+['Mean obs','Mean rean'])
    ax.grid()
    ax.legend()
    ax.set_ylabel('Efficiency (mm/J)')
    fig.tight_layout()
    fig.savefig('eff_bar.png')

    plt.figure(figsize=(8,6))
    plt.plot(oras5['shallow']['time'][:-settings['month_running']],oras5['shallow']['alt']['eff_rolling'],label='ORAS5',linewidth=4)
    plt.plot(ecco['shallow']['time'][:-settings['month_running']],ecco['shallow']['alt']['eff_rolling'],label='ECCO4',linewidth=4)
    for mod in efficiency_ts:
        plt.plot(efficiency_ts[mod]['time'],efficiency_ts[mod]['eff'],label=mod)
    plt.grid()
    plt.xlim([1990,2006])
    plt.ylim([1.0e-22,1.7e-22])
    plt.ylabel('Efficiency (mm/ZJ)')
    plt.xlabel('Start date (yr)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('rolling_efficiencies.png')
    return

def def_settings():
    settings = {}
    settings['dir_data']    = os.getenv('HOME')+'/Data/'
    settings['dir_oras']     = settings['dir_data'] + 'OceanModels/ORAS5/'
    settings['dir_ecco']     = settings['dir_data'] + 'ECCO/v4r4/'

    settings['fn_oras_grid'] = settings['dir_oras'] + 'mesh_mask.nc'
    settings['fn_oras_shallow'] = settings['dir_oras'] + 'ORAS5_steric_ohc_shallow.nc'
    settings['fn_oras_full'] = settings['dir_oras'] + 'ORAS5_steric_ohc_full.nc'

    settings['fn_ecco_shallow'] = settings['dir_ecco'] + 'ECCOv4r4_steric_ohc_shallow_1993_2017.nc'
    settings['fn_ecco_full']    = settings['dir_ecco'] + 'ECCOv4r4_steric_ohc_full_1993_2017.nc'

    settings['steric_products'] = ['EN4_l09','EN4_g10','I17','CZ16','WOA','SIO','JAMSTEC','BOA']
    settings['fn_EN4_l09'] = settings['dir_data'] + 'Steric/EN4/EN4_L09_1950_2019.nc'
    settings['fn_EN4_g10'] = settings['dir_data'] + 'Steric/EN4/EN4_G10_1950_2019.nc'
    settings['fn_I17']     = settings['dir_data'] + 'Steric/I17/I17_1955_2019.nc'
    settings['fn_CZ16']    = settings['dir_data'] + 'Steric/Cheng/Cheng_1940_2019.nc'
    settings['fn_WOA']     = settings['dir_data'] + 'Steric/Levitus/Levitus_2005_2019.nc'
    settings['fn_SIO']     = settings['dir_data'] + 'Steric/SIO/SIO_2004_2019.nc'
    settings['fn_JAMSTEC'] = settings['dir_data'] + 'Steric/JAMSTEC/JAMSTEC_2001_2019.nc'
    settings['fn_BOA']     = settings['dir_data'] + 'Steric/BOA/BOA_2004_2019.nc'
    settings['fn_mask'] = settings['dir_data'] + 'GRACE/JPL_mascon/mask.npy'

    settings['fn_efficiency']       = settings['dir_data'] + 'Budget_GRACE/efficiency/efficiency_combined.npy'

    settings['startyear'] = 2003
    settings['stopyear']  = 2019
    settings['ntime'] = len(np.zeros(12 * (settings['stopyear'] - settings['startyear']+1)))
    year = (np.floor(np.arange(settings['ntime']) / 12) + settings['startyear'])
    settings['time'] = np.around(year + (np.arange(settings['ntime'])/12 + 1/24) - (year - settings['startyear']),4)

    settings['month_running'] = 180
    return settings

def read_oras5(settings):
    # Read ORAS coordinates
    fh = Dataset(settings['fn_oras_grid'],'r')
    fh.set_auto_mask(False)
    lon = fh['nav_lon'][:]
    lon[lon<0]+=360
    lat = fh['nav_lat'][:]
    slm_oras = np.squeeze(fh["tmaskutil"][:])
    area = np.squeeze(fh["e1t"]*fh["e2t"][:])
    fh.close()

    # Interpolate altimetry mask on ORAS grid
    mask = np.load(settings['fn_mask'],allow_pickle=True).all()
    # slm on ORAS grid
    interplnt = interp2d(mask['lon'],mask['lat'],1.0*mask['slm'],kind='linear')
    slm_alt = np.zeros(area.shape)
    for i in range(area.shape[0]):
        for j in range(area.shape[1]):
            slm_alt[i,j] = interplnt(lon[i,j],lat[i,j])
    slm_alt = (np.rint(slm_alt)).astype(np.int)*slm_oras

    oras5 = {}
    oras5['shallow'] = {}
    oras5['full'] = {}

    # Process ORAS full and shallow files
    oras5['shallow'] = proc_ohc_oras(settings['fn_oras_shallow'], area,slm_oras,slm_alt,oras5['shallow'])
    oras5['full']    = proc_ohc_oras(settings['fn_oras_full'],area,slm_oras,slm_alt,oras5['full'])
    return oras5

def proc_ohc_oras(filename,area,slm_oras,slm_alt,oras5):
    print(filename)
    fh = Dataset(filename,'r')
    fh.set_auto_mask(False)
    oras5['time'] = fh['time'][:]
    oras5['glb'] = {}
    oras5['alt'] = {}
    rdata = fh['ohc_2d'][:,:,:]
    oras5['glb']['ohc'] = np.nansum(rdata*slm_oras[np.newaxis,:,:],axis=(1,2))
    oras5['alt']['ohc'] = np.nansum(rdata*slm_alt[np.newaxis,:,:],axis=(1,2))
    rdata = fh['thermosteric_2d'][:,:,:]
    oras5['glb']['steric'] = np.nansum(rdata*(area*slm_oras)[np.newaxis,:,:],axis=(1,2))/(area*slm_oras).sum()
    oras5['alt']['steric'] = np.nansum(rdata*(area*slm_alt)[np.newaxis,:,:],axis=(1,2))/(area*slm_alt).sum()
    fh.close()
    return oras5

def read_ecco(settings):
    slm_alt = np.load(settings['fn_mask'],allow_pickle=True).all()['slm']
    slm_glb = ~np.load(settings['fn_mask'],allow_pickle=True).all()['land']
    ecco = {}
    ecco['shallow'] = {}
    ecco['full'] = {}

    ecco['shallow'] = proc_ohc_ecco(settings['fn_ecco_shallow'], slm_glb,slm_alt,ecco['shallow'])
    ecco['full']    = proc_ohc_ecco(settings['fn_ecco_full'], slm_glb,slm_alt,ecco['full'])
    return ecco

def proc_ohc_ecco(filename,slm_glb,slm_alt,ecco):
    print(filename)
    fh = Dataset(filename,'r')
    fh.set_auto_mask(False)
    area = gentools.grid_area(fh['lat'][:],fh['lon'][:])
    ecco['glb'] = {}
    ecco['alt'] = {}
    rdata = fh['ohc_2d'][:]
    ecco['glb']['ohc'] = np.nansum(rdata*slm_glb[np.newaxis,:,:],axis=(1,2))
    ecco['alt']['ohc'] = np.nansum(rdata*slm_alt[np.newaxis,:,:],axis=(1,2))
    rdata = fh['thermosteric_2d'][:]
    rdata[rdata==-1000] = np.nan
    ecco['glb']['steric'] = np.nansum(rdata*(area*slm_glb)[np.newaxis,:,:],axis=(1,2))/(area*slm_glb).sum()
    ecco['alt']['steric'] = np.nansum(rdata*(area*slm_alt)[np.newaxis,:,:],axis=(1,2))/(area*slm_alt).sum()
    ecco['time'] = fh['time'][:]
    fh.close()
    return ecco

def comp_efficiency(model,settings):
    for dpth in model:
        for msk in ['glb','alt']:
            model[dpth][msk]['eff_rolling'],model[dpth][msk]['eff_tot'] = comp_efficiency_indiv(model[dpth]['time'],model[dpth][msk]['steric'],model[dpth][msk]['ohc'],settings)
    return model

def comp_efficiency_indiv(time,steric,ohc,settings):
    eff_rolling = np.zeros(len(time)-settings['month_running'])
    for i in range(len(eff_rolling)):
        eff_rolling[i] = gentools.lsqtrend(time[i:i+settings['month_running']],steric[i:i+settings['month_running']]) / gentools.lsqtrend(time[i:i+settings['month_running']],ohc[i:i+settings['month_running']])
    eff_tot = gentools.lsqtrend(time,steric) / gentools.lsqtrend(time,ohc)
    return eff_rolling,eff_tot

def compute_obs_eff_indiv(settings):
    ystart = {'EN4_l09': 2003.0, 'EN4_g10': 2003.0, 'I17': 2003.0, 'CZ16': 2003.0, 'CORA': 2003.0, 'WOA': 2005.0, 'SIO': 2005.0, 'JAMSTEC': 2005.0, 'BOA': 2005.0}
    ystop = {'EN4_l09': 2019.99, 'EN4_g10': 2019.99, 'I17': 2019.99, 'CZ16': 2019.99, 'CORA': 2019.0, 'WOA': 2019.99, 'SIO': 2019.99, 'JAMSTEC': 2019.99, 'BOA': 2019.99}
    efficiency_obs = np.zeros((len(settings['steric_products']),2))
    efficiency_ts = {}
    mask = np.load(settings['fn_mask'],allow_pickle=True).all()
    # Read time series
    for idx,prod in enumerate(settings['steric_products']):
        print('  Processing '+prod+'...')
        # Define ystart and ystop for each product
        # Read data
        fname = settings['fn_'+prod]
        file_handle = Dataset(fname,'r')
        file_handle.set_auto_mask(False)
        lat  = file_handle['lat'][:]
        lon  = file_handle['lon'][:]
        time = file_handle['time'][:]
        slm = file_handle['slm'][:]
        area = gentools.grid_area(lat,lon)

        thermosteric_grid = file_handle['thermosteric_2d'][:]
        ohc_grid = file_handle['ohc_2d'][:]
        ohc_grid[ohc_grid == -2e18] = np.nan
        thermosteric_grid[thermosteric_grid < -800] = np.nan
        mask_global_interp    = np.rint(interp2d(mask['lon'], mask['lat'], 1 - mask['land'], kind='linear')(lon, lat)) * slm
        mask_altimetry_interp = np.rint(interp2d(mask['lon'], mask['lat'], mask['slm'], kind='linear')(lon, lat)) * slm
        thermosteric_global = np.nansum((mask_global_interp * area)[np.newaxis, :, :] * thermosteric_grid, axis=(1, 2)) / np.nansum(mask_global_interp * area)
        ohc_global = np.nansum(ohc_grid*mask_global_interp, axis=(1, 2))
        thermosteric_alt = np.nansum((mask_altimetry_interp * area)[np.newaxis, :, :] * thermosteric_grid, axis=(1, 2)) / np.nansum(mask_altimetry_interp * area)
        ohc_alt   = np.nansum(mask_altimetry_interp * ohc_grid, axis=(1, 2))
        eff_ts, _ = comp_efficiency_indiv(time, thermosteric_alt, ohc_alt,settings)
        efficiency_ts[prod] = {}
        efficiency_ts[prod]['time'] = time[:-settings['month_running']]
        efficiency_ts[prod]['eff'] = eff_ts
        acc_prod = (time > ystart[prod]) & (time < ystop[prod]) & (time>2005)
        acc_ts = (settings['time'] > ystart[prod]) & (settings['time'] < ystop[prod]) & (settings['time']>2005)
        efficiency_obs[idx,0] = gentools.lsqtrend(settings['time'][acc_ts],thermosteric_global[acc_prod]) / gentools.lsqtrend(settings['time'][acc_ts],ohc_global[acc_prod])
        efficiency_obs[idx,1] = gentools.lsqtrend(settings['time'][acc_ts],thermosteric_alt[acc_prod]) / gentools.lsqtrend(settings['time'][acc_ts],ohc_alt[acc_prod])
    return(efficiency_obs,efficiency_ts)
