# -------------------------------------
# Read individual steric estimates and
# compute expansion efficieny
# Save global and altimetry time series
# -------------------------------------
import numpy as np
from netCDF4 import Dataset
import mod_gentools as gentools
from scipy.interpolate import interp2d

def main():
    print('Computing gridded statistics...')
    global settings
    from mod_budget_grace_settings import settings
    prepare_mask()
    steric = compute_steric_indiv()
    np.save(settings['fn_steric_indiv'],steric)

    # Statistics of efficiencies
    eff_indiv = np.zeros(len(steric))
    for idx,prod in enumerate(steric):
        eff_indiv[idx] = steric[prod]['altimetry']['efficiency']
    eff_indiv*=(3600*24*365.25)*(4*np.pi*6371000**2)
    eff_indiv.mean()
    1.65*eff_indiv.std()
    return

def compute_steric_indiv():
    global settings, mask
    steric = {}
    # Read time series
    for idx,prod in enumerate(settings['steric_products']):
        steric_indiv = {}
        print('  Processing '+prod+'...')
        # Define ystart and ystop for each product
        ystart = {'EN4_l09': 2003.0, 'EN4_g10': 2003.0, 'I17': 2003.0, 'CZ16': 2003.0, 'CORA': 2003.0, 'WOA': 2005.0, 'SIO': 2005.0, 'JAMSTEC': 2005.0, 'BOA': 2005.0}
        ystop = {'EN4_l09': 2019.99, 'EN4_g10': 2019.99, 'I17': 2019.99, 'CZ16': 2019.99, 'CORA': 2019.0, 'WOA': 2019.99, 'SIO': 2019.99, 'JAMSTEC': 2019.99, 'BOA': 2019.99}
        # Read data
        fname = settings['fn_'+prod]
        file_handle = Dataset(fname,'r')
        file_handle.set_auto_mask(False)
        lat  = file_handle['lat'][:]
        lon  = file_handle['lon'][:]
        time = file_handle['time'][:]
        slm = file_handle['slm'][:]

        acc_prod = (time > ystart[prod]) & (time < ystop[prod])
        acc_ts = (settings['time'] > ystart[prod]) & (settings['time'] < ystop[prod])

        halosteric_grid = np.zeros([settings['ntime'],len(lat),len(lon)])*np.nan
        thermosteric_grid = np.zeros([settings['ntime'],len(lat),len(lon)])*np.nan
        ohc_grid = np.zeros([settings['ntime'],len(lat),len(lon)])*np.nan

        halosteric_grid[acc_ts,...] = file_handle['halosteric_2d'][acc_prod,:,:]
        thermosteric_grid[acc_ts,...] = file_handle['thermosteric_2d'][acc_prod,:,:]
        ohc_grid[acc_ts,...] = file_handle['ohc_2d'][acc_prod,:,:]

        ohc_grid[ohc_grid == -2e18] = np.nan
        thermosteric_grid[thermosteric_grid < -800] = np.nan
        halosteric_grid[halosteric_grid < -800] = np.nan

        area = gentools.grid_area(lat,lon)

        mask_global_interp    = np.rint(interp2d(settings['lon'], settings['lat'], 1 - mask['land'], kind='linear')(lon, lat)) * slm
        mask_altimetry_interp = np.rint(interp2d(settings['lon'], settings['lat'], mask['slm'], kind='linear')(lon, lat)) * slm
        thermosteric_global = np.nansum((mask_global_interp * area)[np.newaxis, :, :] * thermosteric_grid, axis=(1, 2)) / np.nansum(mask_global_interp * area)
        halosteric_global = np.nansum((mask_global_interp * area)[np.newaxis, :, :] * halosteric_grid, axis=(1, 2)) / np.nansum(mask_global_interp * area)
        ohc_global = np.nansum(ohc_grid*mask_global_interp, axis=(1, 2))

        thermosteric_alt = np.nansum((mask_altimetry_interp * area)[np.newaxis, :, :] * thermosteric_grid, axis=(1, 2)) / np.nansum(mask_altimetry_interp * area)
        halosteric_alt   = np.nansum((mask_altimetry_interp * area)[np.newaxis, :, :] * halosteric_grid, axis=(1, 2)) / np.nansum(mask_altimetry_interp * area)
        ohc_alt   = np.nansum(mask_altimetry_interp * ohc_grid, axis=(1, 2))

        eff_glb = gentools.lsqtrend(settings['time'][acc_ts],thermosteric_global[acc_ts]) / gentools.lsqtrend(settings['time'][acc_ts],ohc_global[acc_ts])
        eff_alt = gentools.lsqtrend(settings['time'][acc_ts],thermosteric_alt[acc_ts]) / gentools.lsqtrend(settings['time'][acc_ts],ohc_alt[acc_ts])

        steric_indiv['global'] = {}
        steric_indiv['altimetry'] = {}
        for i in steric_indiv:
            steric_indiv[i]['thermosteric'] = {}
            steric_indiv[i]['halosteric'] = {}
            steric_indiv[i]['ohc'] = {}
            for j in steric_indiv[i]:
                steric_indiv[i][j]['tseries'] =  np.zeros(len(settings['time'])) * np.nan
                steric_indiv[i][j]['tseries_seas'] =  np.zeros(len(settings['time'])) * np.nan

        steric_indiv['global']['thermosteric']['tseries'][acc_ts] = thermosteric_global[acc_ts]
        steric_indiv['global']['halosteric']['tseries'][acc_ts] =  halosteric_global[acc_ts]
        steric_indiv['global']['ohc']['tseries'][acc_ts] =  ohc_global[acc_ts]
        steric_indiv['global']['efficiency'] = eff_glb

        steric_indiv['altimetry']['thermosteric']['tseries'][acc_ts] =  thermosteric_alt[acc_ts]
        steric_indiv['altimetry']['halosteric']['tseries'][acc_ts] =  halosteric_alt[acc_ts]
        steric_indiv['altimetry']['ohc']['tseries'][acc_ts] =  ohc_alt[acc_ts]
        steric_indiv['altimetry']['efficiency'] = eff_alt
        for i in steric_indiv:
            for j in ['thermosteric','halosteric','ohc']:
                steric_indiv[i][j]['trend'] = trend_stats(steric_indiv[i][j]['tseries'])
        steric[prod] = steric_indiv
    return(steric)

def trend_stats(tseries):
    global settings
    # Compute ensemble trend statistics
    trend = {}
    for era in range(len(settings['trend_eras'])):
        tname = str(settings['trend_eras'][era][0])+'-'+str(settings['trend_eras'][era][1])
        trend[tname] = np.zeros(3)
        # Determine which times and ensemble members to accept
        time_acc = ((settings['time'] > settings['trend_eras'][era][0]) & (settings['time'] < settings['trend_eras'][era][1]+1) & np.isfinite(tseries))
        amat = np.ones([time_acc.sum(),2])
        amat[:,1] = settings['time'][time_acc] - settings['time'][time_acc].mean()
        trend[tname] = np.linalg.lstsq(amat,tseries[time_acc],rcond=None)[0][1]
    return(trend)

def prepare_mask():
    # Read masks and glacier data for separation of land mass
    # changes into individual components
    print('  Reading mask and Zemp data...')
    global settings, mask
    mask = np.load(settings['fn_mask'],allow_pickle=True).all()
    # Ocean basin area
    mask['area'] = gentools.grid_area(settings['lat'],settings['lon'])
    return
