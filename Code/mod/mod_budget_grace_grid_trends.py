# ------------------------------------------------
# Compute mean time series and
# trends on a grid for the following:
# - Steric (not via ensemble, but equal weighting
# - Ocean mass
# - Relative sea level
# - Geocentric sea level
# - GRD bottom deformation
# - GIA bottom deformation
# - Mass + steric
# - Difference
# ------------------------------------------------
import numpy as np
from netCDF4 import Dataset
import mod_gentools as gentools
import mod_budget_grace_mscn as mscn
import multiprocessing as mp
import ctypes as ct

def main():
    print('Computing gridded statistics...')
    global settings
    from mod_budget_grace_settings import settings
    prepare_mask()
    steric_tseries, steric_trends  = compute_steric_trends()
    mass_trends,rsl_trends,gsl_trends,sle_rad_trends,sle_rsl_trends = compute_mass_sl_trends()
    gia_gsl_trends = compute_gia_gsl_trends()
    save_data(steric_trends,mass_trends,rsl_trends,gsl_trends,sle_rad_trends,sle_rsl_trends,gia_gsl_trends,steric_tseries)
    return

def compute_steric_trends():
    global settings, mask
    steric_tseries_indiv = {}
    # Read time series
    for idx,prod in enumerate(settings['steric_products']):
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
        acc_time = (time > ystart[prod]) & (time < ystop[prod])
        total_steric_raw = file_handle['totalsteric_2d'][acc_time,:,:]
        total_steric_raw[total_steric_raw < -800] = np.nan
        file_handle.close()

        acc_time = (settings['time'] > ystart[prod]) & (settings['time'] < ystop[prod])
        mask_steric = np.isfinite(total_steric_raw[0,...])
        total_steric = np.zeros([settings['ntime'],len(lat),len(lon)])*np.nan
        total_steric[acc_time,...] = total_steric_raw
        if prod == 'CORA':
            lonmat, latmat = np.meshgrid(lon, lat)
            grid_dist_lon = 0.5 * np.ones(len(lon))
            grid_dist_lat = np.zeros(len(lat))
            for i in range(len(lat)):
                if i == 0:     grid_dist_lat[i] = (lat[i + 1] - lat[i])
                elif i == 544: grid_dist_lat[i] = lat[i] - lat[i - 1]
                else: grid_dist_lat[i] = (lat[i + 1] - lat[i - 1]) / 2
            lond, latd = np.meshgrid(grid_dist_lon, grid_dist_lat)
            radius = 6371000
            area = (np.deg2rad(lonmat + lond / 2) - np.deg2rad(lonmat - lond / 2)) * (np.sin(np.deg2rad(latmat + latd / 2)) - np.sin(np.deg2rad(latmat - latd / 2))) * radius ** 2
            total_steric = np.dstack([total_steric[:, :, 360:], total_steric[:, :, :360]])
            mask_steric = np.hstack([mask_steric[:, 360:], mask_steric[:, :360]])
            area = np.hstack([area[:, 360:], area[:, :360]])
            lon = np.arange(0,360,0.5)
        else: area = gentools.grid_area(lat,lon)
        total_steric_mscn = mscn.masconize_regrid_3d(lat,lon,total_steric,area,mask_steric,~mask['land'],settings)
        total_steric_mscn[~acc_time,...] = np.nan
        steric_tseries_indiv[prod] = total_steric_mscn

    # Compute time series
    steric_tseries = np.zeros([settings['ntime'],360,720])
    steric_ndata   = np.zeros([settings['ntime'],360,720])
    for idx,prod in enumerate(settings['steric_products']):
        acc_idx = np.isfinite(steric_tseries_indiv[prod])
        steric_tseries[acc_idx] += steric_tseries_indiv[prod][acc_idx]
        steric_ndata[acc_idx] += 1
    steric_ndata[steric_ndata==0] = np.nan
    steric_tseries/=steric_ndata

    # Remove baseline
    steric_tseries -= steric_tseries[-12:,:,:].mean(axis=0)

    # Compute trends
    steric_trends = {}
    for era in range(len(settings['trend_eras'])):
        tname = str(settings['trend_eras'][era][0])+'-'+str(settings['trend_eras'][era][1])
        print(tname)
        steric_trends[tname] = np.zeros(mask['land'].shape)
        nprod = np.zeros(mask['land'].shape)*np.nan
        nprod[~mask['land']] = 0
        ystart = settings['trend_eras'][era][0]
        ystop  = settings['trend_eras'][era][-1]
        time_acc = (settings['time'] > ystart) & (settings['time'] < ystop)
        for prod in settings['steric_products']:
            if np.isnan(steric_tseries_indiv[prod][time_acc,100,100]).sum() == 0: # Accept model
                trend_indiv = gentools.field_seas_trend(settings['time'][time_acc],steric_tseries_indiv[prod][time_acc,...])
                nprod[np.isfinite(trend_indiv)]+=1
                steric_trends[tname][np.isfinite(trend_indiv)]+=trend_indiv[np.isfinite(trend_indiv)]
        nprod[nprod==0] = np.nan
        steric_trends[tname]/=nprod
    return(steric_tseries, steric_trends)

def compute_mass_sl_trends():
    global mask, settings, mass_tseries, rsl_tseries, gsl_tseries, sle_rad_tseries, sle_rsl_tseries, probability
    # 1. Fill time series array
    mass_tseries    = mp_empty_float([settings['ntime'],len(settings['lat']),len(settings['lon'])])
    rsl_tseries     = mp_empty_float([settings['ntime'],len(settings['lat']),len(settings['lon'])])
    gsl_tseries     = mp_empty_float([settings['ntime'],len(settings['lat']),len(settings['lon'])])
    sle_rad_tseries = mp_empty_float([settings['ntime'],len(settings['lat']),len(settings['lon'])])
    sle_rsl_tseries = mp_empty_float([settings['ntime'],len(settings['lat']),len(settings['lon'])])

    probability = mp_filled_float(read_probability())
    pool = mp.Pool(settings['nproc'])
    out  = pool.map(read_ocean_ts, settings['ens_range'])
    # Set land and unacceptable times to nan
    mass_tseries[:,mask['land']] = np.nan
    rsl_tseries[:,~mask['slm']] = np.nan
    gsl_tseries[:,~mask['slm']] = np.nan
    sle_rad_tseries[:,mask['land']] = np.nan
    sle_rsl_tseries[:,mask['land']] = np.nan

    mass_tseries[~settings['time_mask_grace'],...] = np.nan
    rsl_tseries[~settings['time_mask_grace'],...] = np.nan
    gsl_tseries[~settings['time_mask_grace'],...] = np.nan
    sle_rad_tseries[~settings['time_mask_grace'],...] = np.nan
    sle_rsl_tseries[~settings['time_mask_grace'],...] = np.nan

    # Remove baseline
    mass_tseries    -= mass_tseries[-12:,:,:].mean(axis=0)
    rsl_tseries     -= rsl_tseries[-12:,:,:].mean(axis=0)
    gsl_tseries     -= gsl_tseries[-12:,:,:].mean(axis=0)
    sle_rad_tseries -= sle_rad_tseries[-12:,:,:].mean(axis=0)
    sle_rsl_tseries -= sle_rsl_tseries[-12:,:,:].mean(axis=0)

    # Trends
    mass_trends    = {}
    rsl_trends     = {}
    gsl_trends     = {}
    sle_rad_trends = {}
    sle_rsl_trends = {}
    for era in range(len(settings['trend_eras'])):
        tname = str(settings['trend_eras'][era][0])+'-'+str(settings['trend_eras'][era][1])
        print(tname)
        mass_trends[tname]    = np.zeros(mask['land'].shape)
        rsl_trends[tname]     = np.zeros(mask['land'].shape)
        gsl_trends[tname]     = np.zeros(mask['land'].shape)
        sle_rad_trends[tname] = np.zeros(mask['land'].shape)
        sle_rsl_trends[tname] = np.zeros(mask['land'].shape)

        ystart = settings['trend_eras'][era][0]
        ystop  = settings['trend_eras'][era][-1]
        time_acc = (settings['time'] > ystart) & (settings['time'] < ystop) & (np.isfinite(mass_tseries[:,100,100]))
        mass_trends[tname]    = mscn.masconize_2d(gentools.field_seas_trend(settings['time'][time_acc],mass_tseries[time_acc,...]),mask,settings)
        rsl_trends[tname]     = mscn.masconize_2d(gentools.field_seas_trend(settings['time'][time_acc],rsl_tseries[time_acc,...]),mask,settings)
        gsl_trends[tname]     = mscn.masconize_2d(gentools.field_seas_trend(settings['time'][time_acc],gsl_tseries[time_acc,...]),mask,settings)
        sle_rad_trends[tname] = mscn.masconize_2d(gentools.field_seas_trend(settings['time'][time_acc],sle_rad_tseries[time_acc,...]),mask,settings)
        sle_rsl_trends[tname] = mscn.masconize_2d(gentools.field_seas_trend(settings['time'][time_acc],sle_rsl_tseries[time_acc,...]),mask,settings)

    return(mass_trends,rsl_trends,gsl_trends,sle_rad_trends,sle_rsl_trends)

def read_ocean_ts(ens):
    print(ens)
    global settings, settings, mass_tseries, rsl_tseries, gsl_tseries, sle_rad_tseries, sle_rsl_tseries, probability
    fname = settings['dir_save_ens'] + 'mass_rsl_ens_' + str(ens).zfill(4) + '.nc'
    file_handle = Dataset(fname, 'r')
    file_handle.set_auto_mask(False)
    mass_tseries+= probability[ens]*file_handle.variables['ewh'][:]
    rsl_tseries+= probability[ens]*file_handle.variables['obs_rsl_grid'][:]
    gsl_tseries+= probability[ens]*file_handle.variables['obs_gsl_grid'][:]
    sle_rad_tseries += probability[ens]*file_handle.variables['sle_rad_grid'][:]
    sle_rsl_tseries += probability[ens]*file_handle.variables['sle_rsl_grid'][:]
    file_handle.close()
    return

def compute_gia_gsl_trends():
    global mask, settings
    probability = read_probability()
    file_handle = Dataset(settings['fn_gia_ens_rad'],'r')
    file_handle.set_auto_mask(False)
    gia_rad   = file_handle.variables['rad'][:settings['num_ens'],...]
    file_handle.close()
    file_handle = Dataset(settings['fn_gia_ens_rsl'],'r')
    file_handle.set_auto_mask(False)
    gia_rsl   = file_handle.variables['rsl'][:settings['num_ens'],...]
    file_handle.close()
    gia_gsl = (gia_rsl + gia_rad)
    gia_gsl = (probability[:,np.newaxis,np.newaxis]*gia_gsl).sum(axis=0)
    gia_gsl_mscn = mscn.masconize_2d(gia_gsl, mask, settings)
    return(gia_gsl_mscn)

def save_data(steric_trends,mass_trends,rsl_trends,gsl_trends,sle_rad_trends,sle_rsl_trends,gia_gsl_trends,steric_tseries):
    global mask, settings, mass_tseries, rsl_tseries, gsl_tseries, sle_rad_tseries, sle_rsl_tseries, probability
    # Save data
    file_handle = Dataset(settings['fn_grid_trends'],'w')
    file_handle.createDimension('lon', len(settings['lon']))
    file_handle.createDimension('lat', len(settings['lat']))
    file_handle.createDimension('time', len(settings['time']))

    file_handle.createVariable('lon', 'f4', ('lon',), zlib=True)[:] = settings['lon']
    file_handle.createVariable('lat', 'f4', ('lat',), zlib=True)[:] = settings['lat']
    file_handle.createVariable('time', 'f4', ('time',), zlib=True)[:] = settings['time']

    # Time series
    file_handle.createVariable('mass_tseries', 'i2', ('time','lat', 'lon',), zlib=True, complevel=6)[:] = 10 * mass_tseries
    file_handle.createVariable('steric_tseries', 'i2', ('time','lat', 'lon',), zlib=True, complevel=6)[:] = 10 * steric_tseries
    file_handle.createVariable('rsl_tseries', 'i2', ('time','lat', 'lon',), zlib=True, complevel=6)[:] = 10 * rsl_tseries
    file_handle.createVariable('gsl_tseries', 'i2', ('time','lat', 'lon',), zlib=True, complevel=6)[:] = 10 * gsl_tseries
    file_handle.createVariable('sle_rad_tseries', 'i2', ('time','lat', 'lon',), zlib=True, complevel=6)[:] = 10 * sle_rad_tseries
    file_handle.createVariable('sle_rsl_tseries', 'i2', ('time','lat', 'lon',), zlib=True, complevel=6)[:] = 10 * sle_rsl_tseries
    file_handle.variables['mass_tseries'].setncattr('scale_factor', 0.1)
    file_handle.variables['steric_tseries'].setncattr('scale_factor', 0.1)
    file_handle.variables['rsl_tseries'].setncattr('scale_factor', 0.1)
    file_handle.variables['gsl_tseries'].setncattr('scale_factor', 0.1)
    file_handle.variables['sle_rad_tseries'].setncattr('scale_factor', 0.1)
    file_handle.variables['sle_rsl_tseries'].setncattr('scale_factor', 0.1)

    # Trends
    for era in range(len(settings['trend_eras'])):
        tname = str(settings['trend_eras'][era][0])+'-'+str(settings['trend_eras'][era][1])
        steric_name    = 'steric_'+str(settings['trend_eras'][era][0])+'_'+str(settings['trend_eras'][era][1])
        mass_name    = 'mass_'+str(settings['trend_eras'][era][0])+'_'+str(settings['trend_eras'][era][1])
        rsl_name     = 'rsl_'+str(settings['trend_eras'][era][0])+'_'+str(settings['trend_eras'][era][1])
        gsl_name     = 'gsl_'+str(settings['trend_eras'][era][0])+'_'+str(settings['trend_eras'][era][1])
        sle_rad_name = 'sle_rad_'+str(settings['trend_eras'][era][0])+'_'+str(settings['trend_eras'][era][1])
        sle_rsl_name = 'sle_rsl_'+str(settings['trend_eras'][era][0])+'_'+str(settings['trend_eras'][era][1])
        budget_name = 'budget_'+str(settings['trend_eras'][era][0])+'_'+str(settings['trend_eras'][era][1])
        rsl_min_mass_name = 'rsl_min_mass_'+str(settings['trend_eras'][era][0])+'_'+str(settings['trend_eras'][era][1])

        diff_name = 'diff_'+str(settings['trend_eras'][era][0])+'_'+str(settings['trend_eras'][era][1])
        file_handle.createVariable(mass_name, 'i2', ('lat', 'lon',), zlib=True, complevel=6)[:] = 100 * mass_trends[tname]
        file_handle.createVariable(steric_name, 'i2', ('lat', 'lon',), zlib=True, complevel=6)[:] = 100 * steric_trends[tname]
        file_handle.createVariable(rsl_name, 'i2', ('lat', 'lon',), zlib=True, complevel=6)[:] = 100 * rsl_trends[tname]
        file_handle.createVariable(gsl_name, 'i2', ('lat', 'lon',), zlib=True, complevel=6)[:] = 100 * gsl_trends[tname]
        file_handle.createVariable(sle_rad_name, 'i2', ('lat', 'lon',), zlib=True, complevel=6)[:] = 100 * sle_rad_trends[tname]
        file_handle.createVariable(sle_rsl_name, 'i2', ('lat', 'lon',), zlib=True, complevel=6)[:] = 100 * sle_rsl_trends[tname]

        file_handle.createVariable(rsl_min_mass_name, 'i2', ('lat', 'lon',), zlib=True, complevel=6)[:] = 100 * (rsl_trends[tname] - mass_trends[tname])
        file_handle.createVariable(budget_name, 'i2', ('lat', 'lon',), zlib=True, complevel=6)[:] = 100 * (steric_trends[tname] + mass_trends[tname])
        file_handle.createVariable(diff_name, 'i2', ('lat', 'lon',), zlib=True, complevel=6)[:] = 100 * (rsl_trends[tname] - steric_trends[tname] - mass_trends[tname])
        file_handle.variables[mass_name].setncattr('scale_factor', 0.01)
        file_handle.variables[steric_name].setncattr('scale_factor', 0.01)
        file_handle.variables[rsl_name].setncattr('scale_factor', 0.01)
        file_handle.variables[gsl_name].setncattr('scale_factor', 0.01)
        file_handle.variables[sle_rad_name].setncattr('scale_factor', 0.01)
        file_handle.variables[sle_rsl_name].setncattr('scale_factor', 0.01)
        file_handle.variables[rsl_min_mass_name].setncattr('scale_factor', 0.01)
        file_handle.variables[budget_name].setncattr('scale_factor', 0.01)
        file_handle.variables[diff_name].setncattr('scale_factor', 0.01)



    file_handle.createVariable('gsl_gia', 'i2', ('lat', 'lon',), zlib=True, complevel=6)[:] = 100 * gia_gsl_trends
    file_handle.variables['gsl_gia'].setncattr('scale_factor', 0.01)
    file_handle.close()
    return

def prepare_mask():
    # Read masks and glacier data for separation of land mass
    # changes into individual components
    print('  Reading mask and Zemp data...')
    global settings, mask
    mask = np.load(settings['fn_mask'],allow_pickle=True).all()
    # Ocean basin area
    mask['area'] = mp_filled_float(gentools.grid_area(settings['lat'],settings['lon']))
    return

def read_probability():
    global settings
    probability = Dataset(settings['fn_gia_ens_rad'],'r').variables['probability'][settings['ens_range']]._get_data()
    probability /= probability.sum()
    return(probability)

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