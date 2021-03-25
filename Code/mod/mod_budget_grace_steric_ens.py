# ------------------------------------
# Create ensembles of steric data sets
# - All obs
# 1. EN4 g10 2003-2019
# 2. EN4 l09 2003-2019
# 3. I17 2003-2019
# 4. CZ16 2003-2019
# 6. WOA 2005-2019
# - Argo only
# 1. SIO 2005-2019
# 2. BOA 2005-2019
# 4. JAMSTEC 2005-2019
# Compute ensemble of global-mean and
# basin-mean steric sea level
# -----------------------------
# Compute OHC from same ensemble
# -----------------------------------
import numpy as np
from netCDF4 import Dataset
import mod_gentools as gentools
from scipy.interpolate import interp2d
def main():
    print('Computing steric ensemble members...')
    from mod_budget_grace_settings import settings
    global settings,dt_start
    prepare_mask()

    steric = np.zeros(len(settings['steric_products']),dtype='object')
    OHC    = np.zeros(len(settings['steric_products']),dtype='object')

    for idx,prod in enumerate(settings['steric_products']):
        steric[idx] = read_steric(prod)
        OHC[idx]    = read_OHC(prod)
    # Generate ensembles
    rand_steric = np.random.randint(0,len(settings['steric_products']),settings['num_ens'])

    steric_ensemble = {}
    steric_ensemble['global'] = np.zeros([settings['num_ens'],settings['ntime']])*np.nan
    steric_ensemble['altimetry'] = np.zeros([settings['num_ens'],settings['ntime']])*np.nan
    steric_ensemble['basin'] = np.zeros([settings['num_ens'],len(settings['region_names']),settings['ntime']])*np.nan
    for ens in range(settings['num_ens']):
        steric_ensemble['global'][ens,:] = steric[rand_steric[ens]]['global']
        steric_ensemble['altimetry'][ens,:] = steric[rand_steric[ens]]['altimetry']
        steric_ensemble['basin'][ens,:,:] = steric[rand_steric[ens]]['basin']

    OHC_ensemble = {}
    OHC_ensemble['global'] = np.zeros([settings['num_ens'],settings['ntime']])*np.nan
    OHC_ensemble['altimetry'] = np.zeros([settings['num_ens'],settings['ntime']])*np.nan

    for ens in range(settings['num_ens']):
        OHC_ensemble['global'][ens,:] = OHC[rand_steric[ens]]['global']
        OHC_ensemble['altimetry'][ens,:] = OHC[rand_steric[ens]]['altimetry']


    np.save(settings['fn_steric_ensemble'],steric_ensemble) # Save data
    np.save(settings['fn_OHC_ensemble'],OHC_ensemble) # Save data
    return

def prepare_mask():
    print('  Reading mask data...')
    global settings, mask
    mask = {}
    mask = np.load(settings['fn_mask'],allow_pickle=True).all()
    # Ocean basin area
    mask['area'] = gentools.grid_area(settings['lat'],settings['lon'])
    mask['ocean_basin_area'] = np.zeros(6)
    for basin in range(6): mask['ocean_basin_area'][basin] = (mask['area']*(mask['basin'] == basin)).sum()
    return

def read_OHC(prod):
    print('  Processing OHC '+prod+'...')
    global settings, mask
    OHC ={}
    OHC['product'] = prod
    # Define ystart and ystop for each product
    ystart = {'EN4_l09':2003.0,'EN4_g10':2003.0, 'I17':2003.0, 'CZ16':2003.0, 'CORA':2003.0, 'WOA':2005.0, 'SIO':2005.0, 'JAMSTEC':2005.0, 'BOA':2005.0}
    ystop  = {'EN4_l09':2019.99,'EN4_g10':2019.99, 'I17':2019.99, 'CZ16':2019.99, 'CORA':2019.0, 'WOA':2019.99, 'SIO':2019.99, 'JAMSTEC':2019.99, 'BOA':2019.99}

    fname = settings['fn_'+prod]
    file_handle = Dataset(fname,'r')
    file_handle.set_auto_mask(False)
    time = file_handle['time'][:]
    acc_time = (time > ystart[prod]) & (time < ystop[prod])
    time = time[acc_time]
    OHC_glb  = file_handle['ohc_ts'][acc_time]
    OHC_grid = file_handle['ohc_2d'][acc_time, :, :]
    lat  = file_handle['lat'][:]
    lon  = file_handle['lon'][:]
    file_handle.close()

    OHC_grid[OHC_grid == -2e18] = np.nan
    mask_steric = np.isfinite(OHC_grid[0,...])

    mask_altimetry_interp = np.rint(interp2d(settings['lon'], settings['lat'], mask['slm'], kind='linear')(lon,lat)) * mask_steric
    ohc_alt = np.nansum(mask_altimetry_interp * OHC_grid, axis=(1, 2))

    # Interpolate on time grid
    acc_time = (settings['time'] > ystart[prod]) & (settings['time'] < ystop[prod])
    OHC['global'] = np.zeros(len(settings['time']))*np.nan
    OHC['global'][acc_time] = OHC_glb
    OHC['altimetry'] = np.zeros(len(settings['time']))*np.nan
    OHC['altimetry'][acc_time] = ohc_alt
    return(OHC)


def read_steric(prod):
    print('  Processing steric '+prod+'...')
    global settings, mask
    steric ={}
    steric['product'] = prod
    # Define ystart and ystop for each product
    ystart = {'EN4_l09':2003.0,'EN4_g10':2003.0, 'I17':2003.0, 'CZ16':2003.0, 'CORA':2003.0, 'WOA':2005.0, 'SIO':2005.0, 'JAMSTEC':2005.0, 'BOA':2005.0}
    ystop  = {'EN4_l09':2020.99,'EN4_g10':2020.99, 'I17':2019.99, 'CZ16':2020.99, 'CORA':2019.0, 'WOA':2019.99, 'SIO':2020.99, 'JAMSTEC':2020.99, 'BOA':2019.99}

    fname = settings['fn_'+prod]
    file_handle = Dataset(fname,'r')
    file_handle.set_auto_mask(False)
    lat  = file_handle['lat'][:]
    lon  = file_handle['lon'][:]
    time = file_handle['time'][:]
    acc_time = (time > ystart[prod]) & (time < ystop[prod]) & ((time>settings['startyear']) & (time<settings['stopyear']+1))
    time = time[acc_time]

    # Fields
    glb_halo = file_handle['halosteric_ts'][acc_time]
    total_steric = file_handle['totalsteric_2d'][acc_time,:,:]
    total_steric[total_steric<-800] = np.nan
    file_handle.close()

    # Interpolate on time grid
    acc_time = (settings['time'] > ystart[prod]) & (settings['time'] < ystop[prod])
    steric['global_halosteric'] = np.zeros(len(settings['time']))*np.nan
    steric['global_halosteric'][acc_time] = glb_halo

    # Determine global and basin-mean fields
    mask_steric = np.isfinite(total_steric[0,...])
    area = gentools.grid_area(lat,lon)

    # Interpolate global grids on steric product grid
    mask_global_interp    = np.rint(interp2d(settings['lon'], settings['lat'], 1-mask['land'], kind='linear')(lon,lat)) * mask_steric
    mask_altimetry_interp = np.rint(interp2d(settings['lon'], settings['lat'], mask['slm'], kind='linear')(lon,lat)) * mask_steric
    steric_global = np.nansum((mask_global_interp*area)[np.newaxis, :, :]    * total_steric,axis=(1,2))/np.nansum(mask_global_interp*area)    # Global steric
    steric_alt    = np.nansum((mask_altimetry_interp*area)[np.newaxis, :, :] * total_steric,axis=(1,2))/np.nansum(mask_altimetry_interp*area) # Altimetry steric

    steric['global'] = np.zeros(len(settings['time'])) * np.nan
    steric['altimetry'] = np.zeros(len(settings['time'])) * np.nan
    steric['global'][acc_time] = steric_global
    steric['altimetry'][acc_time] = steric_alt

    # Compute basin averages
    steric['basin'] = np.zeros([len(settings['region_names']),len(settings['time'])])*np.nan
    for idx in range(len(settings['region_names'])):
        mask_lcl = (mask['basin']==idx)*1.0 # Select mask
        mask_interp = np.rint(interp2d(settings['lon'],settings['lat'],mask_lcl,kind='linear')(lon,lat))*mask_steric # Interpolate masks on grid of steric product
        steric_msk  = np.nansum((mask_interp*area)[np.newaxis,:,:]*total_steric, axis=(1, 2))/np.nansum(mask_interp * area)
        steric['basin'][idx,acc_time] = steric_msk
    return(steric)

def remove_seasonal(time, tseries):
    amat = np.ones([len(time), 6])
    amat[:, 0] = np.sin(2 * np.pi * time)
    amat[:, 1] = np.cos(2 * np.pi * time)
    amat[:, 2] = np.sin(4 * np.pi * time)
    amat[:, 3] = np.cos(4 * np.pi * time)
    amat[:, -1] = time - np.mean(time)
    sol = np.linalg.lstsq(amat, tseries, rcond=None)[0]
    sol[-1] = 0
    tseries_noseas = tseries - np.matmul(amat, sol)
    tseries_noseas = tseries_noseas - np.mean(tseries_noseas[:12])
    return (tseries_noseas)
