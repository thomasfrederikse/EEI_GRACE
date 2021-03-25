# -----------------------------------------------------------------
# Compute ocean mass, land mass, and relative sea level in one pass
# 1. Prepare all data for parallel processing
# 2. Compute all quantities per GIA model
# 3. Save all quantities in netcdf file
# -------------------------------------
# Save the following quantities
# a. Ocean mass change from GRACE (global, basin, grid)
# b. Equivalent land mass change (total and per source)
# c. RSL and RAD resulting due to GRD from this ocean mass change
# d. Relative sea level from altimetry, GIA and GRD
# -----------------------------------------------------------------
import numpy as np
from netCDF4 import Dataset
from   scipy.interpolate import interp1d
import multiprocessing as mp
import ctypes as ct
import os
import mod_gentools as gentools
import datetime as dt
import pySLE3
import mod_budget_grace_mscn as mscn

def main():
    print('Computing ocean mass and altimetry ensemble members...')
    from mod_budget_grace_settings import settings
    global settings,dt_start
    # Prepare data
    prepare_mask()
    prepare_grace()
    prepare_altimetry()
    dt_start = dt.datetime.now().replace(microsecond=0)
    print('  Running over all ensembles...')
    pool = mp.Pool(settings['nproc'])
    out  = pool.map(func=process_ensemble_member, iterable=settings['ens_range'])
    return

# --------------------
# Preparation routines
# --------------------
def prepare_grace():
    # Read GRACE data and interpolate on the common monthly grid
    print('  Reading GRACE data...')
    global settings, grace
    grace = {}
    # Read GRACE data
    file_handle = Dataset(settings['fn_grace'],'r')
    file_handle.set_auto_mask(False)
    grace['lon']   = mp_filled_float(file_handle.variables['lon'][:])
    grace['lat']   = mp_filled_float(file_handle.variables['lat'][:])
    time_original  = file_handle.variables['time'][:]
    ewh_original   = file_handle.variables['ewh_noseas'][:]
    ewh_ste_original = file_handle.variables['ewh_ste'][:]
    file_handle.close()
    print('    Interpolating on common time grid...')
    grace['ewh']     = mp_filled_float(interp1d(time_original, ewh_original, axis=0, kind='linear',fill_value='extrapolate')(settings['time']))
    grace['ewh_ste'] = mp_filled_float(interp1d(time_original, ewh_ste_original, axis=0, kind='linear',fill_value='extrapolate')(settings['time']))
    grace['ewh'] -= grace['ewh'].mean(axis=0)[np.newaxis, :, :]
    # Mask out GRACE/GRACE-FO gap
    grace['ewh'][~settings['time_mask_grace'], :, :] = np.nan
    grace['ewh_ste'][~settings['time_mask_grace'], :, :] = np.nan
    print('    Mascon representation of standard errors...')
    # Compute mascon representation of errors
    grace['ewh_ste_mscn'] = mp_filled_float(mscn.grid2mascon_3d(grace['lat'], grace['lon'], grace['ewh_ste'], settings))
    return

def prepare_altimetry():
    # Read altimetry and store in global variable
    print('  Reading altimetry data...')
    global altimetry, mask, settings
    altimetry = {}
    file_handle = Dataset(settings['fn_altimetry'],'r')
    file_handle.set_auto_mask(False)
    altimetry['lat']   = mp_filled_float(file_handle.variables['lat'][:])
    altimetry['lon']   = mp_filled_float(file_handle.variables['lon'][:])
    altimetry['ssh']   = mp_filled_float(file_handle.variables['ssh'][:])
    altimetry['ssh'][altimetry['ssh']>1599] = np.nan
    altimetry['ssh'][:,~mask['slm']] = np.nan
    file_handle.close()
    return

def prepare_mask():
    # Read masks and glacier data for separation of land mass
    # changes into individual components
    print('  Reading mask and Zemp data...')
    global settings, mask
    mask = np.load(settings['fn_mask'],allow_pickle=True).all()
    # Read glaciers
    zemp_time = np.arange(2002,2017)
    zemp_rate_mean  = np.zeros([len(mask['glacier_num_insitu']),15],dtype=np.float32) # 2002 - 2016
    zemp_rate_sterr = np.zeros([len(mask['glacier_num_insitu']),15],dtype=np.float32)
    zemp_flist = os.listdir(settings['dir_glacier_zemp'])
    for reg,num in enumerate(mask['glacier_num_insitu']):
        fname = settings['dir_glacier_zemp']+[i for i in zemp_flist if '_'+str(num)+'_' in i][0]
        raw_data = np.loadtxt(fname,skiprows=28,delimiter=',',usecols=(0,10,18))
        years      = raw_data[:,0]
        acc_idx = np.in1d(years,zemp_time)
        zemp_rate_mean[reg,:]  = raw_data[acc_idx,1] # Rate in gigatons
        zemp_rate_sterr[reg,:] = raw_data[acc_idx,2] # Uncertainty in gigatons
    mask['zemp_time']       = mp_filled_float(zemp_time)
    mask['zemp_rate_mean']  = mp_filled_float(zemp_rate_mean)
    mask['zemp_rate_sterr'] = mp_filled_float(zemp_rate_sterr)

    # Ocean basin area
    mask['area'] = mp_filled_float(gentools.grid_area(settings['lat'],settings['lon']))
    mask['ocean_basin_area'] = np.zeros(6)
    for basin in range(6): mask['ocean_basin_area'][basin] = (mask['area']*(mask['basin'] == basin)).sum()
    return

# -----------------------------------
# Ensemble member processing routines
# -----------------------------------
def process_ensemble_member(ens):
    global settings,dt_start
    dt_now = dt.datetime.now().replace(microsecond=0)
    print('   Ensemble '+str(ens+1)+'/'+str(settings['num_ens'])+' Elapsed time:',(dt_now - dt_start))
    gia = read_gia_ensemble_member(ens)                             # 1. Read GIA ewh/gsl
    ewh_ptb  = perturbed_ewh_estimate(gia, ens)                     # 2. Generate perturbed land mass/obp estimate
    mass_ctb = separate_mass_ctb(ewh_ptb, ens)                      # 3. Separate land mass estimate into components
    sle_result = solve_sle(ewh_ptb, ens)                            # 4. Solve sea-level equation
    sl_altimetry = compute_sl_altimetry(gia, sle_result['rad'], ens)# 5. Compute relative sea level on grid
    save_ens_nc(ens, ewh_ptb, mass_ctb, sle_result, sl_altimetry)   # 6. Save data into netCDF
    print('   Ensemble '+str(ens+1)+'/'+str(settings['num_ens'])+' done. Took:',(dt.datetime.now().replace(microsecond=0) - dt_now))
    return

def read_gia_ensemble_member(ens):
    dt_now = dt.datetime.now().replace(microsecond=0)
    global settings
    gia = {}
    file_handle = Dataset(settings['fn_gia_ens_rad'],'r')
    file_handle.set_auto_mask(False)
    gia_rad   = file_handle.variables['rad'][ens,...].astype(np.float32)
    file_handle.close()

    file_handle = Dataset(settings['fn_gia_ens_rsl'],'r')
    file_handle.set_auto_mask(False)
    gia_rsl   = file_handle.variables['rsl'][ens,...].astype(np.float32)
    file_handle.close()
    gia['gsl'] = gia_rsl + gia_rad

    file_handle = Dataset(settings['fn_gia_ens_ewh'],'r')
    file_handle.set_auto_mask(False)
    gia_ewh   = file_handle.variables['ewh'][ens,...].astype(np.float32)
    file_handle.close()

    gia['ewh'] = (mscn.masconize_gia_2d(settings['lat'],settings['lon'],gia_ewh,settings)).astype(np.float32)
    print('    ens '+str(ens+1)+' Reading GIA:', (dt.datetime.now().replace(microsecond=0) - dt_now))
    return(gia)

def perturbed_ewh_estimate(gia, ens):
    # Correct GRACE estimate with GIA ensemble member and perturb with measurement noise
    dt_now = dt.datetime.now().replace(microsecond=0)
    global grace, settings
    np.random.seed() # Reset random number generator to avoid clustering in MP
    rnd_mscn = np.random.normal(0,1,grace['ewh_ste_mscn'].shape)*grace['ewh_ste_mscn']    # Random pertubations from GRACE uncertainty
    ewh_ptb = grace['ewh'] + mscn.mascon2grid_3d(rnd_mscn, settings) - (gia['ewh']*(settings['time']-settings['time'].mean())[:,np.newaxis,np.newaxis])
    print('    ens '+str(ens+1)+' Perturbing GRACE mass:', (dt.datetime.now().replace(microsecond=0) - dt_now))
    return(ewh_ptb)

def separate_mass_ctb(ewh_ptb,ens):
    dt_now = dt.datetime.now().replace(microsecond=0)
    # Separate contributions from glaciers, ice sheets, and tws, and ocean mass per basin
    global settings, mask
    area_ocean     = (mask['area']*(mask['land']==False)).sum()
    area_altimetry = (mask['area']*(mask['slm'])).sum()
    mass_ctb = {}
    mass_ctb['ocean']     = (mask['area']*ewh_ptb*(~mask['land'])).sum(axis=(1,2))/area_ocean
    mass_ctb['land']      = (mask['area']*ewh_ptb*(mask['land'])).sum(axis=(1,2))/area_ocean
    mass_ctb['altimetry'] = (mask['area']*ewh_ptb*(mask['slm'])).sum(axis=(1,2))/area_altimetry
    mass_ctb['GrIS']      = (mask['area']*ewh_ptb*mask['GrIS']).sum(axis=(1,2))/area_ocean
    mass_ctb['AIS']       = (mask['area']*ewh_ptb*mask['AIS']).sum(axis=(1,2))/area_ocean
    # Glacier contrib: compute Zemp et al perturbed estimate
    ewh_ptb_glacier   =  ewh_ptb*mask['glacier_mask_grace']
    zemp_real = mask['zemp_rate_mean'] + mask['zemp_rate_sterr'] * np.random.normal(0,1,mask['zemp_rate_sterr'].shape)
    zemp_real = interp1d(mask['zemp_time'],np.cumsum(zemp_real,axis=1),axis=1,kind='linear', fill_value='extrapolate')(settings['time'])
    zemp_real = (zemp_real - zemp_real.mean(axis=1)[:,np.newaxis])
    for idx, reg in enumerate(mask['glacier_num_insitu']):
        ewh_ptb_glacier += zemp_real[idx,:][:,np.newaxis,np.newaxis] * mask['glacier_scale'][mask['glacier_num']==reg,...].squeeze()[np.newaxis,:,:]
    mass_ctb['glac']  = (mask['area']*ewh_ptb_glacier).sum(axis=(1,2))/area_ocean

    # TWS = Ocean mass - ctb
    mass_ctb['tws'] = mass_ctb['land'] - mass_ctb['GrIS'] - mass_ctb['AIS'] - mass_ctb['glac']

    # Ocean basins
    mass_ctb['basins']    = np.zeros([6,len(settings['time'])]) * np.nan
    for basin in range(6):
        mass_ctb['basins'][basin,:] = (mask['area']*ewh_ptb*(mask['basin']==basin)).sum(axis=(1,2))/mask['ocean_basin_area'][basin]
    print('    ens ' + str(ens+1) + ' Separating mass components:', (dt.datetime.now().replace(microsecond=0) - dt_now))
    return(mass_ctb)

def solve_sle(ewh_ptb, ens):
    dt_now = dt.datetime.now().replace(microsecond=0)
    global mask, settings
    # Solve the sea-level equation for each perturbed GRACE field
    # return RSL and radial deformation
    # D/O 90, rotation, CM
    love = np.load(settings['fn_love'],allow_pickle=True).all()
    result = pySLE3.solve(lat=settings['lat'], lon=settings['lon'], time=settings['time'], load=ewh_ptb[settings['time_mask_grace'],...]*mask['land'], slm=~mask['land'],love=love,lmax=90,rotational_feedback=True,geoid_out=False,rad_out=True,rsl_out=True,barystatic_out=False,lod_out=False,verbose=False)
    sle_result = {}
    sle_result['rad']      = np.zeros([len(settings['time']),len(settings['lat']),len(settings['lon'])],dtype=np.float32)*np.nan
    sle_result['rad_mscn'] = np.zeros([len(settings['time']),len(settings['lat']),len(settings['lon'])],dtype=np.float32)*np.nan
    sle_result['rsl_mscn'] = np.zeros([len(settings['time']),len(settings['lat']),len(settings['lon'])],dtype=np.float32)*np.nan
    sle_result['rad'][settings['time_mask_grace'],...]      = 1000*result['rad']
    sle_result['rad_mscn'][settings['time_mask_grace'],...] = 1000*mscn.masconize_3d(result['rad'],mask,settings)
    sle_result['rsl_mscn'][settings['time_mask_grace'],...] = 1000*mscn.masconize_3d(result['rsl'],mask,settings)
    print('    ens ' + str(ens+1) + ' Solving sle:', (dt.datetime.now().replace(microsecond=0) - dt_now))
    return(sle_result)

def compute_sl_altimetry(gia,sle_rad, ens):
    # Compute RSL and GSL from altimetry
    dt_now = dt.datetime.now().replace(microsecond=0)
    global mask, altimetry, settings
    sl_altimetry = {}
    np.random.seed()

    # Compute geocentric sea-level pertubations (Ablain et al. 2019)
    t_rand_highf  = gen_autocov(1.5, 1.2, 2/12, settings)
    t_rand_medf   = gen_autocov(1.2, 1, 1, settings)
    t_rand_wettr  = gen_autocov(1.1, 1.1, 5, settings)
    t_rand_lorbit = gen_autocov(0.5, 0.5, 10, settings)
    t_rand_intmis = gen_randjump(settings)
    t_rand_dorbit = gen_drift(0.1,settings)
    ptb_gmsl_alt  = (t_rand_highf + t_rand_medf + t_rand_wettr + t_rand_lorbit + t_rand_intmis + t_rand_dorbit).astype(np.float32)
    ptb_gmsl_alt -=  ptb_gmsl_alt[24:36].mean()

    # Correction for GIA and contemporary GRD
    gia_corr = (gia['gsl'][np.newaxis,...] * (settings['time']-settings['time'].mean())[:,np.newaxis,np.newaxis])
    sl_altimetry['gsl_grid'] = (altimetry['ssh']                      + ptb_gmsl_alt[:,np.newaxis,np.newaxis]).astype(np.float32)
    sl_altimetry['rsl_grid'] = (altimetry['ssh'] - sle_rad - gia_corr + ptb_gmsl_alt[:,np.newaxis,np.newaxis]).astype(np.float32)

    # Basin and global averages of relative sea level
    sl_altimetry['gsl_altimetry'] = np.zeros(len(settings['time']))
    sl_altimetry['rsl_altimetry'] = np.zeros(len(settings['time']))
    sl_altimetry['corr_grd_altimetry'] = np.zeros(len(settings['time']))
    for t in range(len(settings['time'])):
        slm = np.isfinite(sl_altimetry['gsl_grid'][t,...])
        sl_altimetry['gsl_altimetry'][t]      = np.nansum(sl_altimetry['gsl_grid'][t,...]*slm*mask['area'])/(mask['area']*slm).sum()
        sl_altimetry['rsl_altimetry'][t]      = np.nansum(sl_altimetry['rsl_grid'][t,...]*slm*mask['area'])/(mask['area']*slm).sum()
        sl_altimetry['corr_grd_altimetry'][t] = np.nansum(sle_rad[t,...]                 *slm*mask['area'])/(mask['area']*slm).sum()
    sl_altimetry['corr_gia_altimetry'] = np.nansum(gia['gsl']*mask['slm']*mask['area'])/(mask['area']*mask['slm']).sum() * (settings['time']-settings['time'].mean()) # GIA is one value (trend)
    sl_altimetry['gsl_basins']    = np.zeros([6,len(settings['time'])]) * np.nan
    sl_altimetry['rsl_basins']    = np.zeros([6,len(settings['time'])]) * np.nan
    for basin in range(6):
        sl_altimetry['gsl_basins'][basin,:] = np.nansum(mask['area']*sl_altimetry['gsl_grid']*(mask['basin']==basin),axis=(1,2))/mask['ocean_basin_area'][basin]
        sl_altimetry['rsl_basins'][basin,:] = np.nansum(mask['area']*sl_altimetry['rsl_grid']*(mask['basin']==basin),axis=(1,2))/mask['ocean_basin_area'][basin]

    # # Mask out unaccepted times
    time_acc_rsl = np.isfinite(sl_altimetry['gsl_grid'][:,200,200]) & settings['time_mask_grace']
    sl_altimetry['rsl_altimetry'][~time_acc_rsl] = np.nan
    sl_altimetry['rsl_basins'][:,~time_acc_rsl] = np.nan
    sl_altimetry['rsl_grid'][~time_acc_rsl,...] = np.nan

    # Grid to mascon
    sl_altimetry['rsl_grid'] = mscn.masconize_sealevel_3d(sl_altimetry['rsl_grid'],mask,settings)
    sl_altimetry['gsl_grid'] = mscn.masconize_sealevel_3d(sl_altimetry['gsl_grid'],mask,settings)
    print('    ens ' + str(ens+1) + ' RSL from altimetry:', (dt.datetime.now().replace(microsecond=0) - dt_now))
    return(sl_altimetry)

def save_ens_nc(ens,ewh_ptb,mass_ctb,sle_result,sl_altimetry):
    dt_now = dt.datetime.now().replace(microsecond=0)
    global mask, settings
    fname = settings['dir_save_ens'] + 'mass_rsl_ens_'+str(ens).zfill(4)+'.nc'
    file_handle = Dataset(fname, 'w')
    file_handle.createDimension('lon', len(settings['lon']))
    file_handle.createDimension('lat', len(settings['lat']))
    file_handle.createDimension('time', len(settings['time']))
    file_handle.createDimension('basin', 6)

    file_handle.createVariable('lon', 'f4', ('lon',), zlib=True)[:] = settings['lon']
    file_handle.createVariable('lat', 'f4', ('lat',), zlib=True)[:] = settings['lat']
    file_handle.createVariable('time', 'f4', ('time',), zlib=True)[:] = settings['time']
    file_handle.createVariable('basin', 'i2', ('basin',), zlib=True)[:] = np.arange(6)

    # 3D fiels
    file_handle.createVariable('ewh', 'i2', ('time', 'lat', 'lon',), zlib=True, complevel=6)[:] = ewh_ptb
    file_handle.createVariable('sle_rad_grid', 'i2', ('time', 'lat', 'lon',), zlib=True, complevel=6)[:] = 10 * sle_result['rad_mscn']
    file_handle.createVariable('sle_rsl_grid', 'i2', ('time', 'lat', 'lon',), zlib=True, complevel=6)[:] = 10 * sle_result['rsl_mscn']
    file_handle.createVariable('obs_rsl_grid', 'i2', ('time', 'lat', 'lon',), zlib=True, complevel=6)[:] = 10 * sl_altimetry['rsl_grid']
    file_handle.createVariable('obs_gsl_grid', 'i2', ('time', 'lat', 'lon',), zlib=True, complevel=6)[:] = 10 * sl_altimetry['gsl_grid']

    file_handle.variables['sle_rad_grid'].setncattr('scale_factor', 0.1)
    file_handle.variables['sle_rsl_grid'].setncattr('scale_factor', 0.1)
    file_handle.variables['obs_rsl_grid'].setncattr('scale_factor', 0.1)
    file_handle.variables['obs_gsl_grid'].setncattr('scale_factor', 0.1)

    # Mass changes
    file_handle.createVariable('mass_land', 'i2', ('time',), zlib=True, complevel=6)[:] = 20 * mass_ctb['land']
    file_handle.createVariable('mass_ocean', 'i2', ('time',), zlib=True, complevel=6)[:] = 20 * mass_ctb['ocean']
    file_handle.createVariable('mass_GrIS', 'i2', ('time',), zlib=True, complevel=6)[:] = 20 * mass_ctb['GrIS']
    file_handle.createVariable('mass_AIS', 'i2', ('time',), zlib=True, complevel=6)[:] = 20 * mass_ctb['AIS']
    file_handle.createVariable('mass_glac', 'i2', ('time',), zlib=True, complevel=6)[:] = 20 * mass_ctb['glac']
    file_handle.createVariable('mass_tws', 'i2', ('time',), zlib=True, complevel=6)[:] = 20 * mass_ctb['tws']
    file_handle.createVariable('mass_altimetry', 'i2', ('time',), zlib=True, complevel=6)[:] = 20 * mass_ctb['altimetry']
    file_handle.createVariable('mass_basins', 'i2', ('basin','time',), zlib=True, complevel=6)[:] = 20 * mass_ctb['basins']
    file_handle.variables['mass_land'].setncattr('scale_factor', 0.05)
    file_handle.variables['mass_ocean'].setncattr('scale_factor', 0.05)
    file_handle.variables['mass_GrIS'].setncattr('scale_factor', 0.05)
    file_handle.variables['mass_AIS'].setncattr('scale_factor', 0.05)
    file_handle.variables['mass_glac'].setncattr('scale_factor', 0.05)
    file_handle.variables['mass_tws'].setncattr('scale_factor', 0.05)
    file_handle.variables['mass_altimetry'].setncattr('scale_factor', 0.05)
    file_handle.variables['mass_basins'].setncattr('scale_factor', 0.05)

    # RSL changes
    file_handle.createVariable('obs_rsl_altimetry', 'i2', ('time',), zlib=True, complevel=6)[:] = 20 * sl_altimetry['rsl_altimetry']
    file_handle.createVariable('obs_rsl_basin', 'i2', ('basin','time',), zlib=True, complevel=6)[:] = 20 * sl_altimetry['rsl_basins']
    file_handle.variables['obs_rsl_altimetry'].setncattr('scale_factor', 0.05)
    file_handle.variables['obs_rsl_basin'].setncattr('scale_factor', 0.05)

    # GSL changes
    file_handle.createVariable('obs_gsl_altimetry', 'i2', ('time',), zlib=True, complevel=6)[:] = 20 * sl_altimetry['gsl_altimetry']
    file_handle.createVariable('obs_gsl_basin', 'i2', ('basin','time',), zlib=True, complevel=6)[:] = 20 * sl_altimetry['gsl_basins']
    file_handle.variables['obs_gsl_altimetry'].setncattr('scale_factor', 0.05)
    file_handle.variables['obs_gsl_basin'].setncattr('scale_factor', 0.05)

    # Corrections from GSL to RSL
    file_handle.createVariable('corr_grd_altimetry', 'i2', ('time',), zlib=True, complevel=6)[:] = 20 * sl_altimetry['corr_grd_altimetry']
    file_handle.createVariable('corr_gia_altimetry', 'i2', ('time',), zlib=True, complevel=6)[:] = 20 * sl_altimetry['corr_gia_altimetry']
    file_handle.variables['corr_grd_altimetry'].setncattr('scale_factor', 0.05)
    file_handle.variables['corr_gia_altimetry'].setncattr('scale_factor', 0.05)
    file_handle.close()
    print('    ens ' + str(ens+1) + ' Saving data:', (dt.datetime.now().replace(microsecond=0) - dt_now))
    return

# Altimetry measurement uncertainty functions
def gen_randjump(settings):
    # Jump indices
    jump_1 = np.argmin(np.abs(settings['time'] - (2008 + 8.5 / 12)))
    jump_2 = np.argmin(np.abs(settings['time'] - (2016 + 8.5 / 12)))
    j1_rnd = np.random.normal(0,1) * 0.5
    j2_rnd = np.random.normal(0,1) * 0.5
    t_rand = np.zeros(len(settings['time']))
    t_rand[jump_1:]  = j1_rnd
    t_rand[jump_2:] += j2_rnd
    return(t_rand)

def gen_drift(drift,settings):
    t_rand = drift * np.random.normal(0,1) * (settings['time'] - np.mean(settings['time']))
    return(t_rand)

def gen_autocov(sig_J1, sig_J23, l_factor, settings):
    jump_1 = np.argmin(np.abs(settings['time'] - (2008+(8.5/12))))
    sig_array = np.zeros(len(settings['time']))
    sig_array[:jump_1] = sig_J1
    sig_array[jump_1:] = sig_J23
    t_distance = np.abs(settings['time'][:, np.newaxis] - settings['time'][np.newaxis, :])
    covmat = np.exp(-0.5*(t_distance/l_factor)**2)+np.eye(len(settings['time']))*1.0e-10
    covc = np.linalg.cholesky(covmat)
    t_rand = sig_array * np.matmul(covc, np.random.randn(len(settings['time'])))
    return(t_rand)

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