# ----------------------------------------
# Read ECCO T/S and compute steric/OHC
# Regrid 2d fields onto regular lat/lon
# Save result
# ----------------------------------------
import numpy as np
from netCDF4 import Dataset
import os
import gsw
import mod_gentools as gentools
import ctypes as ct
import multiprocessing as mp
from numba import jit

def main():
    prep_settings()
    read_tile_info()
    set_grids()
    compute_climatology()
    compute_steric()
    regrid()
    save_data()
    return

def prep_settings():
    global settings
    settings = {}
    settings['dir_data']    = os.getenv('HOME')+'/Data/'
    settings['dir_ecco']     = settings['dir_data']+'ECCO/v4r4/'
    settings['fn_tile_info'] = settings['dir_ecco']+'ECCOv4r4_grid.nc'
    settings['fn_ecco_steric'] = settings['dir_ecco'] + 'ECCOv4r4_steric_ohc_shallow_1993_2017.nc'
    settings['time'] = np.arange(1993+1/24,2018+1/24,1/12)
    settings['tvec'] = np.zeros([len(settings['time']),2],dtype=int)
    settings['tvec'][:,0] = np.floor(settings['time'])
    settings['tvec'][:,1] =np.tile(np.arange(12), len(np.unique(settings['tvec'][:,0])))
    settings['nproc'] = 48
    settings['depth_range'] = np.ones(50,dtype=bool)
    #settings['depth_range'][:37] = False # Deep
    settings['depth_range'][37:] = False # Shallow
    return

def read_tile_info():
    global settings, tile_info
    tile_info = {}
    # 1. Read tile info
    fh = Dataset(settings['fn_tile_info'],'r')
    fh.set_auto_mask(False)
    tile_info['lon']       = fh.variables['XC'][:]
    tile_info['lat']       = fh.variables['YC'][:]
    tile_info['area']      = fh.variables['rA'][:]
    tile_info['depth']     = fh.variables['Z'][settings['depth_range']]
    tile_info['depth_l']   = fh.variables['Zu'][settings['depth_range']]
    tile_info['depth_u']   = fh.variables['Zl'][settings['depth_range']]
    tile_info['thickness'] = fh.variables['drF'][settings['depth_range']]
    fh.close()

    # 2. Create land sea mask
    fh = Dataset(settings['dir_ecco']+'SALT/2005/SALT_2005_01.nc','r')
    fh.set_auto_mask(False)
    tile_info['slm'] = np.squeeze(fh.variables['SALT'][:,settings['depth_range'],...]!=0)

    # 3. Pressure and volume
    tile_info['pressure'] = gsw.p_from_z(tile_info['depth'][:,np.newaxis,np.newaxis,np.newaxis],tile_info['lat'][np.newaxis,:,:,:])
    tile_info['volume'] = tile_info['thickness'][:,np.newaxis,np.newaxis,np.newaxis]*tile_info['area']
    tile_info['volume'][~tile_info['slm']] = np.nan
    fh.close()
    return

def set_grids():
    global tile_info, climatology, settings

    global CT_clim,SA_clim
    CT_clim = mp_empty_float([12,tile_info['slm'].shape[0],tile_info['slm'].shape[1],tile_info['slm'].shape[2],tile_info['slm'].shape[3]])
    SA_clim = mp_empty_float([12,tile_info['slm'].shape[0],tile_info['slm'].shape[1],tile_info['slm'].shape[2],tile_info['slm'].shape[3]])

    global thermo_steric_2d,halo_steric_2d,total_steric_2d, ohc_2d
    thermo_steric_2d = mp_empty_float([len(settings['time']),tile_info['slm'].shape[1],tile_info['slm'].shape[2],tile_info['slm'].shape[3]])
    halo_steric_2d   = mp_empty_float([len(settings['time']),tile_info['slm'].shape[1],tile_info['slm'].shape[2],tile_info['slm'].shape[3]])
    total_steric_2d  = mp_empty_float([len(settings['time']),tile_info['slm'].shape[1],tile_info['slm'].shape[2],tile_info['slm'].shape[3]])
    ohc_2d           = mp_empty_float([len(settings['time']),tile_info['slm'].shape[1],tile_info['slm'].shape[2],tile_info['slm'].shape[3]])
    return

def compute_climatology():
    global settings, CT_all, SA_all
    CT_all = mp_empty_float([len(settings['time']),tile_info['slm'].shape[0],tile_info['slm'].shape[1],tile_info['slm'].shape[2],tile_info['slm'].shape[3]])
    SA_all = mp_empty_float([len(settings['time']),tile_info['slm'].shape[0],tile_info['slm'].shape[1],tile_info['slm'].shape[2],tile_info['slm'].shape[3]])
    out = mp.Pool(settings['nproc']).map(read_clim_tstep, range(len(settings['time'])))
    CT_all = remove_trend_clim(settings['time'], CT_all)
    SA_all = remove_trend_clim(settings['time'], SA_all)
    global CT_clim, SA_clim
    for month in range(12):
        month_idx = np.where(settings['tvec'][:,1] == month)[0]
        for t in month_idx:
            CT_clim[month,...]+=CT_all[t,...]
            SA_clim[month,...]+=SA_all[t,...]
        n_years = len(np.unique(settings['tvec'][:,0]))
        CT_clim[month,...]/=n_years
        SA_clim[month,...]/=n_years
        CT_clim[month,~tile_info['slm']] = np.nan
        SA_clim[month,~tile_info['slm']] = np.nan
    CT_all = []
    SA_all = []
    return

def read_clim_tstep(tstep):
    global settings
    global CT_all, SA_all
    CT,SA = read_ecco_values(settings['tvec'][tstep,0],settings['tvec'][tstep,1],settings)
    CT_all[tstep,...] = CT
    SA_all[tstep,...] = SA
    return

def compute_steric():
    global settings
    out = mp.Pool(settings['nproc']).map(compute_steric_tstep,range(len(settings['time'])))
    return

def compute_steric_tstep(tstep):
    print('     Time step '+str(tstep)+'...')
    global settings, tile_info
    global SA_clim, CT_clim

    month = settings['tvec'][tstep,1]

    # Read data from time step
    CT,SA = read_ecco_values(settings['tvec'][tstep,0],settings['tvec'][tstep,1],settings)

    # Densities
    specvol_full = gsw.density.specvol(SA,                 CT,                 tile_info['pressure'])
    specvol_CT   = gsw.density.specvol(SA_clim[month,...], CT,                 tile_info['pressure'])
    specvol_SA   = gsw.density.specvol(SA,                 CT_clim[month,...], tile_info['pressure'])
    specvol_clim = gsw.density.specvol(SA_clim[month,...], CT_clim[month,...], tile_info['pressure'])

    # OHC and anomalies
    global thermo_steric_2d,halo_steric_2d,total_steric_2d, ohc_2d
    cp0 = 3991.8679571196

    ohc_clim = np.nansum(CT_clim[month,...] * 1/specvol_clim * tile_info['volume'] * cp0,axis=0)
    ohc_full = np.nansum(CT * 1/specvol_full      * tile_info['volume'] * cp0,axis=0)

    ohc_2d[tstep,...]           = (ohc_full-ohc_clim)/tile_info['area']
    total_steric_2d[tstep,...]  = np.nansum(1000*((specvol_full / specvol_clim) - 1) * tile_info['thickness'][:,np.newaxis,np.newaxis,np.newaxis],axis=0)
    thermo_steric_2d[tstep,...] = np.nansum(1000*((specvol_CT   / specvol_clim) - 1) * tile_info['thickness'][:,np.newaxis,np.newaxis,np.newaxis],axis=0)
    halo_steric_2d[tstep,...]   = np.nansum(1000*((specvol_SA   / specvol_clim) - 1) * tile_info['thickness'][:,np.newaxis,np.newaxis,np.newaxis],axis=0)
    return

def regrid():
    global settings, tile_info
    global thermo_steric_2d,halo_steric_2d,total_steric_2d, ohc_2d
    global thermo_steric_2d_regrid,halo_steric_2d_regrid,total_steric_2d_regrid, ohc_2d_regrid

    thermo_steric_2d_regrid,halo_steric_2d_regrid,total_steric_2d_regrid,ohc_2d_regrid = interp_ecco(tile_info['lat'], tile_info['lon'], settings['time'], thermo_steric_2d, halo_steric_2d,total_steric_2d,ohc_2d,np.arange(-89.75,90.25,0.5), np.arange(-179.75,180.25,0.5))
    thermo_steric_2d_regrid = np.dstack([thermo_steric_2d_regrid[:,:,360:],thermo_steric_2d_regrid[:,:,:360]])
    halo_steric_2d_regrid = np.dstack([halo_steric_2d_regrid[:,:,360:],halo_steric_2d_regrid[:,:,:360]])
    total_steric_2d_regrid = np.dstack([total_steric_2d_regrid[:,:,360:],total_steric_2d_regrid[:,:,:360]])
    lat = np.arange(-89.75, 90.25, 0.5)
    lon = np.arange(0.25, 360.25, 0.5)
    area = gentools.grid_area(lat,lon)
    ohc_2d_regrid = np.dstack([ohc_2d_regrid[:,:,360:],ohc_2d_regrid[:,:,:360]]) * area
    return

def save_data():
    print('   Saving...')
    global settings, tile_info
    global thermo_steric_2d_regrid,halo_steric_2d_regrid,total_steric_2d_regrid, ohc_2d_regrid

    # SLM
    slm = ~((ohc_2d_regrid==0).sum(axis=0)==len(settings['time']))
    lat = np.arange(-89.75,90.25,0.5)
    lon = np.arange(0.25,360.25,0.5)
    area = gentools.grid_area(lat,lon)
    # Compute global-mean time series
    total_steric_ts  = np.nansum(area*slm*total_steric_2d_regrid,axis=(1,2)) / (area*slm).sum()
    halo_steric_ts   = np.nansum(area*slm*halo_steric_2d_regrid,axis=(1,2)) / (area*slm).sum()
    thermo_steric_ts = np.nansum(area*slm*thermo_steric_2d_regrid,axis=(1,2)) / (area*slm).sum()
    ohc_ts = np.nansum(slm*ohc_2d_regrid,axis=(1,2))

    # Mask out NaN values
    thermo_steric_2d_regrid[:,~slm] = np.nan
    halo_steric_2d_regrid[:,~slm] = np.nan
    total_steric_2d_regrid[:,~slm] = np.nan
    ohc_2d_regrid[:,~slm] = np.nan

    # Save data
    fh = Dataset(settings['fn_ecco_steric'],'w')
    fh.createDimension('time', len(settings['time']))
    fh.createDimension('lat', len(lat))
    fh.createDimension('lon', len(lon))

    fh.createVariable('time', 'f4', ('time',), zlib=True, complevel=4)[:] = settings['time']
    fh.createVariable('lat', 'f4', ('lat'), zlib=True, complevel=4)[:] = lat
    fh.createVariable('lon', 'f4', ('lon',), zlib=True, complevel=4)[:] = lon
    fh.createVariable('slm', 'u1', ('lat', 'lon',), zlib=True, complevel=4)[:] = slm

    # 2D fields
    var_save = ohc_2d_regrid
    fh.createVariable('ohc_2d', 'f4', ('time','lat', 'lon',), zlib=True, complevel=4,least_significant_digit=4)[:] = var_save

    var_save = thermo_steric_2d_regrid*20
    var_save[np.isnan(var_save)] = - 20000
    fh.createVariable('thermosteric_2d', 'i2', ('time','lat', 'lon',), zlib=True, complevel=4,least_significant_digit=4)[:] = np.rint(var_save).astype(int)
    fh.variables['thermosteric_2d'].scale_factor = 1/20
    fh.variables['thermosteric_2d'].missing_value = -20000

    var_save = halo_steric_2d_regrid*20
    var_save[np.isnan(var_save)] = - 20000
    fh.createVariable('halosteric_2d', 'i2', ('time','lat', 'lon',), zlib=True, complevel=4,least_significant_digit=4)[:] = np.rint(var_save).astype(int)
    fh.variables['halosteric_2d'].scale_factor = 1/20
    fh.variables['halosteric_2d'].missing_value = -20000

    var_save = total_steric_2d_regrid*20
    var_save[np.isnan(var_save)] = - 20000
    fh.createVariable('totalsteric_2d', 'i2', ('time','lat', 'lon',), zlib=True, complevel=4,least_significant_digit=4)[:] = np.rint(var_save).astype(int)
    fh.variables['totalsteric_2d'].scale_factor = 1/20
    fh.variables['totalsteric_2d'].missing_value = -20000

    # Global time series
    fh.createVariable('ohc_ts', 'f4', ('time'), zlib=True, complevel=4)[:] = ohc_ts
    fh.createVariable('thermosteric_ts', 'f4', ('time'), zlib=True, complevel=4)[:] = thermo_steric_ts
    fh.createVariable('halosteric_ts', 'f4', ('time'), zlib=True, complevel=4)[:] = halo_steric_ts
    fh.createVariable('totalsteric_ts', 'f4', ('time'), zlib=True, complevel=4)[:] = total_steric_ts
    fh.close()
    print('Done')
    return

def read_ecco_values(year,month,settings):
    fn_sal   = settings['dir_ecco']+'SALT/'+str(year)+'/SALT_'+str(year)+'_'+str(month+1).zfill(2)+'.nc'
    fn_theta = settings['dir_ecco']+'THETA/'+str(year)+'/THETA_'+str(year)+'_'+str(month+1).zfill(2)+'.nc'

    fh=Dataset(fn_sal,'r')
    fh.set_auto_mask(False)
    psal = np.squeeze(fh.variables['SALT'][:,settings['depth_range'],...])
    fh.close()

    fh=Dataset(fn_theta,'r')
    fh.set_auto_mask(False)
    theta = np.squeeze(fh.variables['THETA'][:,settings['depth_range'],...])
    fh.close()

    psal[~tile_info['slm']] = np.nan
    theta[~tile_info['slm']] = np.nan

    pressure = gsw.p_from_z(tile_info['depth'][:,np.newaxis,np.newaxis,np.newaxis],tile_info['lat'][np.newaxis,...])
    SA     = gsw.SA_from_SP(psal,pressure,tile_info['lon'][np.newaxis,...],tile_info['lat'][np.newaxis,...])
    CT     = gsw.CT_from_pt(psal,theta)
    return(CT,SA)

def interp_ecco(lat_in,lon_in,time_in,field_1,field_2,field_3,field_4,lat_out,lon_out):
    print('   Regridding...')
    field_out_1 = mp_empty_float((len(time_in),len(lat_out),len(lon_out)))
    field_out_2 = mp_empty_float((len(time_in),len(lat_out),len(lon_out)))
    field_out_3 = mp_empty_float((len(time_in),len(lat_out),len(lon_out)))
    field_out_4 = mp_empty_float((len(time_in),len(lat_out),len(lon_out)))
    for i in range(len(lat_out)):
        for j in range(len(lon_out)):
            coords = np.unravel_index(np.argmin((lat_in-lat_out[i])**2 + (lon_in-lon_out[j])**2),lat_in.shape)
            field_out_1[:,i,j] = field_1[:,coords[0],coords[1],coords[2]]
            field_out_2[:,i,j] = field_2[:,coords[0],coords[1],coords[2]]
            field_out_3[:,i,j] = field_3[:,coords[0],coords[1],coords[2]]
            field_out_4[:,i,j] = field_4[:,coords[0],coords[1],coords[2]]
    return(field_out_1,field_out_2,field_out_3,field_out_4)

@jit(nopython=True)
def remove_trend_clim(time,field):
    amat = np.ones((len(time),6))
    amat[:,1] = time - time.mean()
    amat[:,2] = np.sin(2*np.pi*(time- time.mean()))
    amat[:,3] = np.cos(2*np.pi*(time- time.mean()))
    amat[:,4] = np.sin(4*np.pi*(time- time.mean()))
    amat[:,5] = np.cos(4*np.pi*(time- time.mean()))
    amat_T  = amat.T
    amat_sq = np.linalg.inv(np.dot(amat_T, amat))
    for i in range(field.shape[1]):
        for j in range(field.shape[2]):
            for k in range(field.shape[3]):
                for l in range(field.shape[4]):
                    sol = np.dot(amat_sq, np.dot(amat_T, field[:, i, j,k,l].astype(np.float64)))
                    sol[0] = 0
                    sol[2:] = 0
                    field[:, i, j,k,l]-=amat@sol
    return(field)

# ----- Multiprocessing routines
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

if __name__ == '__main__':
    main()