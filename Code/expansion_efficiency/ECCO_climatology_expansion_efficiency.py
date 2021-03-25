# -------------------------------------------------------------
# Compute ECCO climatology and make map of expansion efficiency
# -------------------------------------------------------------
import numpy as np
from netCDF4 import Dataset
import os
import gsw

def main():
    settings = set_settings()
    tile_info = read_tile_info(settings)
    CT_clim,SA_clim = compute_climatology(tile_info, settings)
    efficiency = compute_efficiency(CT_clim, SA_clim, tile_info, settings)
    eff_reg    = regrid_efficiency(efficiency, tile_info, settings)
    save_data(eff_reg, settings)
    return

def set_settings():
    settings = {}
    settings['dir_data']    = os.getenv('HOME')+'/Data/'
    settings['dir_ecco']     = settings['dir_data']+'ECCO/v4r4/'
    settings['fn_tile_info'] = settings['dir_ecco']+'ECCOv4r4_grid.nc'
    settings['fn_ecco_efficiency'] = settings['dir_ecco'] + 'efficiency_clim_1993_2017.nc'
    settings['time'] = np.arange(1993+1/24,2018+1/24,1/12)
    settings['tvec'] = np.zeros([len(settings['time']),2],dtype=int)
    settings['tvec'][:,0] = np.floor(settings['time'])
    settings['tvec'][:,1] =np.tile(np.arange(12), len(np.unique(settings['tvec'][:,0])))
    settings['depth_range'] = np.ones(50,dtype=bool)
    return(settings)

def read_tile_info(settings):
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
    tile_info['pressure'] = gsw.p_from_z(tile_info['depth'][:,np.newaxis,np.newaxis,np.newaxis],tile_info['lat'][np.newaxis,:,:,:])
    fh.close()
    return(tile_info)

def compute_climatology(tile_info, settings):
    CT_clim = np.zeros([tile_info['slm'].shape[0],tile_info['slm'].shape[1],tile_info['slm'].shape[2],tile_info['slm'].shape[3]])
    SA_clim = np.zeros([tile_info['slm'].shape[0],tile_info['slm'].shape[1],tile_info['slm'].shape[2],tile_info['slm'].shape[3]])

    for t in range(len(settings['time'])):
        print(t)
        CT,SA=read_ecco_values(settings['tvec'][t,0],settings['tvec'][t,1],tile_info,settings)
        CT_clim+=CT
        SA_clim+=SA
    CT_clim/=len(settings['time'])
    SA_clim/=len(settings['time'])
    return(CT_clim,SA_clim)

def compute_efficiency(CT_clim,SA_clim,tile_info, settings):
    cp0      = 3991.86795711963 # J/kg/K
    # Eff (m/J) = 1/area * dVs/dT * 1/cp0
    efficiency = (1/tile_info['area'][tile_info['slm'][0,...]].sum()) * gsw.specvol_first_derivatives(SA_clim,CT_clim,tile_info['pressure'])[1] * (1/cp0)  * 1000
    return(efficiency)

def regrid_efficiency(efficiency,tile_info,settings):
    print('   Regridding...')
    lat_out = np.arange(-89.75,90.25,0.5)
    lon_out = np.arange(-179.75,180.25,0.5)
    lat_in = tile_info['lat']
    lon_in = tile_info['lon']
    efficiency_regrid = np.zeros((efficiency.shape[0], len(lat_out), len(lon_out)))
    for i in range(len(lat_out)):
        for j in range(len(lon_out)):
            coords = np.unravel_index(np.argmin((lat_in - lat_out[i]) ** 2 + (lon_in - lon_out[j]) ** 2), lat_in.shape)
            efficiency_regrid[:, i, j] = efficiency[:, coords[0], coords[1], coords[2]]
    eff_reg = {}
    eff_reg['lat'] = lat_out
    eff_reg['lon'] = lon_out
    eff_reg['depth'] = tile_info['depth']
    eff_reg['efficiency'] = efficiency_regrid
    return(eff_reg)

def save_data(eff_reg,settings):
    fh = Dataset(settings['fn_ecco_efficiency'],'w')
    fh.createDimension('depth', len(eff_reg['depth']))
    fh.createDimension('lat', len(eff_reg['lat']))
    fh.createDimension('lon', len(eff_reg['lon']))

    fh.createVariable('depth', 'f4', ('depth',), zlib=True, complevel=4)[:] = eff_reg['depth']
    fh.createVariable('lat', 'f4', ('lat'), zlib=True, complevel=4)[:] = eff_reg['lat']
    fh.createVariable('lon', 'f4', ('lon',), zlib=True, complevel=4)[:] = eff_reg['lon']

    fh.createVariable('eff_mm_zj', 'f4', ('depth','lat', 'lon',), zlib=True, complevel=4)[:] = eff_reg['efficiency']*1e21
    fh.createVariable('eff_mmyrwm2', 'f4', ('depth','lat', 'lon',), zlib=True, complevel=4)[:] = eff_reg['efficiency']*(3600*24*365.25)*(4*np.pi*6371000**2)

    fh.close()
    print('Done')
    return


def read_ecco_values(year,month,tile_info,settings):
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

if __name__ == '__main__':
    main()
