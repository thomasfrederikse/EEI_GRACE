# Read expansion efficiency maps and average over longitudes
import numpy as np
from netCDF4 import Dataset
import os
import mod_gentools as gentools
from scipy.interpolate import interp1d

def main():
    settings = {}
    settings['dir_data'] = os.getenv('HOME') + '/Data/'
    settings['fn_woa_efficiency'] = settings['dir_data'] + 'Steric/WOA_climatology/efficiency_clim_2005_2017.nc'
    settings['fn_ecco_efficiency'] = settings['dir_data'] + 'ECCO/v4r4/efficiency_clim_1993_2017.nc'
    # settings['dir_gmt'] = os.getenv('HOME') + '/Scripts/GMT/Papers/GRACE_budget/efficiency/'
    settings['dir_gmt'] = os.getenv('HOME') + '/Scripts/GMT/Papers/GRACE_budget/ts_bdg_eff_glb/'
    settings['dlevels'] = np.arange(7000,0,-10)
    settings['fn_efficiency']       = settings['dir_data'] + 'Budget_GRACE/efficiency/efficiency_combined.npy'


    WOA = {}
    fh = Dataset(settings['fn_woa_efficiency'],'r')
    fh.set_auto_mask(False)
    WOA['depth'] = fh['depth'][:]
    WOA['lat'] = fh['lat'][:]
    WOA['lon'] = fh['lon'][:]
    WOA['efficiency'] = fh['eff_mmyrwm2'][:]
    fh.close()

    ECCO = {}
    fh = Dataset(settings['fn_ecco_efficiency'],'r')
    fh.set_auto_mask(False)
    ECCO['depth'] = fh['depth'][:]
    ECCO['lat'] = fh['lat'][:]
    ECCO['lon'] = fh['lon'][:]
    ECCO['efficiency'] = fh['eff_mmyrwm2'][:]
    fh.close()


    WOA['eff_int'] = interp1d(-WOA['depth'],np.nanmean(WOA['efficiency'],axis=2),axis=0,fill_value=np.nan,kind='linear',bounds_error=False)(settings['dlevels'])
    ECCO['eff_int'] = interp1d(-ECCO['depth'],np.nanmean(ECCO['efficiency'],axis=2),axis=0,fill_value=np.nan,kind='linear',bounds_error=False)(settings['dlevels'])

    # Save data
    fh = Dataset(settings['dir_gmt']+'eff_woa.nc','w')
    fh.createDimension('depth', len(settings['dlevels']))
    fh.createDimension('lat', len(WOA['lat']))

    fh.createVariable('depth', 'f4', ('depth',), zlib=True, complevel=4)[:] = -settings['dlevels']
    fh.createVariable('lat', 'f4', ('lat'), zlib=True, complevel=4)[:] = WOA['lat']
    fh.createVariable('efficiency', 'f4', ('depth','lat'), zlib=True, complevel=4)[:] = WOA['eff_int']
    fh.close()

    fh = Dataset(settings['dir_gmt']+'eff_ecco.nc','w')
    fh.createDimension('depth', len(settings['dlevels']))
    fh.createDimension('lat', len(ECCO['lat']))

    fh.createVariable('depth', 'f4', ('depth',), zlib=True, complevel=4)[:] = -settings['dlevels']
    fh.createVariable('lat', 'f4', ('lat'), zlib=True, complevel=4)[:] = ECCO['lat']
    fh.createVariable('efficiency', 'f4', ('depth','lat'), zlib=True, complevel=4)[:] = ECCO['eff_int']
    fh.close()

    efficiency = np.load(settings['fn_efficiency'],allow_pickle=True).all()
    # Bar graph
    gmt_save_trend_ci(settings['dir_gmt'], 'ORAS_shallow', 0.7, efficiency['oras5']['shallow']['alt']['eff_rolling'].mean(), efficiency['oras5']['shallow']['alt']['eff_rolling'].std())
    gmt_save_trend_ci(settings['dir_gmt'], 'ECCO_shallow', 0.9, efficiency['ecco']['shallow']['alt']['eff_rolling'].mean(), efficiency['ecco']['shallow']['alt']['eff_rolling'].std())
    gmt_save_trend_ci(settings['dir_gmt'], 'mean_shallow', 1.1, efficiency['shallow'][0], efficiency['shallow'][1])
    gmt_save_trend_ci(settings['dir_gmt'], 'obs_shallow',  1.3, efficiency['obs']['trend_alt'].mean(), efficiency['obs']['trend_alt'].std())

    gmt_save_trend_ci(settings['dir_gmt'], 'ORAS_full', 1.7, efficiency['oras5']['full']['alt']['eff_rolling'].mean(), efficiency['oras5']['full']['alt']['eff_rolling'].std())
    gmt_save_trend_ci(settings['dir_gmt'], 'ECCO_full', 1.9, efficiency['ecco']['full']['alt']['eff_rolling'].mean(), efficiency['ecco']['full']['alt']['eff_rolling'].std())
    gmt_save_trend_ci(settings['dir_gmt'], 'mean_full', 2.1, efficiency['full'][0], efficiency['full'][1])
    return


def gmt_save_trend_ci(gmt_dir,fname,position,trend,ci):
    save_array = np.array([position,trend/1e-21,1.65*ci/1e-21])
    np.savetxt(gmt_dir + fname + '_trend.txt', save_array[np.newaxis,:], fmt='%4.3f;%4.3f;%4.3f')
    return