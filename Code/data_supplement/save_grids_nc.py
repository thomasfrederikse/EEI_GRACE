# ---------------------------------------------
# Read the NetCDF file with time series and
# trends and reformat into netCDF file with
# more details
# ---------------------------------------------
import numpy as np
from netCDF4 import Dataset, date2num
import datetime as dt
import os

def main():
    settings = {}
    settings['fn_read'] = os.getenv('HOME')+'/Data/Budget_GRACE/stats/grid_trends.nc'
    settings['fn_write'] = os.getenv('HOME')+'/Data/Budget_GRACE/data_supplement/grid_tseries.nc'
    settings['fn_mask'] = os.getenv('HOME')+'/Data/GRACE/JPL_mascon/mask.npy'
    settings['time_mask_grace'] = np.ones(204,dtype=bool)
    settings['time_mask_grace'][174:185] = False

    # Read mask
    mask = np.load(settings['fn_mask'],allow_pickle=True).all()

    # Read data
    fh_read = Dataset(settings['fn_read'],'r')
    fh_read.set_auto_mask(False)
    lat = fh_read['lat'][:]
    lon = fh_read['lon'][:]
    time = fh_read['time'][:]
    date_array = []
    for year in range(2003,2020):
        for month in range(1,13):
            date_array.append(dt.datetime(year, month, 15))

    fh = Dataset(settings['fn_write'], 'w')
    fh.copyright   = '(c) 2021 California Institute of Technology. Government sponsorship acknowledged.'
    fh.information = "This work is a data supplement to 'Earth’s Energy Imbalance from the ocean perspective (2005 - 2019)' by Maria Hakuba, Thomas Frederikse, Felix Landerer"

    # Dimensions
    fh.createDimension('lon', len(lon))
    fh.createDimension('lat', len(lat))
    fh.createDimension('time', len(time))

    # Basic variables
    fh_lon = fh.createVariable('lon', 'f4', ('lon'), zlib=True,complevel=9)
    fh_lon[:] = lon
    fh_lon.long_name = 'Longitude'
    fh_lon.units = 'Degrees East'

    fh_lat = fh.createVariable('lat', 'f4', ('lat'), zlib=True,complevel=9)
    fh_lat[:] = lat
    fh_lat.long_name = 'Latitude'
    fh_lat.units = 'Degrees North'

    fh_time = fh.createVariable('time', 'i2', ('time'), zlib=True,complevel=9)
    fh_time[:] = date2num(date_array,'days since 2000-01-01 00:00:00 UTC')
    fh_time.units = 'days since 2000-01-01 00:00:00 UTC'
    fh_time.long_name = 'Time'

    mean_save = np.rint(20*fh_read['steric_tseries'][:]).astype(int)
    mean_save[:,~mask['slm']] = -20000
    fh_grid = fh.createVariable('Steric', 'i2', ('time','lat','lon'), zlib=True,complevel=9)
    fh_grid[:] = mean_save
    fh_grid.missing_value = -20000
    fh_grid.scale_factor=0.05
    fh_grid.units = 'mm'
    fh_grid.long_name = 'Steric sea-level anomalies'

    mean_save = np.rint(20*fh_read['mass_tseries'][:]).astype(int)
    mean_save[:,~mask['slm']] = -20000
    mean_save[~settings['time_mask_grace'],:,:] = -20000
    fh_grid = fh.createVariable('Ocean mass', 'i2', ('time','lat','lon'), zlib=True,complevel=9)
    fh_grid[:] = mean_save
    fh_grid.missing_value = -20000
    fh_grid.scale_factor = 0.05
    fh_grid.units = 'mm'
    fh_grid.long_name = ' Ocean mass anomalies'

    mean_save = np.rint(20*fh_read['gsl_tseries'][:]).astype(int)
    mean_save[:,~mask['slm']] = -20000
    mean_save[~settings['time_mask_grace'],:,:] = -20000
    fh_grid = fh.createVariable('GSL', 'i2', ('time','lat','lon'), zlib=True,complevel=9)
    fh_grid[:] = mean_save
    fh_grid.missing_value = -20000
    fh_grid.scale_factor = 0.05
    fh_grid.units = 'mm'
    fh_grid.long_name = ' Geocentric sea-level anomalies'

    mean_save = np.rint(20*fh_read['rsl_tseries'][:]).astype(int)
    mean_save[:,~mask['slm']] = -20000
    mean_save[~settings['time_mask_grace'],:,:] = -20000
    fh_grid = fh.createVariable('RSL', 'i2', ('time','lat','lon'), zlib=True,complevel=9)
    fh_grid[:] = mean_save
    fh_grid.missing_value = -20000
    fh_grid.scale_factor = 0.05
    fh_grid.units = 'mm'
    fh_grid.long_name = ' Relative sea-level anomalies'
    fh.close()
    return




