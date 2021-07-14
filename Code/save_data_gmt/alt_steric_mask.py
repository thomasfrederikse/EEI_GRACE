# Save masks from altimetry and steric
import numpy as np
from netCDF4 import Dataset
import os
from scipy.interpolate import interp2d
import mod_gentools as gentools
import matplotlib.pyplot as plt

def main():
    dir_data = os.getenv('HOME') + '/Data/'
    fn_gmt = os.getenv('HOME') + '/Scripts/GMT/Papers/GRACE_budget/mask_map/mask.nc'
    fn_steric = dir_data + 'Budget_GRACE/steric/steric_indiv.npy'
    fn_mask   = dir_data + 'GRACE/JPL_mascon/mask.npy'

    mask = np.load(fn_mask,allow_pickle=True).all()
    steric = np.load(fn_steric,allow_pickle=True).all()

    file_handle = Dataset(fn_gmt, 'w')
    file_handle.createDimension('lon', len(mask['lon']))
    file_handle.createDimension('lat', len(mask['lat']))

    file_handle.createVariable('lon', 'f4', ('lon',), zlib=True)[:] = mask['lon']
    file_handle.createVariable('lat', 'f4', ('lat',), zlib=True)[:] = mask['lat']

    file_handle.createVariable('region', 'f4', ('lat','lon',), zlib=True)[:] = mask['basin']
    file_handle.createVariable('slm', 'i2', ('lat','lon',), zlib=True)[:] = mask['slm']
    area = gentools.grid_area(mask['lat'],mask['lon'])
    for prod in steric:
        steric[prod]['mask']['slm_glb']
        mask_int = np.rint(interp2d(steric[prod]['mask']['lon'], steric[prod]['mask']['lat'], steric[prod]['mask']['slm_glb'], kind='linear')(mask['lon'], mask['lat']))
        mask_int[(mask['lat'] < steric[prod]['mask']['lat'].min()) | (mask['lat'] > steric[prod]['mask']['lat'].max())] = 0
        file_handle.createVariable(prod, 'i2', ('lat', 'lon',), zlib=True)[:] = mask_int
        print(np.sum(area*mask_int*mask['slm'])/np.sum(area*mask['slm']))
    file_handle.close()


    return
