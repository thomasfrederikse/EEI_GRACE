# =============================================
# Create masks for GRACE budget
# 1. Global mask
# 2. Altimetry mask
#    - All points where altimetry has coverage
# 3. Basin masks
#    - All points where altimetry has coverage
# 4. Ice sheet and glacier masks
# =============================================
import numpy as np
import os
from netCDF4 import Dataset
import mod_gentools as gentools
from scipy import stats
def main():
    settings = {}
    settings['dir_data']    = os.getenv('HOME') + '/Data/'
    settings['fn_grace_mask_sl'] = settings['dir_data'] + 'GRACE/JPL_mascon/LAND_MASK.CRIv01.nc'
    settings['fn_grace_mask_hy'] = settings['dir_data'] + 'GRACE/JPL_mascon/CLM4.SCALE_FACTOR.JPL.MSCNv01CRIv01.nc'
    settings['fn_mascon_coords'] = settings['dir_data'] + 'GRACE/JPL_mascon/mascon_coords.npy'
    settings['fn_gfo'] = settings['dir_data'] + 'GRACE/JPL_mascon/JPL_mascon_RL06v02_CRI_noGIA_noEQ_noseas.nc'

    settings['fn_rgi_mask']   = settings['dir_data']    + 'Glaciers/RGI/regions.nc'
    settings['fn_basin_mask'] = settings['dir_data']    + 'Basins/ocean_regions_thompson.grd'
    settings['fn_mask_nc'] = settings['dir_data'] +'GRACE/JPL_mascon/mask.nc'
    settings['fn_mask_npy'] = settings['dir_data'] +'GRACE/JPL_mascon/mask.npy'

    settings['fn_alt'] = settings['dir_data'] +'Altimetry/CDS/CDS_monthly_2003_2019.nc'

    grace_mask = read_GRACE_mask(settings)
    basin_mask = read_basin_mask(settings)
    glacier_mask = read_glacier_mask(grace_mask, settings)
    alt_mask = read_alt_mask(settings)
    mscn_mask = comp_mscn_mask(grace_mask, basin_mask, alt_mask, settings)
    mask = collect_masks(grace_mask, glacier_mask, basin_mask, mscn_mask, alt_mask,settings)
    save_mask(mask, settings)
    return

def collect_masks(grace_mask, glacier_mask, basin_mask,mscn_mask, alt_mask,settings):
    print('   Computing all masks...')
    mask = {}
    mask['lat'] = grace_mask['lat'].astype(np.float32)
    mask['lon'] = grace_mask['lon'].astype(np.float32)
    mask['land'] = grace_mask['mask_land']

    # Ice sheet masks
    msk_IS = grace_mask['mask_land'] & (~grace_mask['mask_no_is'])
    mask['GrIS'] = msk_IS.copy()
    mask['AIS'] = msk_IS.copy()
    mask['GrIS'][:180, :] = False
    mask['AIS'][180:, :] = False

    # Glacier mask
    mask['glacier_num_grace']   = glacier_mask['num_grace']
    mask['glacier_num_insitu']  = glacier_mask['num_insitu']
    mask['glacier_num']         = glacier_mask['num']

    mask['glacier_scale']       = glacier_mask['scale'].astype(np.float32)
    mask['glacier_mask_all']    = glacier_mask['mask_all']
    mask['glacier_mask_grace']  = glacier_mask['mask_grace']
    mask['glacier_mask_insitu'] = glacier_mask['mask_insitu']

    mask['basin'] = mscn_mask['basin'].astype(np.float32)
    mask['slm'] = mscn_mask['global']

    # TWS mask
    mask['tws'] = grace_mask['mask_no_is'] & (~mask['glacier_mask_all'])
    return(mask)

def save_mask(mask, settings):
    np.save(settings['fn_mask_npy'],mask)
    # Prepare glaciers
    rgi_num = np.sort(np.hstack((mask['glacier_num_grace'], mask['glacier_num_insitu'])))
    rgi_grace = np.zeros(len(rgi_num),dtype=bool)
    rgi_grace[np.in1d(rgi_num,mask['glacier_num_grace'])] = True

    file_handle = Dataset(settings['fn_mask_nc'], 'w')
    file_handle.createDimension('lat', len(mask['lat']))
    file_handle.createDimension('lon', len(mask['lon']))
    file_handle.createDimension('rgi', len(rgi_num))

    file_handle.createVariable('lon', 'f4', ('lon',),zlib=True)[:] = mask['lon']
    file_handle.createVariable('lat', 'f4', ('lat',),zlib=True)[:] = mask['lat']
    file_handle.createVariable('rgi', 'i2', ('rgi',),zlib=True)[:] = rgi_num
    file_handle.createVariable('rgi_grace', 'i2', ('rgi',),zlib=True)[:] = rgi_grace

    file_handle.createVariable('land', 'i2', ('lat', 'lon',),zlib=True)[:] = mask['land']
    file_handle.createVariable('slm', 'i2', ('lat', 'lon',),zlib=True)[:] = mask['slm']
    file_handle.createVariable('ocean_basin', 'f4', ('lat', 'lon',),zlib=True)[:] = mask['basin']
    file_handle.createVariable('GrIS', 'i2', ('lat', 'lon',),zlib=True)[:] = mask['GrIS']
    file_handle.createVariable('AIS', 'i2', ('lat', 'lon',),zlib=True)[:] = mask['AIS']
    file_handle.createVariable('glaciers', 'i2', ('lat', 'lon',),zlib=True)[:] = mask['glacier_mask_all']
    file_handle.createVariable('glaciers_grace', 'i2', ('lat', 'lon',),zlib=True)[:] = mask['glacier_mask_grace']
    file_handle.createVariable('glaciers_small', 'i2', ('lat', 'lon',),zlib=True)[:] = mask['glacier_mask_insitu']
    file_handle.createVariable('tws', 'i2', ('lat', 'lon',),zlib=True)[:] = mask['tws']
    file_handle.createVariable('glacier_scale', 'f4', ('rgi','lat', 'lon',),zlib=True)[:] = mask['glacier_scale']
    file_handle.close()
    return

def comp_mscn_mask(grace_mask,basin_mask,alt_mask,settings):
    # Read GRACE and get unique mascons with CRI
    fh = Dataset(settings['fn_gfo'],'r')
    fh.set_auto_mask(False)
    ewh_grace = fh.variables['ewh'][:]
    uniq_vals_0 = np.unique(ewh_grace[0, :, :])
    mascon_slm = np.zeros(len(uniq_vals_0),dtype=bool)
    for i in range(len(uniq_vals_0)):
        mascon_slm[i] = (grace_mask['mask_land'][(ewh_grace[0, :, :]==uniq_vals_0[i])]).sum() == 0

    mscn_mask = {}
    mscn_mask['global'] = np.zeros(grace_mask['mask_land'].shape,dtype=bool)
    mscn_mask['basin']  = np.zeros(grace_mask['mask_land'].shape)*np.nan
    for i in range(len(mascon_slm)):
        if mascon_slm[i]:
            # Number of GRACE mascon points
            mscn_lcl = ewh_grace[0, :, :] == uniq_vals_0[i]
            n_mscn = mscn_lcl.sum()
            n_alt = alt_mask['slm'][mscn_lcl].sum()
            if n_alt/n_mscn > 0.5:
                mscn_mask['global'][mscn_lcl] = True
                basin_nums = basin_mask['num'][mscn_lcl]
                basin_nums = basin_nums[np.isfinite(basin_nums)]
                if len(basin_nums)>0:
                    mscn_mask['basin'][mscn_lcl] = stats.mode(basin_nums).mode[0]
    mscn_cp = mscn_mask['basin'].copy()
    mscn_mask['basin'][197:202, 553:569] = 2
    mscn_mask['basin'][np.isnan(mscn_cp)] = np.nan
    mscn_mask['basin'][197,561] = 3
    mscn_mask['basin'][197,562] = 3
    mscn_mask['basin'][207,538] = 3
    mscn_mask['basin'][207,539] = 3
    mscn_mask['basin'][201,548] = 3
    return(mscn_mask)

def read_GRACE_mask(settings):
    # Read GRACE mask and mascon definitions
    print('   Processing GRACE masks...')
    grace_mask = {}
    grace_mask['lat'] = Dataset(settings['fn_grace_mask_sl'], 'r').variables["lat"][:]._get_data()
    grace_mask['lon'] = Dataset(settings['fn_grace_mask_sl'], 'r').variables["lon"][:]._get_data()
    grace_mask['mask_land'] = Dataset(settings['fn_grace_mask_sl'], 'r').variables["land_mask"][:]._get_data().astype(bool)
    grace_mask['mask_no_is'] = (Dataset(settings['fn_grace_mask_hy'], 'r').variables["scale_factor"][:]._get_data()>-9999)
    grace_mask['mscn_coords'] = np.load(settings['fn_mascon_coords'], allow_pickle=True)
    return(grace_mask)

def read_alt_mask(settings):
    print('   Processing altimetry masks...')
    alt_mask = {}
    fh=Dataset(settings['fn_alt'],'r')
    fh.set_auto_mask(False)
    ssh = fh.variables['ssh'][:]
    ssh[ssh>1500] = np.nan
    # tst = np.sum(np.isfinite(ssh),axis=0)
    alt_mask['slm'] = np.sum(np.isfinite(ssh),axis=0)>=(0.4*ssh.shape[0])
    alt_mask['lat'] = fh.variables['lat'][:]
    alt_mask['lon'] = fh.variables['lon'][:]
    fh.close()
    return(alt_mask)

def read_basin_mask(settings):
    basin_mask = {}
    file_handle = Dataset(settings['fn_basin_mask'], 'r')
    file_handle.set_auto_mask(False)
    basin_mask['num'] = file_handle.variables['z'][:]
    file_handle.close()
    basin_mask['num'][basin_mask['num'] == 0] = np.nan
    basin_mask['num'] = basin_mask['num'] - 1
    return(basin_mask)

def read_glacier_mask(grace_mask, settings):
    print('   Processing glacier masks...')
    glacier_mask = {}
    glacier_mask['lat'] = grace_mask['lat']
    glacier_mask['lon'] = grace_mask['lon']
    area = gentools.grid_area(grace_mask['lat'],grace_mask['lon'])

    # Read RGI masks and areas and remove the Greenland and Antarctic Periphery
    glacier_mask['num_grace']   = np.array([1, 3, 4, 6, 7, 9, 17]) # Glacier regions where mascons are dominated by GIS
    glacier_mask['num_insitu']  = np.array([2, 8, 10, 11, 12, 13, 14, 15, 16, 18]) # Regions where GIS contribution is determined from Zemp et al (2019)
    glacier_mask['num'] = np.sort(np.hstack([glacier_mask['num_grace'],glacier_mask['num_insitu']]))

    # Read and remove the Greenland and Antarctic Periphery
    rgi_mask = Dataset(settings['fn_rgi_mask'], 'r').variables['mask'][:]._get_data().astype(bool)
    rgi_area = Dataset(settings['fn_rgi_mask'], 'r').variables['total_area'][:]._get_data() * 1e6  # m^2
    rgi_mask = np.delete(rgi_mask, 4, axis=0)[:-1, :, :]
    rgi_area = np.delete(rgi_area, 4, axis=0)[:-1, :, :]

    # Compute mascons that contain glaciers and determine load scale:
    # Multiply scale by mass loss in GT to get local load in kg/m^2
    rgi_scale_mscn = np.zeros([rgi_mask.shape[0], len(grace_mask['lat']), len(grace_mask['lon'])])
    for reg in range(rgi_mask.shape[0]):
        print('      RGI region '+str(glacier_mask['num'][reg])+'...')
        scale_lcl = np.zeros([len(grace_mask['lat']), len(grace_mask['lon'])])
        glac_area_rgi_region = rgi_area[reg, :, :].sum()
        for k in range(len(grace_mask['mscn_coords'])):
            lat_acc = np.where((grace_mask['lat'] >= grace_mask['mscn_coords'][k, 0]) & (grace_mask['lat'] < grace_mask['mscn_coords'][k, 1]))[0]
            lon_acc = np.where((grace_mask['lon'] >= grace_mask['mscn_coords'][k, 2]) & (grace_mask['lon'] < grace_mask['mscn_coords'][k, 3]))[0]
            glac_area_in_mscn = rgi_area[reg, lat_acc[0]:lat_acc[-1] + 1, lon_acc[0]:lon_acc[-1] + 1].sum()
            area_mscn = np.maximum(1, (area[lat_acc[0]:lat_acc[-1] + 1, lon_acc[0]:lon_acc[-1] + 1] * grace_mask['mask_land'][lat_acc[0]:lat_acc[-1] + 1, lon_acc[0]:lon_acc[-1] + 1]).sum())
            scale_lcl[lat_acc[0]:lat_acc[-1] + 1, lon_acc[0]:lon_acc[-1] + 1] = (glac_area_in_mscn / glac_area_rgi_region) * (1 / area_mscn)
        rgi_scale_mscn[reg, :, :] = scale_lcl * grace_mask['mask_no_is'] * 1e12
        rgi_scale_mscn[reg, :, :] = rgi_scale_mscn[reg, :, :] * 1e12 / (rgi_scale_mscn[reg, :, :]*area).sum()
    glacier_mask['scale'] = rgi_scale_mscn
    glacier_mask['mask_all']    = (rgi_scale_mscn.sum(axis=0)>0)
    glacier_mask['mask_grace']  = (rgi_scale_mscn[np.in1d(glacier_mask['num'],glacier_mask['num_grace']),:,:].sum(axis=0)>0)
    glacier_mask['mask_insitu'] = (rgi_scale_mscn[np.in1d(glacier_mask['num'],glacier_mask['num_insitu']),:,:].sum(axis=0)>0)
    return(glacier_mask)

if __name__ == '__main__':
    main()
