# -------------------------------------
# Mascon conversion/regridding routines
# -------------------------------------
import numpy as np
def grid2mascon_3d(lat,lon,grid,settings):
    print('   Transforming field into mascons...')
    coords = np.load(settings['fn_mscn_coords'])
    mscn = np.zeros([coords.shape[0],grid.shape[0]],dtype=np.float32)
    for k in range(len(coords)):
        lat_acc = np.where((lat >= coords[k,0]) & (lat < coords[k,1]))[0]
        lon_acc = np.where((lon >= coords[k,2]) & (lon < coords[k,3]))[0]
        weight = np.cos(np.deg2rad(lat[lat_acc])) / np.mean(np.cos(np.deg2rad(lat[lat_acc])))  # Weight by cos lat
        mscn[k,:] = np.nanmean(weight[:,np.newaxis]*grid[:,lat_acc[0]:lat_acc[-1]+1,lon_acc[0]:lon_acc[-1] + 1],axis=(1,2))
    return(mscn)

def mascon2grid_3d(mascon,settings):
    grid = np.zeros([mascon.shape[1],len(settings['lat']),len(settings['lon'])],dtype=np.float32)
    coords = np.load(settings['fn_mscn_coords'])
    for k in range(len(coords)):
        lat_acc = np.where((settings['lat'] >= coords[k,0]) & (settings['lat'] < coords[k,1]))[0]
        lon_acc = np.where((settings['lon'] >= coords[k,2]) & (settings['lon'] < coords[k,3]))[0]
        grid[:,lat_acc[0]:lat_acc[-1]+1,lon_acc[0]:lon_acc[-1] + 1] = mascon[k,:][:,np.newaxis,np.newaxis]
    return(grid)

def masconize_gia_2d(lat,lon,field,settings):
    coords     = np.load(settings['dir_data'] + 'GRACE/JPL_mascon/mascon_coords.npy')
    field_mscn = np.zeros([len(settings['lat']), len(settings['lon'])],dtype=np.float32)
    for k in range(len(coords)):
        lat_acc = np.where((lat >= coords[k,0]) & (lat < coords[k,1]))[0]
        lon_acc = np.where((lon >= coords[k,2]) & (lon < coords[k,3]))[0]
        weight = np.cos(np.deg2rad(lat[lat_acc])) / np.mean(np.cos(np.deg2rad(lat[lat_acc])))  # Weight by cos lat
        field_mscn[lat_acc[0]:lat_acc[-1]+1,lon_acc[0]:lon_acc[-1]+1] = np.nanmean(weight[:,np.newaxis] * field[lat_acc[0]:lat_acc[-1]+1,lon_acc[0]:lon_acc[-1]+1])
    return(field_mscn)

def masconize_2d(field,mask,settings):
    coords     = np.load(settings['dir_data'] + 'GRACE/JPL_mascon/mascon_coords.npy')
    field_mscn = np.zeros([len(settings['lat']), len(settings['lon'])],dtype=np.float32) * np.nan
    area = mask['area'] * (~mask['land'])
    for k in range(len(coords)):
        lat_acc = np.where((settings['lat'] >= coords[k,0]) & (settings['lat'] < coords[k,1]))[0]
        lon_acc = np.where((settings['lon'] >= coords[k,2]) & (settings['lon'] < coords[k,3]))[0]
        if lat_acc.shape[0] > 0:
            acc_mask  = (~mask['land'])[lat_acc[0]:lat_acc[-1]+1,lon_acc[0]:lon_acc[-1]+1]
            if np.sum(acc_mask)>0:
                area_lcl = area[lat_acc[0]:lat_acc[-1]+1,lon_acc[0]:lon_acc[-1]+1]
                area_lcl = area_lcl / area_lcl.sum()
                field_mscn[lat_acc[0]:lat_acc[-1] + 1, lon_acc[0]:lon_acc[-1] + 1] = np.nansum(area_lcl * field[lat_acc[0]:lat_acc[-1] + 1, lon_acc[0]:lon_acc[-1] + 1])
    field_mscn[mask['land']] = np.nan
    field_mscn[np.isnan(field)] = np.nan
    return(field_mscn)

def masconize_3d(grid,mask,settings):
    # Mask field points that are not in mask
    coords    = np.load(settings['dir_data'] + 'GRACE/JPL_mascon/mascon_coords.npy')
    grid_mscn = np.zeros(grid.shape,dtype=np.float32)*np.nan
    # Average a grid into mascons per mascon
    for k in range(len(coords)):
        lat_acc = np.where((settings['lat'] >= coords[k,0]) & (settings['lat'] < coords[k,1]))[0]
        lon_acc = np.where((settings['lon'] >= coords[k,2]) & (settings['lon'] < coords[k,3]))[0]
        if lat_acc.shape[0] > 0:
            area_lcl = mask['area'][lat_acc[0]:lat_acc[-1]+1,lon_acc[0]:lon_acc[-1]+1]
            area_lcl = area_lcl / area_lcl.sum()
            grid_mscn[:,lat_acc[0]:lat_acc[-1]+1,lon_acc[0]:lon_acc[-1]+1]=np.nansum(area_lcl[np.newaxis,...]*grid[:,lat_acc[0]:lat_acc[-1]+1,lon_acc[0]:lon_acc[-1]+1],axis=(1,2))[:,np.newaxis,np.newaxis]
    return(grid_mscn)

def masconize_sealevel_3d(rsl,mask,settings):
    # Mask field points that are not in mask
    coords     = np.load(settings['dir_data'] + 'GRACE/JPL_mascon/mascon_coords.npy')
    rsl_mscn = np.zeros(rsl.shape,dtype=np.float32)*np.nan
    area_alt = mask['area'] * mask['slm']
    # Average a grid into mascons per mascon
    for k in range(len(coords)):
        lat_acc = np.where((settings['lat'] >= coords[k,0]) & (settings['lat'] < coords[k,1]))[0]
        lon_acc = np.where((settings['lon'] >= coords[k,2]) & (settings['lon'] < coords[k,3]))[0]
        if lat_acc.shape[0] > 0:
            acc_mask  = mask['slm'][lat_acc[0]:lat_acc[-1]+1,lon_acc[0]:lon_acc[-1]+1]
            if np.sum(acc_mask)>0:
                area_lcl = area_alt[lat_acc[0]:lat_acc[-1]+1,lon_acc[0]:lon_acc[-1]+1]
                area_lcl = area_lcl / area_lcl.sum()
                rsl_mscn[:,lat_acc[0]:lat_acc[-1]+1,lon_acc[0]:lon_acc[-1]+1]=np.nansum(area_lcl[np.newaxis,...]*rsl[:,lat_acc[0]:lat_acc[-1]+1,lon_acc[0]:lon_acc[-1]+1],axis=(1,2))[:,np.newaxis,np.newaxis]
    # Set points outside GRACE SLM to nan
    rsl_mscn[:,~mask['slm']] = np.nan
    rsl_mscn[np.isnan(rsl)] = np.nan
    return(rsl_mscn)

def masconize_regrid_3d(lat,lon,field,area,mask_in,mask_out,settings):
    print('   Transforming field into mascons...')
    # Mask field points that are not in mask
    field[:,mask_in==False] = np.nan
    coords     = np.load(settings['dir_data'] + 'GRACE/JPL_mascon/mascon_coords.npy')
    field_mscn = np.zeros([settings['ntime'],len(settings['lat']), len(settings['lon'])],dtype=np.float32)*np.nan
    # Average a grid into mascons per mascon
    for k in range(len(coords)):
        lat_in_acc = np.where((lat >= coords[k,0]) & (lat < coords[k,1]))[0]
        lon_in_acc = np.where((lon >= coords[k,2]) & (lon < coords[k,3]))[0]
        lat_out_acc = np.where((settings['lat'] >= coords[k,0]) & (settings['lat'] < coords[k,1]))[0]
        lon_out_acc = np.where((settings['lon'] >= coords[k,2]) & (settings['lon'] < coords[k,3]))[0]
        if lat_in_acc.shape[0] > 0:
            acc_mask_in  = mask_in[lat_in_acc[0]:lat_in_acc[-1]+1,lon_in_acc[0]:lon_in_acc[-1]+1]
            acc_mask_out = mask_out[lat_out_acc[0]:lat_out_acc[-1]+1,lon_out_acc[0]:lon_out_acc[-1]+1]
            if (np.sum(acc_mask_in)>0) & (np.sum(acc_mask_out)>0):
                area_lcl = area[lat_in_acc[0]:lat_in_acc[-1]+1,lon_in_acc[0]:lon_in_acc[-1]+1] * acc_mask_in
                area_lcl = area_lcl / np.sum(area_lcl)
                field_mscn[:,lat_out_acc[0]:lat_out_acc[-1]+1,lon_out_acc[0]:lon_out_acc[-1]+1] = np.nansum(area_lcl[np.newaxis,...] * field[:,lat_in_acc[0]:lat_in_acc[-1]+1,lon_in_acc[0]:lon_in_acc[-1]+1], axis=(1,2))[:,np.newaxis,np.newaxis]
    # Set points outside GRACE SLM to nan
    mask_grace = mask_out.astype(np.float32)
    mask_grace[mask_grace==0] = np.nan
    field_mscn = field_mscn * mask_grace[np.newaxis,:,:]
    return(field_mscn)
