# ------------------------------------------------------
# Read  expansion efficieny from ECCO/ORAS
# and save for GMT
# ------------------------------------------------------
from netCDF4 import Dataset
import numpy as np
import os
import mod_gentools as gentools

def main():
    settings = {}
    settings['dir_data'] = os.getenv('HOME') + '/Data/'
    settings['dir_gmt'] = os.getenv('HOME') + '/Scripts/GMT/Papers/GRACE_budget/reanalysis_steric_ohc/'
    settings['dir_oras']     = settings['dir_data'] + 'OceanModels/ORAS5/'
    settings['dir_ecco']     = settings['dir_data'] + 'ECCO/v4r4/'

    settings['fn_oras_shallow'] = settings['dir_oras'] + 'ORAS_steric_ohc_shallow.nc'
    settings['fn_oras_deep']    = settings['dir_oras'] + 'ORAS_steric_ohc_deep.nc'
    settings['fn_oras_full']    = settings['dir_oras'] + 'ORAS_steric_ohc_full.nc'

    settings['fn_ecco_shallow'] = settings['dir_ecco'] + 'ECCOv4r4_steric_ohc_shallow_1993_2017.nc'
    settings['fn_ecco_deep']    = settings['dir_ecco'] + 'ECCOv4r4_steric_ohc_deep_1993_2017.nc'
    settings['fn_ecco_full']    = settings['dir_ecco'] + 'ECCOv4r4_steric_ohc_full_1993_2017.nc'
    settings['fn_steric_indiv'] = settings['dir_data'] + 'Budget_GRACE/steric/steric_indiv.npy'
    settings['fn_ohc_stats'] = settings['dir_data'] + 'Budget_GRACE/stats/OHC_stats.npy'
    settings['fn_sealevel_stats'] = settings['dir_data'] + 'Budget_GRACE/stats/sealevel_stats.npy'

    types = ['deep', 'shallow', 'full']
    model = 'ecco'
    time = Dataset(settings['fn_' + model + '_full'])['time'][:]._get_data()
    for tp in types:
        print(model,tp)
        ohc = Dataset(settings['fn_' + model + '_' + tp])['ohc_ts'][:]._get_data()
        steric = Dataset(settings['fn_' + model + '_' + tp])['thermosteric_ts'][:]._get_data()
        ohc -= ohc[-12:].mean()
        steric -= steric[-12:].mean()
        gmt_save_tseries(settings['dir_gmt'], model + '_steric_'+tp, time, steric)
        gmt_save_tseries(settings['dir_gmt'], model + '_ohc_'+tp, time, ohc / 1e21)
    model = 'oras'
    time = Dataset(settings['fn_' + model + '_full'])['time'][:]._get_data()
    for tp in types:
        print(model,tp)
        ohc = gentools.remove_seasonal(time,Dataset(settings['fn_' + model + '_' + tp])['ohc_ts'][:]._get_data())
        steric = gentools.remove_seasonal(time,1000*Dataset(settings['fn_' + model + '_' + tp])['thermosteric_ts'][:]._get_data())
        ohc -= ohc[-24:-12].mean()
        steric -= steric[-24:-12].mean()

        gmt_save_tseries(settings['dir_gmt'], model + '_steric_'+tp, time, steric)
        gmt_save_tseries(settings['dir_gmt'], model + '_ohc_'+tp, time, ohc / 1e21)

    steric_indiv = np.load(settings['fn_steric_indiv'],allow_pickle=True).all()
    ts_indiv = np.arange(2003+1/24,2020+1/24,1/12)
    products = ['thermosteric', 'ohc']
    for idx,prod in enumerate(steric_indiv):
        for stype in products:
            mfac = 1.0
            if stype=='ohc': mfac = 1e-21
            gmt_save_tseries(settings['dir_gmt'],prod+'_global_'+stype, ts_indiv, mfac*(steric_indiv[prod]['global'][stype]['tseries']-steric_indiv[prod]['global'][stype]['tseries'][-24:-12].mean()))

    # Geodetic approach
    sealevel_stats = np.load(settings['fn_sealevel_stats'],allow_pickle=True).all()
    steric_geo = sealevel_stats['rsl_min_mass']['altimetry']['tseries'][:, 1]
    steric_geo -= np.nanmean(steric_geo[-24:-12])
    gmt_save_tseries(settings['dir_gmt'],'steric_geo',ts_indiv,steric_geo)

    ohc_stats = np.load(settings['fn_ohc_stats'],allow_pickle=True).all()
    ohc_geo = ohc_stats['altmass_total']['altimetry']['tseries'][:, 1] / 1e21
    ohc_geo -= np.nanmean(ohc_geo[-24:-12])
    gmt_save_tseries(settings['dir_gmt'],'ohc_geo',ts_indiv,ohc_geo)
    return


def gmt_save_tseries(gmt_dir, fname, time, tseries):
    save_array = (np.array([time, tseries])).T
    np.savetxt(gmt_dir + fname + '_ts.txt', save_array, fmt='%4.5f;%4.5f')
    return





