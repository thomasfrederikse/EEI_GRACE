import numpy as np
import mod_budget_grace_settings
from netCDF4 import Dataset

def main():
    mod_budget_grace_settings.init()
    from mod_budget_grace_settings import settings
    global settings

    steric = np.load(settings['fn_steric_indiv'],allow_pickle=True).all()
    ohu_annual = compute_ohu_annual(steric)
    products = ['thermosteric', 'halosteric', 'ohc']
    years = np.unique(np.floor(settings['time']))
    other_terms = np.load(settings['fn_other_terms'],allow_pickle=True).all()
    # Steric and OHC
    for idx,prod in enumerate(steric):
        for reg in steric[prod]:
            for stype in products:
                mfac = 1.0
                if stype=='ohc': mfac = 1e-21
                gmt_save_tseries(settings['dir_gmt']+'steric_ts/',prod+'_'+reg+'_'+stype, settings['time'], mfac*(steric[prod][reg][stype]['tseries']-steric[prod][reg][stype]['tseries'][-12:].mean()))
                if reg == 'global': gmt_save_trend(settings['dir_gmt']+'steric_ts/',prod+'_'+reg+'_'+stype, mfac*steric[prod][reg][stype]['trend']['2005-2019'], idx+0.675)
                if reg == 'altimetry': gmt_save_trend(settings['dir_gmt']+'steric_ts/',prod+'_'+reg+'_'+stype, mfac*steric[prod][reg][stype]['trend']['2005-2019'], idx+0.225)
        # OHU
        gmt_save_tseries(settings['dir_gmt'] + 'steric_ts/', prod + '_OHU_annual', years, ohu_annual[idx,:])
        corr_indiv = np.corrcoef((ohu_annual[idx,2:]), (other_terms['CERES']['eei'][2:]))[0,1]
        gmt_save_trend(settings['dir_gmt'] + 'steric_ts/', prod + '_OHU', corr_indiv, idx)

    trend = 0
    for prod in steric:
        print(prod + ' ' + str(steric[prod]['altimetry']['ohc']['trend']['2005-2019']))
        trend+= steric[prod]['altimetry']['ohc']['trend']['2005-2019']/len(steric)
    return

def compute_ohu_annual(steric):
    global settings
    years = np.unique(np.floor(settings['time']))
    ohu_annual = np.zeros([len(steric),len(years)])*np.nan
    for idx,prod in enumerate(steric):
        ohc_lcl = steric[prod]['altimetry']['ohc']['tseries']
        ohc_gradient = np.gradient(ohc_lcl) / (4 * np.pi * 6371000 ** 2) / (3600 * 24 * 30.5)
        for t_idx, year in enumerate(years):
            acc_idx = (settings['time'] >= year) & (settings['time'] < year + 1)
            ohu_annual[idx,t_idx] = np.nanmean(ohc_gradient[acc_idx])
    return(ohu_annual)

def gmt_save_tseries(gmt_dir, fname, time, tseries):
    # Save mean and confidence intervals
    save_array = (np.array([time, tseries])).T
    np.savetxt(gmt_dir + fname + '_ts.txt', save_array, fmt='%4.3f;%4.3f')
    return

def gmt_save_trend(gmt_dir, fname, trend, t_idx):
    save_array = np.array([t_idx,trend])
    np.savetxt(gmt_dir + fname + '_trend.txt', save_array[np.newaxis,:], fmt='%4.3f;%4.3f')
    return