# -----------------------------------------------
# Compare OHU from CERES with Argo and alt-mass
# - 1. Compute monthly OHU for each component
# - 3. Estimate non-OHU EEI from GCOS
# - 2. Annual means from monthly OHU
# -----------------------------------------------
import numpy as np
import os
from netCDF4 import Dataset,num2date, date2num
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
import scipy.io as scio
import datetime as dt
def main():
    global settings
    set_settings()
    # CERES   = read_CERES()
    OHC     = np.load(settings['fn_OHC'],allow_pickle=True).all()
    OHU     = np.load(settings['fn_OHU'],allow_pickle=True).all()
    OHU_indiv = read_OHU_indiv()
    CERES = read_EEI_ceres()

    # Plot OHU
    plt.plot(settings['years'],CERES['eei'],'o-',color='C2')
    plt.plot(settings['years'],OHU['altmass']['tseries'][:,1]/2,'o-',color='C1')
    plt.plot(settings['years'],OHU['hydrography']['tseries'][:,1],'o-',color='C0')

    acc_idx = np.isfinite(OHU['altmass']['tseries'][:,1])
    pearsonr((OHU['hydrography']['tseries'][acc_idx,1][2:]),(CERES['eei'][acc_idx][2:]))
    pearsonr((OHU['altmass']['tseries'][acc_idx,1][2:]),(CERES['eei'][acc_idx][2:]))

    np.std(signal.detrend(OHU['hydrography']['tseries'][acc_idx,1][2:]))
    np.std(signal.detrend(OHU['altmass']['tseries'][acc_idx,1][2:]))
    np.std(signal.detrend(CERES['eei'][acc_idx][2:]))

    # Save
    scio.savemat('OHU.mat',OHU)
    scio.savemat('OHC.mat',OHC)

    corr_indiv = np.zeros(OHU_indiv.shape[0])
    for i in range(OHU_indiv.shape[0]):
        corr_indiv[i] = pearsonr((OHU_indiv[i,2:]), (CERES['eei'][2:]))[0]

    # Plot individual EEI
    steric_indiv = np.load(settings['fn_steric_indiv'],allow_pickle=True).all()
    for idx,prod in enumerate(steric_indiv):
        plt.plot(settings['years'],OHU_indiv[idx,:],'o-',label=prod+' '+str(corr_indiv[idx])[:4])
    plt.plot(settings['years'],CERES['eei'],'o-',color='black',label='CERES')
    plt.legend(fontsize=9,ncol=2)
    plt.ylabel('OHU (w/m$^2$)',fontsize=9)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.grid()
    plt.tight_layout()
    plt.savefig('corr_steric.pdf')

    # Plot EEI and OHU
    GCOS = read_HC_GCOS()
    plt.plot(settings['years'],CERES['eei_annual'],'o-',color='C2',label='CERES')
    plt.plot(settings['years'],ohu_argo_annual,'o-',color='C0',label='hydrography')
    plt.plot(settings['years'],ohu_altm_annual,'o-',color='C1',label='altmass')
    plt.plot(settings['years'],GCOS['hu_ice_annual'],'o-',color='C4',label='GCOS ice')
    plt.plot(settings['years'],GCOS['hu_gnd_annual'],'o-',color='C5',label='GCOS ground')
    plt.plot(settings['years'],GCOS['hu_atm_annual'],'o-',color='C6',label='GCOS atmosphere')
    plt.plot(settings['years'],GCOS['hu_atm_annual']+GCOS['hu_gnd_annual']+GCOS['hu_ice_annual'],'o-',color='C3',label='GCOS sum of non-ocean')
    plt.grid()
    plt.ylabel('Heat uptake (W/m^2)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('GCOS_comp.pdf')
    return


def set_settings():
    global settings
    settings = {}
    settings['dir_data']    = os.getenv('HOME')+'/Data/'
    settings['fn_ceres']    = settings['dir_data'] + 'Steric/CERES/CERES_EBAF-TOA_Ed4.1_Subset_200003-202004.nc'
    settings['fn_OHC']   = settings['dir_data'] + 'Budget_GRACE/stats/OHC_stats.npy'
    settings['fn_OHU']   = settings['dir_data'] + 'Budget_GRACE/stats/OHU_stats.npy'
    settings['fn_steric_indiv'] = settings['dir_data'] + 'Budget_GRACE/steric/steric_indiv.npy'
    settings['fn_ceres_eei']    = settings['dir_data'] + 'Steric/CERES/CERES_annual_eei.txt'
    settings['fn_GCOS'] = settings['dir_data'] + 'Steric/GCOS/GCOS_all_heat_content_1960-2018_ZJ_vonSchuckmann.nc'
    settings['years'] = np.arange(2003,2020)
    settings['time']  = np.arange(2003+1/24,2020+1/24,1/12)
    return

def read_OHU_indiv():
    global settings
    steric_indiv = np.load(settings['fn_steric_indiv'],allow_pickle=True).all()
    OHU_indiv = np.zeros([len(steric_indiv),len(settings['years'])])*np.nan
    for idx,prod in enumerate(steric_indiv):
        ohc_lcl = steric_indiv[prod]['altimetry']['ohc']['tseries']
        ohc_gradient = np.gradient(ohc_lcl) / (4 * np.pi * 6371000 ** 2) / (3600 * 24 * 30.5)
        for t_idx, year in enumerate(settings['years']):
            acc_idx = (settings['time'] >= year) & (settings['time'] < year + 1)
            OHU_indiv[idx,t_idx] = np.nanmean(ohc_gradient[acc_idx])
    return(OHU_indiv)

def read_EEI_ceres():
    global settings
    # --------------------------------------
    # Read CERES data and compute annual EEI
    # --------------------------------------
    CERES ={}
    file_handle = Dataset(settings['fn_ceres'], 'r')
    file_handle.set_auto_mask(False)
    ceres_time  = file_handle.variables['time'][:]
    ceres_eei  = file_handle.variables['gtoa_net_all_mon'][:]
    ceres_date = num2date(file_handle.variables['time'][:], units=file_handle.variables['time'].units)
    # Compute daily data
    daterange = np.arange('2003-01', '2020-01', dtype='datetime64[D]')
    years = daterange.astype('datetime64[Y]').astype(int) + 1970
    months = daterange.astype('datetime64[M]').astype(int) % 12 + 1
    ceres_daily = np.zeros(len(daterange)) * np.nan
    for t in range(len(ceres_date)):
        yr = ceres_date[t].year
        mn = ceres_date[t].month
        acc_idx = (years==yr) & (months==mn)
        if acc_idx.sum()>0:
            ceres_daily[acc_idx] = ceres_eei[t]

    # Average trend
    CERES['eei_trend_2005_2019'] = ceres_daily[(years>=2005)&(years<=2019)].mean()

    # Annual means
    CERES['eei'] = np.zeros(len(settings['years']))
    for idx, year in enumerate(settings['years']):
        CERES['eei'][idx] = ceres_daily[years==year].mean()
    file_handle.close()

    return(CERES)

def read_OHC_indiv():
    global settings
    EEI = np.zeros(len(settings['steric_products']),dtype='object')
    for idx,prod in enumerate(settings['steric_products']):
        print('  Processing OHC '+prod+'...')
        EEI[idx] = {}
        EEI[idx]['product'] = prod
        # Define ystart and ystop for each product
        ystart = {'EN4_l09': 2003.0, 'EN4_g10': 2003.0, 'I17': 2003.0, 'CZ16': 2003.0, 'CORA': 2003.0, 'WOA': 2005.0, 'SIO': 2005.0, 'JAMSTEC': 2005.0, 'BOA': 2005.0}
        ystop = {'EN4_l09': 2019.99, 'EN4_g10': 2019.99, 'I17': 2019.99, 'CZ16': 2019.99, 'CORA': 2019.0, 'WOA': 2019.99, 'SIO': 2019.99, 'JAMSTEC': 2019.99, 'BOA': 2019.99}

        fname = settings['fn_OHC_'+prod]
        file_handle = Dataset(fname,'r')
        file_handle.set_auto_mask(False)
        time = file_handle['t'][:]
        acc_time = (time > ystart[prod]) & (time < ystop[prod])
        time = time[acc_time]
        OHC_glb = file_handle['ts_OHC'][acc_time]
        file_handle.close()

        # OHC to EEI
        acc_time = (settings['time'] > ystart[prod]) & (settings['time'] < ystop[prod])
        OHC = np.zeros(len(settings['time']))*np.nan
        OHC[acc_time] = remove_seasonal(time, OHC_glb)
        probability = read_probability()

        EEI[idx]['eei_annual'] = np.zeros(len(settings['years']))*np.nan
        for y_idx, year in enumerate(settings['years']):
            acc_idx = (np.floor(settings['time']).astype(int) == year)
            acc_idx[np.isnan(OHC)] = False
            if np.isfinite(OHC[acc_idx]).sum()>6:
                amat = np.ones([acc_idx.sum(), 2])
                amat[:,1] = settings['time'][acc_idx]
                amat_T = amat.T
                amat_sq = np.linalg.inv(np.dot(amat_T, amat))
                EEI[idx]['eei_annual'][y_idx] = np.dot(amat_sq, np.dot(amat_T, OHC[acc_idx]))[1]/(24*365.25*3600)/(4*np.pi*6371000**2)

        # Correlation with CERES
        corr_ceres = np.zeros(len(EEI))
        for i in range(len(EEI)):
            corr_ceres[i] = pearsonr(signal.detrend(EEI[i]['eei_annual'][2:-1]), signal.detrend(CERES['eei_annual'][2:-1]))[0]  # Correlation = 0.84
    return(OHC)


def read_HC_GCOS():
    # ----------------------
    # Read GCOS heat content
    # ----------------------
    global settings
    GCOS = {}
    fh = Dataset(settings['fn_GCOS'],'r')
    fh.set_auto_mask(False)
    time = 1950+fh.variables['time'][:]/365.25
    hc_ice    = fh.variables['energy_cryosphere'][:]
    hc_ground = fh.variables['ground_heat_content'][:]
    hc_atm    = fh.variables['atmospheric_heat_content'][:]
    hc_deep    = fh.variables['ohc_below_2000m'][:]
    fh.close()
    hc_ice_interp = interp1d(time[:-2], hc_ice[:-2],kind='linear',fill_value='extrapolate')(settings['time'])
    hc_gnd_interp = interp1d(time, hc_ground,kind='linear',fill_value='extrapolate')(1.0*settings['time'])
    hc_atm_interp = interp1d(time, hc_atm,kind='linear',fill_value='extrapolate')(settings['time'])
    hc_deep = interp1d(time, hc_deep,kind='linear',fill_value='extrapolate')(settings['time'])
    hc_non_oc = hc_ice_interp + hc_gnd_interp + hc_atm_interp
    hc_non_oc-=hc_non_oc.mean()
    hc_deep-=hc_deep.mean()
    GCOS['hc_non_ocean'] = hc_non_oc
    GCOS['hc_deep'] = hc_deep

    # EEI using OHU approach
    time = settings['time'][24:192]
    hu_ice = np.gradient(hc_ice_interp)[24:192]/(4*np.pi*6371000**2)/(3600*24*30.5)
    hu_gnd = np.gradient(hc_gnd_interp)[24:192]/(4*np.pi*6371000**2)/(3600*24*30.5)
    hu_atm = np.gradient(hc_atm_interp)[24:192]/(4*np.pi*6371000**2)/(3600*24*30.5)

    GCOS['hu_ice_annual'] = np.zeros(len(settings['years'])) * np.nan
    GCOS['hu_gnd_annual'] = np.zeros(len(settings['years'])) * np.nan
    GCOS['hu_atm_annual'] = np.zeros(len(settings['years'])) * np.nan

    for idx, year in enumerate(settings['years']):
        acc_idx = (time>=year) & (time<year+1)
        GCOS['hu_ice_annual'][idx] = np.nanmean(hu_ice[acc_idx])
        GCOS['hu_gnd_annual'][idx] = np.nanmean(hu_gnd[acc_idx])
        GCOS['hu_atm_annual'][idx] = np.nanmean(hu_atm[acc_idx])
    return(GCOS)

def annual_mean(tseries):
    global settings
    # --------------------------------------------------
    # Compute annual mean from monthly data
    # 1. Interpolate on our grid
    # 2. Apply 24-month lowpass filter to avoid aliasing
    # 3. Remove interpolated numbers
    # 4. Compute annual means
    # --------------------------------------------------

    # Interpolate monthly time series
    acc_idx = np.isfinite(tseries)
    tseries_interp = interp1d(settings['time'][acc_idx], tseries[acc_idx],kind='linear',fill_value='extrapolate')(1.0*settings['time'])
    # Low pass filter
    b,a = signal.butter(2, 2/24,btype='lowpass')
    tseries_lpf = signal.filtfilt(b,a,tseries_interp,padtype=None,method='gust')
    tseries_lpf[~acc_idx] = np.nan
    # Compute annual means from low-pass filtered time series
    tseries_annual = np.zeros(len(settings['years']))*np.nan
    for idx, year in enumerate(settings['years']):
        acc_idx = (settings['time']>=year) & (settings['time']<year+1)
        if np.isfinite(tseries[acc_idx]).sum() > 5: # Require at least 5 valid monthly data to compute annual mean
            tseries_annual[idx] = np.nanmean(tseries_lpf[acc_idx])
    return(tseries_annual)

def remove_seasonal(time, tseries):
    amat = np.ones([len(time), 6])
    amat[:, 0] = np.sin(2 * np.pi * time)
    amat[:, 1] = np.cos(2 * np.pi * time)
    amat[:, 2] = np.sin(4 * np.pi * time)
    amat[:, 3] = np.cos(4 * np.pi * time)
    amat[:, -1] = time - np.mean(time)
    sol = np.linalg.lstsq(amat, tseries, rcond=None)[0]
    sol[-1] = 0
    tseries_noseas = tseries - np.matmul(amat, sol)
    tseries_noseas = tseries_noseas - np.mean(tseries_noseas[:12])
    return (tseries_noseas)

def read_probability():
    global settings
    probability = Dataset(settings['fn_gia_ens_rad'],'r').variables['probability'][settings['ens_range']]._get_data()
    probability /= probability.sum()
    return(probability)
