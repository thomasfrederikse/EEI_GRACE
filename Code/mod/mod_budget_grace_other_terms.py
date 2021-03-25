# --------------------------------------
# Set and read other EEI/sea level terms
# Deep-ocean steric sea level/OHC
# Non-ocean EEI
# --------------------------------------
import numpy as np
from netCDF4 import Dataset, num2date
import mod_gentools as gentools
from scipy.interpolate import interp1d
def main():
    print('Reading other EEI/sea level terms...')
    from mod_budget_grace_settings import settings
    global settings,dt_start
    deep_ocean = def_deep_steric_ohc()
    non_ocean  = read_non_ocean_ohc()
    CERES      = read_CERES()
    # Save data
    other_terms = {}
    other_terms['deep_ocean'] = deep_ocean
    other_terms['non_ocean'] = non_ocean
    other_terms['CERES'] = CERES
    np.save(settings['fn_other_terms'],other_terms)
    print('Done')
    return

def def_deep_steric_ohc():
    global settings
    # Deep (<2000m) steric and OHC
    # From Desbruyeres et al. 2016
    deep_ocean = {}

    # OHC (From Terrawatt to Joules
    ohc_mean  = 33*1e12*3600*24*365.25
    ohc_sterr = 21*1e12*3600*24*365.25/2

    deep_ocean['ohc_trend']    = np.array([ohc_mean-1.65*ohc_sterr,ohc_mean,ohc_mean+1.65*ohc_sterr])
    deep_ocean['ohc_ts_ens']   = np.random.normal(loc=ohc_mean,scale=ohc_sterr,size=settings['num_ens'])[:,np.newaxis] * (settings['time']-settings['time'][-12:].mean())[np.newaxis,:]
    deep_ocean['ohc_ts']   = np.percentile(deep_ocean['ohc_ts_ens'],[5,50,95],axis=0).T

    # Steric
    steric_mean  = 0.117
    steric_sterr = 0.073/2

    deep_ocean['steric_trend']    = np.array([steric_mean-1.65*steric_sterr,steric_mean,steric_mean+1.65*steric_sterr])
    deep_ocean['steric_ts_ens']   = np.random.normal(loc=steric_mean,scale=steric_sterr,size=settings['num_ens'])[:,np.newaxis] * (settings['time']-settings['time'][-12:].mean())[np.newaxis,:]
    deep_ocean['steric_ts']   = np.percentile(deep_ocean['steric_ts_ens'],[5,50,95],axis=0).T
    return(deep_ocean)

def read_non_ocean_ohc():
    # Read Von Schuckman et al. 2020
    non_ocean = {}
    fh = Dataset(settings['fn_EEI_GCOS'],'r')
    fh.set_auto_mask(False)
    time = fh['time'][:]/365.24 + 1950

    # Ground
    hu_ground = np.gradient(fh['ground_heat_content'][:])*1e21
    hu_ground_ci = 0.028*3600*24*365*4*np.pi*6371000**2*(1-0.71)/2

    hc_ground_ens = interp1d(time,np.cumsum(hu_ground[np.newaxis,:]+np.random.normal(loc=0,scale=hu_ground_ci,size=settings['num_ens'])[:,np.newaxis],axis=1),kind='linear',fill_value='extrapolate')(settings['time'])
    hc_ground_ens-=hc_ground_ens[:,-12:].mean(axis=1)[:,np.newaxis]

    # Atmosphere
    hc_atm     = (fh['atmospheric_heat_content'][:])*1e21
    hc_atm_unc = (fh['atmospheric_heat_content_uncertainty'][:])*1e21/2
    hc_atm_ens = interp1d(time,hc_atm[np.newaxis,:]+hc_atm_unc*np.random.normal(loc=0,scale=1,size=settings['num_ens'])[:,np.newaxis],kind='linear',fill_value='extrapolate')(settings['time'])
    hc_atm_ens-=hc_atm_ens[:,-12:].mean(axis=1)[:,np.newaxis]

    # Ice
    hc_ice     = (fh['energy_cryosphere'][:-1])*1e21
    hc_ice_unc = (fh['energy_cryosphere_uncertainty'][:-1])*1e21/2
    hc_ice_ens = interp1d(time[:-1],hc_ice[np.newaxis,:]+hc_ice_unc*np.random.normal(loc=0,scale=1,size=settings['num_ens'])[:,np.newaxis],kind='linear',fill_value='extrapolate')(settings['time'])
    hc_ice_ens-=hc_ice_ens[:,-12:].mean(axis=1)[:,np.newaxis]

    non_ocean['ts_ens'] = hc_ice_ens + hc_atm_ens + hc_ground_ens

    # Trend ensemble
    non_ocean['trend'] = {}
    for period in settings['trend_eras']:
        t_acc = (settings['time']>period[0]) & (settings['time']<period[1]+1)
        ens_era = non_ocean['ts_ens'][:,t_acc]
        ens_trend_era = np.zeros(ens_era.shape[0])

        # Design matrix
        amat = np.ones([len(settings['time'][t_acc]), 3])
        amat[:, 1] = settings['time'][t_acc] - settings['time'][t_acc].mean()
        amat[:, 2] = 0.5 * (settings['time'][t_acc] - settings['time'][t_acc].mean()) ** 2
        amat_T = amat.T
        amat_sq = np.linalg.inv(np.dot(amat_T, amat))
        # Loop over all ensemble members
        for i in range(ens_era.shape[0]):
            sol = np.dot(amat_sq, np.dot(amat_T, ens_era[i, :]))
            ens_trend_era[i] = sol[1]
            non_ocean['trend'][str(period[0])+'-'+str(period[1])] = np.percentile(ens_trend_era,[5,50,95])
    return(non_ocean)

def read_CERES():
    global settings
    # --------------------------------------
    # Read CERES data and compute annual EEI
    # --------------------------------------
    CERES ={}
    file_handle = Dataset(settings['fn_ceres'], 'r')
    file_handle.set_auto_mask(False)
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

    # Annual means
    CERES['years'] = np.sort(np.unique(np.floor(settings['time'])))
    CERES['eei'] = np.zeros(len(CERES['years']))
    for idx, year in enumerate(CERES['years']):
        CERES['eei'][idx] = ceres_daily[years==year].mean()

    # EEI trend
    CERES['trend'] = {}
    for period in settings['trend_eras']:
        t_acc = (CERES['years']>period[0]-0.5) & (CERES['years']<period[1]+0.5)
        CERES['trend'][str(period[0])+'-'+str(period[1])] = gentools.lsqtrend(CERES['years'][t_acc],CERES['eei'][t_acc])
    file_handle.close()

    # CERES mean
    CERES['eei_mean'] = {}
    for era in settings['trend_eras']:
        tname = str(era[0])+'-'+str(era[1])
        t_acc = (years >= era[0]) & (years <= era[1])
        CERES['eei_mean'][tname] = ceres_daily[t_acc].mean()
    return(CERES)