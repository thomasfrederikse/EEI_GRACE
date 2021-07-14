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
    settings['dir_gmt'] = os.getenv('HOME') + '/Scripts/GMT/Papers/GRACE_budget/eff_ts/'
    settings['fn_efficiency']       = settings['dir_data'] + 'Budget_GRACE/efficiency/efficiency_combined.npy'



    efficiency = np.load(settings['fn_efficiency'],allow_pickle=True).all()

    for model in efficiency['obs']['models']:
        gmt_save_tseries(settings['dir_gmt'],model,efficiency['obs']['ts_glb'][model]['time']+7.5,efficiency['obs']['ts_glb'][model]['eff']*1e21)

    #WOA
    gmt_save_tseries(settings['dir_gmt'],'WOA',efficiency['obs']['ts_glb']['I17']['time'][-2:]+7.5,efficiency['obs']['trend_glb'][efficiency['obs']['models'].index('WOA')]*np.ones(2)*1e21)


    gmt_save_tseries(settings['dir_gmt'],'ORAS',efficiency['oras5']['shallow']['time'][:-180]+7.5,efficiency['oras5']['shallow']['alt']['eff_rolling']*1e21)
    gmt_save_tseries(settings['dir_gmt'],'ECCO',efficiency['ecco']['shallow']['time'][:-180]+7.5,efficiency['ecco']['shallow']['alt']['eff_rolling']*1e21)



def gmt_save_tseries(gmt_dir, fname, time, tseries):
    save_array = (np.array([time, tseries])).T
    np.savetxt(gmt_dir + fname + '_ts.txt', save_array, fmt='%4.5f;%4.5f')
    return
