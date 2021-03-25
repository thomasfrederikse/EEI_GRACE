# Save global and basin-mean time series as excel file
# Steric global
# OHC global
# OHU global annual
# Steric basin
import numpy as np
import mod_budget_grace_settings
import pandas as pd

def main():
    mod_budget_grace_settings.init()
    from mod_budget_grace_settings import settings
    global settings
    sealevel_stats = np.load(settings['fn_sealevel_stats'],allow_pickle=True).all()
    OHC_stats = np.load(settings['fn_OHC_stats'],allow_pickle=True).all()
    OHU_stats = np.load(settings['fn_OHU_stats'],allow_pickle=True).all()
    other_terms = np.load(settings['fn_other_terms'],allow_pickle=True).all()

    # Sea level
    fn = {}
    fn['global'] = settings['dir_budget'] + 'data_supplement/timeseries_global.xlsx'
    fn['basins'] = settings['dir_budget'] + 'data_supplement/timeseries_basins.xlsx'

    write_array = pd.ExcelWriter(fn['global'])
    sealevel = pd.DataFrame(data=sealevel_stats['rsl']['altimetry']['tseries'], index=settings['time'], columns=['Observed global-mean sea level [lower]','Observed global-mean sea level [mean]','Observed global-mean sea level [upper]'])
    sealevel['Hydrography (0-2000m)  [lower]'] = sealevel_stats['steric']['altimetry']['tseries'][:,0]
    sealevel['Hydrography (0-2000m) [mean]'] = sealevel_stats['steric']['altimetry']['tseries'][:,1]
    sealevel['Hydrography (0-2000m) [upper]'] = sealevel_stats['steric']['altimetry']['tseries'][:,2]
    sealevel['Ocean mass  [lower]'] = sealevel_stats['mass']['altimetry']['tseries'][:,0]
    sealevel['Ocean mass [mean]'] = sealevel_stats['mass']['altimetry']['tseries'][:,1]
    sealevel['Ocean mass [upper]'] = sealevel_stats['mass']['altimetry']['tseries'][:,2]
    sealevel['Sum of contributors [lower]'] = sealevel_stats['budget']['altimetry']['tseries'][:,0]
    sealevel['Sum of contributors [mean]'] = sealevel_stats['budget']['altimetry']['tseries'][:,1]
    sealevel['Sum of contributors [upper]'] = sealevel_stats['budget']['altimetry']['tseries'][:,2]
    sealevel['GMSL - ocean mass [lower]'] = sealevel_stats['rsl_min_mass']['altimetry']['tseries'][:,0]
    sealevel['GMSL - ocean mass [mean]'] = sealevel_stats['rsl_min_mass']['altimetry']['tseries'][:,1]
    sealevel['GMSL - ocean mass [upper]'] = sealevel_stats['rsl_min_mass']['altimetry']['tseries'][:,2]
    sealevel['GMSL - ocean mass - hydrography [lower]'] = sealevel_stats['diff']['altimetry']['tseries'][:,0]
    sealevel['GMSL - ocean mass - hydrography [mean]'] = sealevel_stats['diff']['altimetry']['tseries'][:,1]
    sealevel['GMSL - ocean mass - hydrography [upper]'] = sealevel_stats['diff']['altimetry']['tseries'][:,2]
    sealevel.to_excel(write_array, sheet_name='Sea-level budget (mm)')

    OHC = pd.DataFrame(data=OHC_stats['hydrography']['global']['tseries'], index=settings['time'], columns=['Hydrography (0-2000m) [lower]','Hydrography (0-2000m) [mean]','Hydrography (0-2000m) [upper]'])
    OHC['GMSL - ocean mass [lower]'] = OHC_stats['altmass_total']['altimetry']['tseries'][:,0]
    OHC['GMSL - ocean mass [mean]'] = OHC_stats['altmass_total']['altimetry']['tseries'][:,1]
    OHC['GMSL - ocean mass [upper]'] = OHC_stats['altmass_total']['altimetry']['tseries'][:,2]
    OHC['GMSL - ocean mass - (0-2000m) hydrography [lower]'] = OHC_stats['altmass_deep']['altimetry']['tseries'][:,0]
    OHC['GMSL - ocean mass - (0-2000m) hydrography [mean]'] = OHC_stats['altmass_deep']['altimetry']['tseries'][:,1]
    OHC['GMSL - ocean mass - (0-2000m) hydrography [upper]'] = OHC_stats['altmass_deep']['altimetry']['tseries'][:,2]
    OHC['Hydrography (0-2000m) [lower]'] = OHC_stats['hydrography']['altimetry']['tseries'][:,0]
    OHC['Hydrography (0-2000m) [mean]'] = OHC_stats['hydrography']['altimetry']['tseries'][:,1]
    OHC['Hydrography (0-2000m) [upper]'] = OHC_stats['hydrography']['altimetry']['tseries'][:,2]

    OHC.to_excel(write_array, sheet_name='Ocean heat content (J)')

    OHU = pd.DataFrame(data=OHU_stats['hydrography']['tseries'], index=OHU_stats['hydrography']['years'], columns=['Hydrography (0-2000m) [lower]','Hydrography (0-2000m) [mean]','Hydrography (0-2000m) [upper]'])
    OHU['GMSL - ocean mass [lower]'] = OHU_stats['altmass']['tseries'][:,0]
    OHU['GMSL - ocean mass [mean]'] = OHU_stats['altmass']['tseries'][:,1]
    OHU['GMSL - ocean mass [upper]'] = OHU_stats['altmass']['tseries'][:,2]
    OHU['CERES EBAF'] = other_terms['CERES']['eei']
    OHU.to_excel(write_array, sheet_name='Ocean heat uptake (W m^-2)')
    write_array.close()

    basin_name = ['Subpolar North Atlantic', 'Indian Ocean-South Pacific', 'Subtropical North Atlantic', 'East Pacific', 'South Atlantic', 'Northwest Pacific']
    write_array = pd.ExcelWriter(fn['basins'])
    for idx,name in enumerate(basin_name):
        sealevel = pd.DataFrame(data=sealevel_stats['rsl']['basin'][idx]['tseries'], index=settings['time'], columns=['Observed basin-mean sea level [lower]', 'Observed basin-mean sea level [mean]', 'Observed basin-mean sea level [upper]'])
        sealevel['Hydrography (0-2000m)  [lower]'] = sealevel_stats['steric']['basin'][idx]['tseries'][:,0]
        sealevel['Hydrography (0-2000m) [mean]'] = sealevel_stats['steric']['basin'][idx]['tseries'][:,1]
        sealevel['Hydrography (0-2000m) [upper]'] = sealevel_stats['steric']['basin'][idx]['tseries'][:,2]
        sealevel['Ocean mass  [lower]'] = sealevel_stats['mass']['basin'][idx]['tseries'][:,0]
        sealevel['Ocean mass [mean]'] = sealevel_stats['mass']['basin'][idx]['tseries'][:,1]
        sealevel['Ocean mass [upper]'] = sealevel_stats['mass']['basin'][idx]['tseries'][:,2]
        sealevel['Sum of contributors [lower]'] = sealevel_stats['budget']['basin'][idx]['tseries'][:,0]
        sealevel['Sum of contributors [mean]'] = sealevel_stats['budget']['basin'][idx]['tseries'][:,1]
        sealevel['Sum of contributors [upper]'] = sealevel_stats['budget']['basin'][idx]['tseries'][:,2]
        sealevel['GMSL - ocean mass [lower]'] = sealevel_stats['rsl_min_mass']['basin'][idx]['tseries'][:,0]
        sealevel['GMSL - ocean mass [mean]'] = sealevel_stats['rsl_min_mass']['basin'][idx]['tseries'][:,1]
        sealevel['GMSL - ocean mass [upper]'] = sealevel_stats['rsl_min_mass']['basin'][idx]['tseries'][:,2]
        sealevel['GMSL - ocean mass - hydrography [lower]'] = sealevel_stats['diff']['basin'][idx]['tseries'][:,0]
        sealevel['GMSL - ocean mass - hydrography [mean]'] = sealevel_stats['diff']['basin'][idx]['tseries'][:,1]
        sealevel['GMSL - ocean mass - hydrography [upper]'] = sealevel_stats['diff']['basin'][idx]['tseries'][:,2]
        sealevel.to_excel(write_array, sheet_name=name+' (mm)')
    write_array.close()
    return
