# Save all the global trends to an Excel file

import pandas as pd
import mod_budget_grace_settings
mod_budget_grace_settings.init()
import numpy as np

def main():
    global settings
    with pd.ExcelWriter(settings['dir_budget']+'data_supplement/Data/trends_global.xlsx') as writer:
        for era in settings['trend_eras']:
            pname = str(era[0])+'-'+str(era[1])
            result = sl_table(pname)
            result.to_excel(writer,sheet_name=pname)
    return

def sl_table(period):
    global settings
    sealevel_stats = np.load(settings['fn_sealevel_stats'], allow_pickle=True).all()
    OHC_stats = np.load(settings['fn_OHC_stats'], allow_pickle=True).all()
    EEI_stats = np.load(settings['fn_EEI_stats'], allow_pickle=True).all()
    corr_gia_stats = np.load(settings['fn_corr_gia_stats'], allow_pickle=True).all()
    corr_grd_stats = np.load(settings['fn_corr_grd_stats'], allow_pickle=True).all()
    other_terms = np.load(settings['fn_other_terms'],allow_pickle=True).all()

    # Sea level table
    sl_table = pd.DataFrame(columns=['Sea level (mm yr-1) 5th','Sea level (mm yr-1) mean','Sea level (mm yr-1) 95th'])
    sl_table.loc['Global-mean sea level'] = sealevel_stats["rsl"]['altimetry']['trend'][period]
    sl_table.loc['Global-mean geocentric sea level'] = sealevel_stats["gsl"]['altimetry']['trend'][period]
    sl_table.loc['Correction for GIA'] = -corr_gia_stats['altimetry']['trend'][period]
    sl_table.loc['Correction for contemporary GRD'] = -corr_grd_stats['altimetry']['trend'][period]
    sl_table.loc['Barystatic sea level'] = sealevel_stats["mass"]['altimetry']['trend'][period]
    sl_table.loc['Glaciers'] = -sealevel_stats["mass_ctb"]['mass_glac']['trend'][period]
    sl_table.loc['Greenland Ice Sheet'] = -sealevel_stats["mass_ctb"]['mass_GrIS']['trend'][period]
    sl_table.loc['Antarctic Ice Sheet'] = -sealevel_stats["mass_ctb"]['mass_AIS']['trend'][period]
    sl_table.loc['Terrestrial Water Storage'] = -sealevel_stats["mass_ctb"]['mass_tws']['trend'][period]
    sl_table.loc['Hydrographic observations 0-2000m'] = sealevel_stats["steric"]['altimetry']['trend'][period]
    sl_table.loc['Barystatic + hydrography'] = sealevel_stats["budget"]['altimetry']['trend'][period]
    sl_table.loc['GMSL - barystatic'] = sealevel_stats["rsl_min_mass"]['altimetry']['trend'][period]
    sl_table.loc['GMSL - barystatic - hydrography'] = sealevel_stats["diff"]['altimetry']['trend'][period]
    sl_table.loc['Deep ocean below 2000m'] = other_terms['deep_ocean']['steric_trend']

    # OHC table
    ohc_table = pd.DataFrame(columns=['OHC (ZJ yr-1) 5th','OHC (ZJ yr-1) mean','OHC (ZJ yr-1) 95th'])
    ohc_table.loc['Hydrographic observations 0-2000m'] = OHC_stats["hydrography"]['altimetry']['trend'][period]/1e21
    ohc_table.loc['GMSL - barystatic']                 = OHC_stats["altmass_total"]['altimetry']['trend'][period]/1e21
    ohc_table.loc['GMSL - barystatic - hydrography']   = OHC_stats["altmass_deep"]['altimetry']['trend'][period]/1e21
    ohc_table.loc['Deep ocean below 2000m'] = other_terms['deep_ocean']['ohc_trend']/1e21
    ohc_table.loc['Non-ocean terms'] = other_terms['non_ocean']['trend'][period]/1e21
    ohc_table_wm2 = ohc_table*1e21/3600/24/365/4/np.pi/6371000**2
    ohc_table_wm2.columns = ['OHC (W m-2) 5th','OHC (W m-2) mean','OHC (W m-2) 95th']

    # EEI table
    eei_table = pd.DataFrame(columns=['EEI (W m-2) 5th','EEI (W m-2) mean','EEI (W m-2) 95th'])
    eei_table.loc['Hydrographic observations 0-2000m'] = EEI_stats['hydrography']['trend'][period]
    eei_table.loc['GMSL - barystatic'] = EEI_stats['altmass']['trend'][period]
    eei_table.loc['CERES'] = [np.nan, other_terms['CERES']['eei_mean'][period], np.nan]

    result = pd.concat([sl_table, ohc_table,ohc_table_wm2,eei_table], axis=1)
    return(result)