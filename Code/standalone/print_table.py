# -----------------------------------------
# Print the LaTeX table with global numbers
# -----------------------------------------
import numpy as np
import mod_budget_grace_settings
import mod_gentools as gentools
from netCDF4 import Dataset
mod_budget_grace_settings.init()
from mod_budget_grace_settings import settings

def main():
    global settings
    sealevel_stats = np.load(settings['fn_sealevel_stats'], allow_pickle=True).all()
    OHC_stats = np.load(settings['fn_OHC_stats'], allow_pickle=True).all()
    EEI_stats = np.load(settings['fn_EEI_stats'], allow_pickle=True).all()
    corr_gia_stats = np.load(settings['fn_corr_gia_stats'], allow_pickle=True).all()
    corr_grd_stats = np.load(settings['fn_corr_grd_stats'], allow_pickle=True).all()
    other_terms = np.load(settings['fn_other_terms'],allow_pickle=True).all()
    period = '2005-2019'
    print_table = []
    # Rij 1 GMSL
    print_table.append('Global-mean sea level &'+ print_sl_trend('rsl','altimetry',sealevel_stats,period) + '& - & & - & & - & \\\\')
    print_table.append('Global-mean geocentric sea level &'+ print_sl_trend('gsl','altimetry',sealevel_stats,period) + '& - & & - & & - & \\\\')
    print_table.append('Barystatic sea level &'+ print_sl_trend('mass','altimetry',sealevel_stats,period) + '& - & & - & & - & \\\\')
    print_table.append('Glaciers &'+ print_sl_trend('mass_ctb','mass_glac',sealevel_stats,period) + '& - & & - & & - & \\\\')
    print_table.append('Greenland Ice Sheet &'+ print_sl_trend('mass_ctb','mass_GrIS',sealevel_stats,period) + '& - & & - & & - & \\\\')
    print_table.append('Antarctic Ice Sheet &'+ print_sl_trend('mass_ctb','mass_AIS',sealevel_stats,period) + '& - & & - & & - & \\\\')
    print_table.append('Terrestrial Water Storage &'+ print_sl_trend('mass_ctb','mass_tws',sealevel_stats,period) + '& - & & - & & - & \\\\')
    print_table.append('Hydrographic observations 0-2000m&'+ print_sl_trend('steric','altimetry',sealevel_stats,period) + '&' + print_ohc_trend('hydrography','altimetry',OHC_stats,period)+ '&'+print_ohu_trend('hydrography','altimetry',OHC_stats,period)+' &' + print_eei_trend('hydrography',EEI_stats,period) + '\\\\')
    print_table.append('Barystatic + hydrography &'+ print_sl_trend('budget','altimetry',sealevel_stats,period) + '& - & & - & & - & \\\\')
    print_table.append('GMSL - barystatic &'+ print_sl_trend('rsl_min_mass','altimetry',sealevel_stats,period) + '&' + print_ohc_trend('altmass_total','altimetry',OHC_stats,period)+ '&'+print_ohu_trend('altmass_total','altimetry',OHC_stats,period)+' &' + print_eei_trend('altmass',EEI_stats,period) + '\\\\')
    print_table.append('GMSL - barystatic - hydrography &'+ print_sl_trend('diff','altimetry',sealevel_stats,period) + '&' + print_ohc_trend('altmass_deep','altimetry',OHC_stats,period)+ '&'+print_ohu_trend('altmass_deep','altimetry',OHC_stats,period)+'  & - & \\\\')
    print_table.append('Deep ocean &'+print_trend(other_terms['deep_ocean']['steric_trend'])+'&'+ print_trend(other_terms['deep_ocean']['ohc_trend']/1e21)+'&'+ print_trend(other_terms['deep_ocean']['ohc_trend']/3600/24/365/4/np.pi/6371000**2) + '& - & \\\\')
    print_table.append('Non-ocean terms&-& &'+print_trend(other_terms['non_ocean']['trend'][period]/1e21)+'&'+ print_trend(other_terms['non_ocean']['trend'][period]/3600/24/365/4/np.pi/6371000**2) +'& - & \\\\')
    print_table.append('CERES EBAF & - & & - & & - & & 0.82 & \\\\')
    for i in print_table:
        print(i)
    return

def print_sl_trend(varname,basin,stats,period):
    if varname == 'mass_ctb':
        trend_mn = "{:.2f}".format(-stats[varname][basin]['trend'][period][1])
        trend_lo = "{:.2f}".format(-stats[varname][basin]['trend'][period][2])
        trend_hi = "{:.2f}".format(-stats[varname][basin]['trend'][period][0])
    else:
        trend_mn = "{:.2f}".format(stats[varname][basin]['trend'][period][1])
        trend_lo = "{:.2f}".format(stats[varname][basin]['trend'][period][0])
        trend_hi = "{:.2f}".format(stats[varname][basin]['trend'][period][2])
    print_str = trend_mn + ' &['+trend_lo+' '+trend_hi+']'
    return(print_str)

def print_ohc_trend(varname,basin,stats,period):
    trend_mn = "{:.1f}".format(stats[varname][basin]['trend'][period][1]/1e21)
    trend_lo = "{:.1f}".format(stats[varname][basin]['trend'][period][0]/1e21)
    trend_hi = "{:.1f}".format(stats[varname][basin]['trend'][period][2]/1e21)
    print_str = trend_mn + ' &['+trend_lo+' '+trend_hi+']'
    return(print_str)

def print_ohu_trend(varname,basin,stats,period):
    eei_fac = (3600*24*365.25*4*np.pi*6371000**2)
    trend_mn = "{:.2f}".format(stats[varname][basin]['trend'][period][1]/eei_fac)
    trend_lo = "{:.2f}".format(stats[varname][basin]['trend'][period][0]/eei_fac)
    trend_hi = "{:.2f}".format(stats[varname][basin]['trend'][period][2]/eei_fac)
    print_str = trend_mn + ' &['+trend_lo+' '+trend_hi+']'
    return(print_str)

def print_eei_trend(varname,stats,period):
    trend_mn = "{:.2f}".format(stats[varname]['trend'][period][1])
    trend_lo = "{:.2f}".format(stats[varname]['trend'][period][0])
    trend_hi = "{:.2f}".format(stats[varname]['trend'][period][2])
    print_str = trend_mn + ' &['+trend_lo+' '+trend_hi+']'
    return(print_str)

def print_trend(trend):
    trend_mn = "{:.2f}".format(trend[1])
    trend_lo = "{:.2f}".format(trend[0])
    trend_hi = "{:.2f}".format(trend[2])
    print_str = trend_mn + ' &['+trend_lo+' '+trend_hi+']'
    return(print_str)

def read_eei_non_ocean():
    fh = Dataset(settings['fn_EEI_GCOS'],'r')
    fh.set_auto_mask(False)
    time = 1950+fh.variables['time'][:]/365.25
    hc_ice    = fh.variables['energy_cryosphere'][:]
    hc_ground = fh.variables['ground_heat_content'][:]
    hc_atm    = fh.variables['atmospheric_heat_content'][:]
    hc_oc_upper    = fh.variables['ohc_0-2000m'][:]
    hc_oc_deep    = fh.variables['ohc_below_2000m'][:]

    eei_non_ocean = gentools.lsqtrend(time[45:-1],(hc_ice+hc_ground+hc_atm)[45:-1])/365.25/24/3600/(4*np.pi*6371000**2)
    eei_ocean_deep = gentools.lsqtrend(time[45:-1],hc_oc_deep[45:-1])/365.25/24/3600/(4*np.pi*6371000**2)
    eei_ocean_upper = gentools.lsqtrend(time[45:-1],hc_oc_upper[45:-1])/365.25/24/3600/(4*np.pi*6371000**2)
    fh.close()
    return(eei_non_ocean)