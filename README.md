# Code and data supplement for "Earth’s Energy Imbalance from the ocean perspective (2005 - 2019)"
This repository contains the scripts and data generated for the manuscript "Earth’s Energy Imbalance from the ocean perspective (2005 - 2019)"
Maria Hakuba<sup>1</sup>, Thomas Frederikse<sup>1</sup>, and Felix Landerer<sup>1</sup>

<sup>1</sup> Jet Propulsion Laboratory, California Institute of Technology, Pasadena, California, USA  

© 2021. California Institute of Technology. Government sponsorship acknowledged.
The research was carried out at the Jet Propulsion Laboratory, California Institute of Technology, under a contract with the National Aeronautics and Space Administration (80NM0018D0004)

This supplement contains the scripts used to compute the sea-level budget, the thermal expansion efficiencies of the ocean, and the estimates ocean heat content and ocean heat uptake changes. It also contains the resulting global, basin-mean, and gridded time series of all involved quantities.

## Directory `Code`
This directory contains the Python scripts used to compute all the results of the paper. Please note that these scripts depend on some 3rd-party dependencies, which have to be installed manually, as well as external data sets. Please see the paper and references herein for details on these data sets. 

The script `Budget_GRACE_main.py` calls all the routines that do the actual computation of the sea-level budget and OHC/OHU estimates. The script `est_expansion_efficiency.py` in the `expansion_efficiency` directory computes the expansion efficiencies. 

## Directory `Data`

### `grid_tseries.nc`
   NetCDF file with gridded estimates of ocean mass, steric sea level, geocentric sea level, and relative sea level. All units are in mm. Multiple free software packages are available to view and modify NetCDF files. For example [python](https://unidata.github.io/netcdf4-python/), [Julia](https://github.com/Alexander-Barth/NCDatasets.jl), [GMT](https://www.generic-mapping-tools.org/), [ncview](http://meteora.ucsd.edu/~pierce/ncview_home_page.html), and many others. 

### `trends_global.xlsx`
	Excel sheets with linear trends in sea level, ocean heat content and the EEI over various periods. Each period has its own tab. 
 
### `timeseries_global.xlsx`
   Excel sheets with time series of all components of the global sea-level budget, ocean heat content, and ocean heat uptake. For each quantity for which uncertainty estimates are available, the first column denotes the lower bound (5th percentile), the second column the best estimate, and the third column the upper bound (95th percentile). For sea level, the units are in mm. For OHC (second tab), the units are Joules, and for OHU, the units are in W/m^2. 

### `timeseries_basins.xlsx`
   Excel sheets with time series of all components of the basin-mean sea-level budget. Each basin has its own tab. For each quantity, the first column denotes the lower bound (5th percentile), the second column the best estimate, and the third column the upper bound (95th percentile). The units are in mm.
