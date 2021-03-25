# -------------------------------------------------------------------------------- #
# Compute GRACE sea-level budget over 2003-2019
# Over global mean / regions from Thompson et al (2014) / individual mascons
# Large ensemble to quantify global and regional errors
#
# Datasets
# - Altimetry:  DUACS2018
# - GIA:        Large ensemble from Caron et al. (2018)
# - Ocean mass: JPL RL06 mascon solution
# - Steric:     Computed from ensemble of gridded T/S observations using TEOS-10
#               EN4/Cheng/I17/Levitus/SIO/JAMSTEC/BOA
#
# Written by Thomas Frederikse, Maria Hakuba, Felix Landerer, NASA JPL
# (c) 2021. California Institute of Technology. Government sponsorship acknowledged.
# -------------------------------------------------------------------------------- #
import mod_budget_grace_settings
import mod_budget_grace_mass_rsl_ens
import mod_budget_grace_steric_ens
import mod_budget_grace_global_basin_stats
import mod_budget_grace_grid_trends
import mod_budget_grace_other_terms
from importlib import *

def main():
    # Initialize settings
    reload(mod_budget_grace_settings)
    mod_budget_grace_settings.init()
    mod_budget_grace_mass_rsl_ens.main()       # Compute ensemble members of mass and relative sea level
    mod_budget_grace_steric_ens.main()         # Collect all steric sea level
    mod_budget_grace_other_terms.main()
    mod_budget_grace_global_basin_stats.main() # Statistics for global and basin-mean values
    mod_budget_grace_grid_trends.main()
    return

if __name__ == '__main__':
    main()


