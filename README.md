# Homework #1
Here are my scripts and plots for HW1. The climatological period I chose is from 1980 to 1999.
(P.S. We are still recovering the yiwen89419's atmosphere data currently!)

## diag_atm.py
For the atmosphere data, you have to plot the surface air temperature (SAT) and precipitation (PREC).
### SAT
* "TREFHT" and "TS" are the temperature at reference height (2m) and on the sea surface, respectively. Both are okay.
* Calculating anomaly is optional. If you plot the observation without subtracting its climatology, you might see something interesting around 1950 !(^^)!
### PREC 
* Sum of "PRECC" (convective) and "PRECL" (large-scale).
* Please pay attention to their units!

## diag_ocn.py
For the ocean data, the two assigned directories are two different ocean models: the TIMCOM model in the yhtseng00 directory, and the POP model in the yiwen89419 directory.
Because the temperature variable is 3D (lev, lat, lon), which is very large. Please take care of your RAM used.
### TIMCOM
* "TOPO.nc" might be useful for calculating the weighted mean.
* "lev_c" (55) is the center of the vertical grid, while "lev_f" (56) is the surface (interfaces) of it. Use lev_f to calculate the grid depth.
### POP
* "B1850_TAI.pop.h.once.nc" might be useful for calculating the weighted mean.
* "TAREA" is the grid area.
* "dz" is the grid depth.
* Because of the irregular grid, you might need to use "griddata" instead of "interp2d" for regridding, and "pcolormesh" instead of "contourf" for plotting.
