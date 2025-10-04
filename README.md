# Homework #1
Here are my scripts and plots for HW1. No need to create any files before running my Python code (-ω-)/  
Please don't just copy and paste my code, since you still have to do the final project on your own (; ･`д･´)  
(P.S. We're still recovering the yiwen89419's atmosphere data currently... I'll update once the files are back ( ;∀;))

## diag_atm.py
For the atmosphere data, you have to plot the surface air temperature (SAT) and precipitation (PREC).
### SAT
* "TREFHT" and "TS" are the temperature at reference height (2m) and on the sea surface, respectively. Both are okay for the HW.
* Calculating anomaly is optional. If you plot the observation without subtracting its climatology, you might see something interesting around 1950 !(^^)!
### PREC 
* Sum of "PRECC" (convective) and "PRECL" (large-scale).
* Please pay attention to their units!

## diag_ocn.py
For the ocean data, the two assigned directories are two different ocean models: the TIMCOM model in the yhtseng00 directory, and the POP model in the yiwen89419 directory.
Because the temperature variable is 3D (lev, lat, lon), which is very large. Please be aware of the RAM used.
### TIMCOM (yhtseng00)
* The coordinates are in "TOPO.nc".
* "lev_c" (55) is the center location of the vertical grid, while "lev_f" (56) is the surface (interfaces) location of it. Use lev_f to calculate the grid depth.
### POP (yiwen89419)
* "B1850_TAI.pop.h.once.nc" might be useful for calculating the weighted mean.
* "TAREA" is the grid area, while "dz" is the grid depth.
* Because of the irregular grid, you might need to use "griddata" instead of "interp2d" for regridding, and "pcolormesh" instead of "contourf" for plotting.

## diag_ice.py
For the sea ice data, you need to read the sea ice concentration (SIC) "aice" and calculate the sea ice area (SIA, or sea ice extent).
* Please pay attention to their units!
* SIA is defined as the area where the SIC is over 15%.
* "tarea" or "AREA" is the grid area for calculating SIA.
* There are some missing data in some datasets around the north pole. Please fill it back when calculating the sea ice area.
* For students who read yiwen89419's, because the coordinate contains NaN, you might need to use "tripcolor" with "Triangulation".
