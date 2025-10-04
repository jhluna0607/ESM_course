#%%
import glob
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

# =========================
#      USER SETTINGS      
# -------------------------
model = 'TaiESM-TIMCOM'      # TaiESM (yiwen) or TaiESM-TIMCOM (yhtseng)
climo_year = [1980, 1999]    # Climatological time period
# =========================

# READ MODEL files and select SAT and PREC
def read_taiesm(model):
    SAT, PREC = [], []

    # File list for different models
    match model:
        case 'TaiESM-TIMCOM':
            root = '/work/yhtseng00/scratch/taiesm_timcom/archive/BTHIST_test08a/'
        case 'TaiESM':
            root = '/work/yiwen89419/TAIESM/taiesm_work/archive/B1850_TAI/'
    fli = sorted(glob.glob(root+'atm/hist/B*.nc'), key=str.casefold)

    # Open each ncfile
    for fnm in fli:
        print(fnm)
        with xr.open_dataset(fnm, engine='netcdf4') as ds:
            # Correct timestamps
            ds = convert_time(model, ds)
            # Change units
            SAT.append(ds['TREFHT'] - 273.15)
            PREC.append((ds['PRECC'] + ds['PRECL']) * 86400 * 1000)

    # Concatenate
    SAT  = xr.concat(SAT, dim='time')
    PREC = xr.concat(PREC, dim='time')
    return SAT, PREC

# READ BEST SAT observation
def read_best(fnm):
    with xr.open_dataset(fnm, engine='netcdf4') as ds:
        # time (cftime)
        ds = convert_time('BEST', ds)
        # Rename the coordinates name
        ds = ds.rename({'latitude': 'lat'})
        ds = ds.rename({'longitude': 'lon'})
        # (-180-180) to (0-360)
        ds = convert_lon_360(ds)
    return ds

# READ GPCP PREC observation
def read_gpcp(fnm):
    fnm = root + 'gpcp.mon.mean.197901-201904.nc'
    ds = xr.open_dataset(fnm, engine='netcdf4')
    return ds

# CORRECT timestamps (datetime64)
def convert_time(dsnm, ds):
    match dsnm:
        # For TaiESM-TIMCOM: shift back one month manually
        case 'TaiESM-TIMCOM':
            raw_time = ds['time'].values[0]
            time = pd.to_datetime(str(raw_time))
            y, m = time.year, time.month
            if m == 1:
                y, m = y-1, 12
            else:
                m = m-1
            time = pd.Timestamp(year=y, month=m, day=1)
            ds['time'] = [time]

        # For BEST: from string to datetime64
        case 'BEST':
            time = ds['time'].values.astype(str)
            yr = [int(t[:4]) for t in time]
            mn = [int(t[4:6]) for t in time]
            time = [pd.Timestamp(year=y, month=m, day=1) for y, m in zip(yr, mn)]
            ds['time'] = time
    return ds

# CONVERT longitude coordinate from (-180,180) to (0,360)
def convert_lon_360(ds):
    ds['lon'] = ds['lon'].where(ds['lon'] > 0, ds['lon'] + 360)
    ds = ds.sortby(ds.lon)
    return ds

# CALCULATE climatology map
def climo_map(da, climo_year):
    st = pd.Timestamp(year=climo_year[0], month=1, day=1)
    en = pd.Timestamp(year=climo_year[1], month=12, day=1)
    da_climo_map = da.sel(time=slice(st,en)).mean(dim='time')
    return da_climo_map

# CALCULATE climatology (12 mons) and anomaly
def anomaly(da, climo_year):
    st = pd.Timestamp(year=climo_year[0], month=1, day=1)
    en = pd.Timestamp(year=climo_year[1], month=12, day=1)
    SATc = da.sel(time=slice(st,en)).groupby('time.month').mean('time')     # (12,lat,lon)
    SATa = da.groupby('time.month') - SATc                                  # (time,lat,lon)
    return SATc, SATa

# WEIGHTING mean according to different latitude
def weighted_mean(var):
    weights = np.cos(np.deg2rad(var['lat']))
    return var.weighted(weights).mean(dim=('lat','lon'))

# REGRID the coarser dataset (for unmeshgrid)
def regrid_to_match(da1, da2):
    from scipy.interpolate import griddata

    # Decide which grid is finer
    gridsize_da1 = da1['lon'].size * da1['lat'].size
    gridsize_da2 = da2['lon'].size * da2['lat'].size

    # A = target, B = source
    A, B = (da1, da2) if gridsize_da1 > gridsize_da2 else (da2, da1)

    # Source grid (B) points and values
    lonB, latB = np.meshgrid(B['lon'].values, B['lat'].values)
    points = np.column_stack([lonB.ravel(), latB.ravel()])
    data   = B.values.ravel()

    # Target grid (A)
    lonA, latA = np.meshgrid(A['lon'].values, A['lat'].values)
    varg = griddata(points, data, (lonA, latA), method='linear')

    # Wrap back into DataArray
    da = xr.DataArray(data=varg, dims=['lat','lon'],
                      coords=dict(lat=(['lat'], A['lat'].values), 
                                  lon=(['lon'], A['lon'].values)))
    if gridsize_da1 > gridsize_da2:
        da2 = da
    else:
        da1 = da
    return da1, da2

# PLOT GMSAT time series
def plot_mean_ts(obs,mdl,title,fignm):
    fig, axs = plt.subplots(figsize=(5,2.5), nrows=1, ncols=1)
    plt.plot(obs['time'], obs, 'k', linewidth=1.2)
    plt.plot(mdl['time'], mdl, 'r', linewidth=1.2)
    plt.grid(alpha=0.2)
    plt.xlim(obs['time'].values[0], obs['time'].values[-1])
    plt.title(title,fontweight='bold',fontsize=14)
    plt.legend(['Observation', model])
    fig.savefig(fignm, bbox_inches='tight')
    plt.show()

# PLOT climatology maps
def plot_climo_map(var2, cvmin, cvmax, ccmap, dv, dcmap, title, unit, fignm):
    import cartopy as cart
    import cartopy.crs as ccrs
    import matplotlib.ticker as mticker
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    # General settings
    xtick = np.arange(0,361,90)
    ytick = np.arange(-90,91,45)
    titles = [model,'Observation','Difference']
    var2.append(var2[0]-var2[1])

    # Create a clear canva
    proj = ccrs.PlateCarree(central_longitude=180)
    fig,ax = plt.subplots(figsize=(5,8), nrows=3, ncols=1, facecolor='white',\
                            subplot_kw={'projection':proj})
    plt.subplots_adjust(hspace=.1)

    # Each subplot (model, obs, diff)
    for v, var in enumerate(var2):
        if v <= 1:
            vmin, vmax, cmap = cvmin, cvmax, ccmap
        else:
            vmin, vmax, cmap = -dv, dv, dcmap
        cn = ax[v].pcolormesh(var['lon'], var['lat'], var, \
                            cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        ax[v].add_feature(cart.feature.COASTLINE)
        gl = ax[v].gridlines(crs=ccrs.PlateCarree(), draw_labels=False, color='gray', alpha=0.2)
        gl.xlocator = mticker.FixedLocator(xtick)
        gl.ylocator = mticker.FixedLocator(ytick)
        gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        ax[v].set_xticks(xtick-180)
        ax[v].set_xticklabels(['0\N{DEGREE SIGN}','90\N{DEGREE SIGN}E','180\N{DEGREE SIGN}',\
                            '90\N{DEGREE SIGN}W','0\N{DEGREE SIGN}'],fontsize=8)
        ax[v].set_yticks(ytick)
        ax[v].set_yticklabels(['90\N{DEGREE SIGN}S','45\N{DEGREE SIGN}S','0\N{DEGREE SIGN}',\
                            '45\N{DEGREE SIGN}N','90\N{DEGREE SIGN}N'],fontsize=8)
        titlem = title + ' - ' + titles[v] + ' (' + unit + ')'
        ax[v].set_title(titlem, fontweight='bold')
        cbar = fig.colorbar(cn, orientation='vertical', extend='both', shrink=.8)
        cbar.ax.tick_params(axis='y', labelsize=8)
    fig.savefig(fignm, bbox_inches='tight')
    plt.show()

#%% MODEL DATA PROCESS =============================================
# Read data
SAT, PREC = read_taiesm(model)                      # (time,lat,lon)

# Climatological map
SAT_climo_map = climo_map(SAT, climo_year)          # (lat, lon)
PREC_climo_map = climo_map(PREC, climo_year)

# Global mean SATa time series 
_, SATa = anomaly(SAT)                              # (time,lat,lon)
SAT_mon_ts = weighted_mean(SATa)
SAT_yr_ts = SAT_mon_ts.resample(time='1YE').mean()
del PREC,SAT,SATa

#%% OBSERVATION =====================================================
# SAT ---------------------------------------------------------------
root = '/work/j07hcl00/work/taiesm_diag/cvdp.v5.1.1/data_org/OBS/'
fnm = root + 'best.tas.185001-201902.nc'
ds = read_best(fnm)

# Climatology map (lat, lon)
SATobs_climo_map = climo_map(ds['tas'], climo_year)
SAT_climo_map, SATobs_climo_map = regrid_to_match(SAT_climo_map, SATobs_climo_map)

# Global mean SATa time series
_, SATa = anomaly(ds['tas'], climo_year)                  # (time,lat,lon)
SATobs_mon_ts = weighted_mean(SATa)
SATobs_yr_ts = SATobs_mon_ts.resample(time='1YE').mean()

# PREC --------------------------------------------------------------
fnm = root + 'gpcp.mon.mean.197901-201904.nc'
ds = read_gpcp(fnm)

# Climatology map (lat, lon)
PRECobs_climo_map = climo_map(ds['precip'], climo_year)
PREC_climo_map, PRECobs_climo_map = regrid_to_match(PREC_climo_map, PRECobs_climo_map)
del ds,fnm,root,SATa

#%% PLOTTING ===================================================
import cmaps

# SAT map
plot_climo_map([SAT_climo_map, SATobs_climo_map],
               cvmin=-14, cvmax=34, ccmap=cmaps.WhBlGrYeRe.to_seg(N=12),
               dv=4, dcmap=cmaps.cmp_b2r.to_seg(N=16),
               title=f'SAT',
               unit=f'\N{DEGREE SIGN}C',
               fignm=f'{model}/SAT_map')

# SAT time series
plot_mean_ts(SATobs_yr_ts,SAT_yr_ts,
             title=f'Global mean SAT (\N{DEGREE SIGN}C)',
             fignm=f'{model}/SAT_ts')

# Precipitation map
plot_climo_map([PREC_climo_map, PRECobs_climo_map],
               cvmin=0, cvmax=13, ccmap=cmaps.WhiteBlueGreenYellowRed.to_seg(N=13),
               dv=3, dcmap=cmaps.MPL_BrBG.to_seg(N=15),
               title=f'Preci.',
               unit=f'mm/day',
               fignm=f'{model}/PREC_map')
