#%%
import glob
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# =========================
#      USER SETTINGS      
# -------------------------
model = 'TaiESM-TIMCOM'      # TaiESM (yiwen) or TaiESM-TIMCOM (yhtseng)
climo_year = [1980, 1999]    # Climatological time period
# =========================

# READ TIMCOM (TaiESM-TIMCOM) MODEL files
def read_taiesm(model):
    match model:
        case 'TaiESM-TIMCOM':
            # File list
            root = '/work/yhtseng00/scratch/taiesm_timcom/archive/BTHIST_test08a/'
            fli = sorted(glob.glob(root+'ocn/hist/DATA*.nc'), key=str.casefold)
            # Load coordinate
            dc = timcom_grid(root+'ocn/hist/TOPO.nc')

        case 'TaiESM':
            # File list
            root = '/work/yiwen89419/TAIESM/taiesm_work/archive/B1850_TAI/'
            fli = sorted(glob.glob(root+'ocn/hist/B1850_TAI.pop.h.0*.nc'), key=str.casefold)
            # Load coordinate
            dc = pop_grid(root+'ocn/hist/B1850_TAI.pop.h.once.nc')

    # Open each ncfile
    SST, SST_mon_ts, VAT_mon_ts = [], [], []
    for fnm in fli:
        print(fnm)
        with xr.open_dataset(fnm, engine='netcdf4', decode_timedelta=True) as ds:
            # Correct timestamps
            ds = convert_time(model, ds, fnm)
            match model:
                case 'TaiESM-TIMCOM':
                    # Assign coordinate (time,lat,lon,lev)
                    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon', 'level-center': 'lev'})
                    ds = ds.assign_coords({**dc.coords, 'time': ds['time'].values})
                    # NaN values
                    temp = ds['temperature'].where(ds['temperature'] != 0)
                    temp = temp.expand_dims(time=[ds['time'].values[0]])

                case 'TaiESM':
                    # Assign coordinate
                    ds = ds.rename({'nlat': 'lat', 'nlon': 'lon', 'z_t': 'lev'})
                    ds = ds.assign_coords(dict(lon=(['lat','lon'], ds['TLONG'].values), 
                                               lat=(['lat','lon'], ds['TLAT'].values)))
                    temp = ds['TEMP']

        # Save SST & VAT
        sst = temp.isel(lev=0, drop=True).copy(deep=True)
        SST.append(sst)
        SST_mon_ts.append(area_weighted_mean(sst, dc['area']))
        VAT_mon_ts.append(volume_weighted_mean(temp, dc['vol']))

    # Concatenate
    SST = xr.concat(SST, dim='time')
    SST_mon_ts = xr.concat(SST_mon_ts, dim='time')
    VAT_mon_ts = xr.concat(VAT_mon_ts, dim='time')
    return SST, SST_mon_ts, VAT_mon_ts

# READ HadISST observation
def read_hadisst(fnm):
    ds = xr.open_dataset(fnm, engine='netcdf4')

    ds = convert_time('HadISST', ds)
    lon, lat = np.meshgrid(ds['lon'], ds['lat'])
    ds = ds.assign_coords(dict(lon=(['lat','lon'], lon),
                               lat=(['lat','lon'], lat)))
    return ds

# LOAD & CALCULATE TIMCOM grid (include grid size)
def timcom_grid(fnm):
    with xr.open_dataset(fnm, engine='netcdf4') as ds:
        # Calculate grid length
        deg2m = 111200
        dlon = ds['lon'].differentiate('longitude') * deg2m
        dlat = ds['lat'].differentiate('latitude') * deg2m * np.cos(np.deg2rad(ds['lat']))
        dz = ds['lev_f'].diff('level-face') * 1e-2

        # Calculate grid size
        area = (dlon * dlat).transpose('latitude','longitude')
        vol = (area * dz).transpose('level-face','latitude','longitude')

        # Meshgrid lon and lat
        lon, lat = np.meshgrid(ds['lon'], ds['lat'])
        dc = xr.Dataset(data_vars=dict(area=(['lat','lon'], area.values),
                                       vol=(['lev','lat','lon'], vol.values)),
                        coords=dict(lon=(['lat','lon'], lon), 
                                    lat=(['lat','lon'], lat),
                                    lev=('lev',ds['lev_c'].values)))
    return dc

# LOAD & CALCULATE POP grid (include grid size)
def pop_grid(fnm):
    with xr.open_dataset(fnm, engine='netcdf4') as ds:
        ds = ds.rename({'nlat': 'lat', 'nlon': 'lon', 'z_t': 'lev'})
        ds = ds.rename_vars({'TAREA': 'area'})
        ds['vol'] = ds['area']*ds['dz']
        return ds

# AREA-WEIGHTED mean
def area_weighted_mean(var, area):
    return var.weighted(area).mean(dim=('lat','lon'))

# VOLUME-WEIGHTED mean
def volume_weighted_mean(var, vol):
    return var.weighted(vol).mean(dim=('lev','lat','lon'))

# CORRECT timestamps (datetime64)
def convert_time(dsnm, ds, fnm=None):
    match dsnm:
        # For TaiESM-TIMCOM: read the file name
        case 'TaiESM-TIMCOM':
            base = pd.Timestamp('1850-01-01')
            y, m = int(fnm[-9:-5]), int(fnm[-5:-3])
            ms = (y-1)*12+m-1
            time = base + pd.DateOffset(months=ms)
            ds['time'] = [time]
        
        # For TaiESM: shift back one month manually
        case 'TaiESM':
            raw_time = ds['time'].values[0]
            y = raw_time.year + 1849
            m = raw_time.month - 1
            if m == 0:
                m, y = 12, y-1
            time = pd.Timestamp(y, m, 1)
            ds['time'] = [time]

        # For HadISST: from string to cftime
        case 'HadISST':
            time = ds['time'].values.astype(str)
            yr = [int(t[:4]) for t in time]
            mn = [int(t[4:6]) for t in time]
            time = [pd.Timestamp(year=y, month=m, day=1) for y, m in zip(yr, mn)]
            ds['time'] = time
    return ds

# CALCULATE climatology map
def climo_map(da, climo_year):
    st = pd.Timestamp(year=climo_year[0], month=1, day=1)
    en = pd.Timestamp(year=climo_year[1], month=12, day=1)
    da_climo_map = da.sel(time=slice(st,en)).mean(dim='time')
    return da_climo_map

# REGRID the coarser dataset (for meshgrid)
def regrid_to_match(da1, da2):
    from scipy.interpolate import griddata

    # Decide which grid is finer
    gridsize_da1, gridsize_da2 = da1.size, da2.size

    # A = target, B = source
    A, B = (da1, da2) if gridsize_da1 > gridsize_da2 else (da2, da1)

    # Source grid (B) points and values
    lonB, latB = B['lon'].values, B['lat'].values
    points = np.column_stack([lonB.ravel(), latB.ravel()])
    data   = B.values.ravel()

    # Target grid (A)
    lonA, latA = A['lon'].values, A['lat'].values
    varg = griddata(points, data, (lonA, latA), method='linear')

    # Wrap back into DataArray
    da = xr.DataArray(data=varg, dims=['lat','lon'],
                      coords=dict(lat=(['lat','lon'], A['lat'].values), 
                                  lon=(['lat','lon'], A['lon'].values)))
    if gridsize_da1 > gridsize_da2:
        da2 = da
    else:
        da1 = da
    return da1, da2

# PLOT time series
def plot_mean_ts(mdl,title,fignm):
    fig, axs = plt.subplots(figsize=(5,2.5), nrows=1, ncols=1)
    plt.plot(mdl['time'], mdl, 'r', linewidth=1.2)
    plt.grid(alpha=0.2)
    plt.xlim(mdl['time'].values[0], mdl['time'].values[-1])
    plt.title(title,fontweight='bold',fontsize=14)
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
    lon = ((var2[0]['lon'] + 180) % 360) - 180

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
        cn = ax[v].pcolormesh(lon, var['lat'], var, \
                            cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        ax[v].add_feature(cart.feature.COASTLINE)
        gl = ax[v].gridlines(crs=ccrs.PlateCarree(), draw_labels=False, color='gray', alpha=0.2)
        gl.xlocator = mticker.FixedLocator(xtick)
        gl.ylocator = mticker.FixedLocator(ytick)
        gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        ax[v].set_xticks(xtick-180)
        ax[v].set_xticklabels(['0\N{DEGREE SIGN}','90\N{DEGREE SIGN}E','180\N{DEGREE SIGN}',\
                            '90\N{DEGREE SIGN}W','0\N{DEGREE SIGN}'])
        ax[v].set_yticks(ytick)
        ax[v].set_yticklabels(['90\N{DEGREE SIGN}S','45\N{DEGREE SIGN}S','0\N{DEGREE SIGN}',\
                            '45\N{DEGREE SIGN}N','90\N{DEGREE SIGN}N'])
        titlem = title + ' - ' + titles[v] + ' (' + unit + ')'
        ax[v].set_title(titlem, fontweight='bold')
        cbar = fig.colorbar(cn, orientation='vertical', extend='both', shrink=.8)
        cbar.ax.tick_params(axis='y', labelsize=8)
    fig.savefig(fignm, bbox_inches='tight')
    plt.show()

#%% MODEL DATA PROCESS =============================================
# Read data
SST, SST_mon_ts, VAT_mon_ts = read_taiesm(model)

# Climatology map
SST_climo_map = climo_map(SST, climo_year)

# Global mean SST and VAT time series
SST_yr_ts = SST_mon_ts.resample(time='1YE').mean()
VAT_yr_ts = VAT_mon_ts.resample(time='1YE').mean()
del SST

#%% OBSERVATION =====================================================
root = '/work/j07hcl00/work/taiesm_diag/cvdp.v5.1.1/data_org/OBS/'
fnm = root + 'hadisst.sst.187001-201812.nc'
ds = read_hadisst(fnm)

# Climatology map (lat, lon)
SSTobs_climo_map = climo_map(ds['sst'], climo_year)
SST_climo_map, SSTobs_climo_map = regrid_to_match(SST_climo_map, SSTobs_climo_map)
del ds,fnm,root

#%% PLOTTING ===================================================
import cmaps

# SST time series
plot_mean_ts(SST_yr_ts,
             title=f'Global mean SST (\N{DEGREE SIGN}C)',
             fignm=f'{model}/SST_ts')

# VAT time series
plot_mean_ts(VAT_yr_ts,
             title=f'Global mean VAT (\N{DEGREE SIGN}C)',
             fignm=f'{model}/VAT_ts')

# SST map
plot_climo_map([SST_climo_map, SSTobs_climo_map],
               cvmin=-3, cvmax=33, ccmap=cmaps.WhBlGrYeRe.to_seg(N=12),
               dv=3, dcmap=cmaps.cmp_b2r.to_seg(N=15),
               title=f'SST',
               unit=f'\N{DEGREE SIGN}C',
               fignm=f'{model}/SST_map')
