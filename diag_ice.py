#%%
import glob
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# =========================
#      USER SETTINGS      
# -------------------------
model = 'TaiESM'      # TaiESM (yiwen) or TaiESM-TIMCOM (yhtseng)
climo_year = [1979, 2005]    # Climatological time period
# =========================

# READ MODEL files
def read_taiesm(model, climo_year):
    SIC, SIA_NH, SIA_SH = [], [], []

    # # File list for different models
    match model:
        case 'TaiESM-TIMCOM':
            root = '/work/yhtseng00/scratch/taiesm_timcom/archive/BTHIST_test08a/'
            fli = sorted(glob.glob(root+'ice/hist/BTHIST*.nc'), key=str.casefold)

        case 'TaiESM':
            root = '/work/yiwen89419/TAIESM/taiesm_work/archive/B1850_TAI/'
            fli = sorted(glob.glob(root+'ice/hist/B1850*.nc'), key=str.casefold)

    # Open each ncfile
    for fnm in fli:
        with xr.open_dataset(fnm, engine='netcdf4') as ds:
            # Correct timestamps            
            ds = convert_time(model, ds)
            # Skip the year out of range
            yr = ds.indexes['time'].year
            if (yr < climo_year[0]) | (yr > climo_year[1]):
                continue       
            print(fnm)
            
            # Fill NaN values
            ds['aice'] = ds['aice'].where(ds['aice'] != 0)
            ds['aice'] = ds['aice'].where(ds['aice'] <=100)

            # Sea ice concentration (SIC)
            SIC.append(ds['aice'])

            # Total sea ice area (SIA)
            sia_nh, sia_sh = ice_extent(model, ds)
            SIA_NH.append(sia_nh)
            SIA_SH.append(sia_sh)

    # Concatenate
    SIC = xr.concat(SIC, dim='time')
    SIA_NH = xr.concat(SIA_NH, dim='time')
    SIA_SH = xr.concat(SIA_SH, dim='time')
    return SIC, SIA_NH, SIA_SH

# READ NSIDC observation
def read_nsidc(fnm, hem, climo_year):
    with xr.open_dataset(fnm, engine='netcdf4') as ds:

        # Assign names same with models'
        ds = ds.rename({'latitude': 'TLAT', 'longitude': 'TLON'})
        ds = ds.rename_dims({'ygrid': 'nj', 'xgrid': 'ni'})
        ds = ds.rename_vars({'AREA': 'tarea', 'SIC': 'aice'})

        # Convert units
        ds['tarea'] = ds['tarea'] * 1e6     # km2 -> m2
        ds['aice']  = ds['aice'] * 1e2       # fraction -> %

        # Fill the hole
        if hem == 'NH':
            mask = (ds['TLAT'] >= 84) & ds['aice'].isnull()
            ds['aice'] = ds['aice'].where(~mask, 100)

        # Calculate SIA
        match hem:
            case 'NH':
                sia, _ = ice_extent('OBS', ds)
            case 'SH':
                _, sia = ice_extent('OBS', ds)
        sia = sia.where(sia > 2e12, np.nan)

    # Time selection
    st = pd.Timestamp(year=climo_year[0], month=1, day=1)
    en = pd.Timestamp(year=climo_year[1], month=12, day=1)
    sia = sia.sel(time=slice(st,en))
    return sia

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

        # For TaiESM: add 1850 and shift back one month manually
        case 'TaiESM':
            raw_time = ds['time'].values[0]
            y = raw_time.year + 1849
            m = raw_time.month - 1
            if m == 0:
                m, y = 12, y-1
            time = pd.Timestamp(y, m, 1)
            ds['time'] = [time]
    return ds

# CALCULATING SIA
def ice_extent(dsnm, ds):

    # area where SIC >= 15%
    sic = ds['aice']
    sic = xr.where(sic >= 15, 100, 0)

    # North Hemisphere
    mask = (ds['TLAT']>0)
    aice_nh = (sic.where(mask)*1e-2*ds['tarea']).sum(dim=('nj','ni'))
    if dsnm == 'TaiESM-TIMCOM':
        re = 6371220
        mlat = ds['TLAT'].max().values
        np_area = 2*np.pi*re**2*(1-np.cos(np.deg2rad(90-mlat)))
        aice_nh = aice_nh + np_area

    # South Hemisphere
    mask = (ds['TLAT']<0)
    aice_sh = (sic.where(mask)*1e-2*ds['tarea']).sum(dim=('nj','ni'))
    return aice_nh, aice_sh

# CALCULATE climatology map
def climo_map(da, climo_year):
    st = pd.Timestamp(year=climo_year[0], month=1, day=1)
    en = pd.Timestamp(year=climo_year[1], month=12, day=1)
    da_climo_map = da.sel(time=slice(st,en)).mean(dim='time')
    return da_climo_map

# PLOT climatology maps
def plot_climo_map(var,HM,vmin,vmax,cmap,title,unit,fignm):
    import matplotlib.path as mpath
    import cartopy as cart
    import cartopy.crs as ccrs
    import matplotlib.ticker as mticker
    from matplotlib.tri import Triangulation
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    lonlat_proj = ccrs.PlateCarree()
    match HM:
        case 'NH':
            proj = ccrs.NorthPolarStereo(central_longitude=0)
            d01 = [-180,180,45,90]
        case 'SH':
            proj = ccrs.SouthPolarStereo(central_longitude=0)
            d01 = [-180,180,-90,-45]
    fig, axs = plt.subplots(figsize=(5,5), nrows=1, ncols=1, facecolor='white',\
                                subplot_kw={'projection':proj})
    match model:
        case 'TaiESM-TIMCOM':
            cn = axs.pcolormesh(var['TLON'], var['TLAT'], var, \
                                cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        case 'TaiESM':
            [lon_valid, lat_valid, var_valid] = nan_coord(var)
            tri = Triangulation(lon_valid.flatten(), lat_valid.flatten())
            cn = plt.tripcolor(tri, var_valid.flatten(), cmap=cmap, vmin=vmin, vmax=vmax, \
                          transform=ccrs.PlateCarree()) 

    axs.set_extent(d01, lonlat_proj)
    axs.gridlines()
    axs.add_feature(cart.feature.LAND, facecolor='darkgrey')
    axs.add_feature(cart.feature.COASTLINE)
    titlem = HM + ' ' + title + ' - ' + model + unit
    axs.set_title(titlem, fontweight='bold')
    cbar = fig.colorbar(cn, orientation='vertical', extend='both', shrink=.5)
    cbar.ax.tick_params(axis='y', labelsize=8)

    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    axs.set_boundary(circle, transform=axs.transAxes)
    fig.savefig(fignm, bbox_inches='tight')
    plt.show()

# REMOVE the NaN values in coordinates (for TaiESM when plotting)... 
def nan_coord(var):
    lon = var['TLON'].values
    lat = var['TLAT'].values
    var1 = var.values
    mask = (~np.isnan(lon)) | (~np.isnan(lat))
    lon_valid = lon[mask]
    lat_valid = lat[mask]
    var_valid = var1[mask]
    return lon_valid, lat_valid, var_valid

# PLOT SIA time series
def plot_mean_ts(obs,mdl,title,unit,fignm):
    hem = ['NH ', 'SH ']
    fig,ax = plt.subplots(figsize=(5,6), nrows=2, ncols=1)
    plt.subplots_adjust(hspace=.3)
    for v in range(2):
        ax[v].plot(obs[v]['time'], obs[v], 'k', linewidth=1.2)
        ax[v].plot(mdl[v]['time'], mdl[v], 'r', linewidth=1.2)
        ax[v].grid(alpha=0.2)
        ax[v].set_xlim(obs[v]['time'].values[0], obs[v]['time'].values[-1])
        titles = hem[v] + title + ' - ' + model + unit
        ax[v].set_title(titles,fontweight='bold',fontsize=14)
        ax[v].legend(['Observation', model])
    fig.savefig(fignm, bbox_inches='tight')
    plt.show()

#%% MODEL DATA PROCESS =========================================
# Read data
[SIC, SIA_NH_mon_ts, SIA_SH_mon_ts] = read_taiesm(model, climo_year)

# Climatology map (lat, lon)
SIC_climo_map = climo_map(SIC, climo_year)

# Sea ice area (extent) time series
SIA_NH_yr_ts = SIA_NH_mon_ts.resample(time='1YE').mean()
SIA_SH_yr_ts = SIA_SH_mon_ts.resample(time='1YE').mean()

#%% OBSERVATION =====================================================
root = '/work/j07hcl00/work/taiesm_diag/cvdp.v5.1.1/data_org/OBS/'

# North Hemisphere
fnm = root + 'seaice_conc_monthly_nh_NOAA_NSIDC_CDR.v03r01.197811-201702.nc'
SIAobs_NH_mon_ts = read_nsidc(fnm, 'NH', climo_year)
SIAobs_NH_yr_ts = SIAobs_NH_mon_ts.resample(time='1YE').mean()

# North Hemisphere
fnm = root + 'seaice_conc_monthly_sh_NOAA_NSIDC_CDR.v03r01.197811-201702.nc'
SIAobs_SH_mon_ts = read_nsidc(fnm, 'SH', climo_year)
SIAobs_SH_yr_ts = SIAobs_SH_mon_ts.resample(time='1YE').mean()


#%% PLOTTING ===================================================
import cmaps

# NH SIC map
plot_climo_map(SIC_climo_map, HM='NH',
               vmin=0, vmax=100, cmap=cmaps.NCV_bright.to_seg(N=10),
               title=f'SIC',
               unit=r' $(10^{12}\,\mathrm{m}^2)$',
               fignm=f'{model}/SIC_NH_map')

# SH SIC map
plot_climo_map(SIC_climo_map, HM='SH',
               vmin=0, vmax=100, cmap=cmaps.NCV_bright.to_seg(N=10),
               title=f'SIC',
               unit=r' $(10^{12}\,\mathrm{m}^2)$',
               fignm=f'{model}/SIC_SH_map')

# SIA time series
plot_mean_ts([SIAobs_NH_yr_ts*1e-12, SIAobs_SH_yr_ts*1e-12], 
             [SIA_NH_yr_ts*1e-12, SIA_SH_yr_ts*1e-12],
             title=f'SIA',
             unit=r' $(10^{12}\,\mathrm{m}^2)$',
             fignm=f'{model}/SIA_ts')
