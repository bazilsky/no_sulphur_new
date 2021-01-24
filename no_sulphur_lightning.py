import iris,glob,datetime, time,re,sys
import numpy as np
import scipy as sp
import numpy.ma as ma
import matplotlib.pyplot as plt
import iris.coord_categorisation
import cf_units
import iris.quickplot as qplt

from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors

from cartopy import config
import cartopy.crs as ccrs
import matplotlib as mpl
from matplotlib.ticker import LogFormatter

import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


iris.FUTURE.cell_datetime_objects=True


trial1=iris.cube.CubeList()
trial2=iris.cube.CubeList()

def plot_diff(slice1,title,lim1,lim2,step):
    #plt.figure()
    plt.figure(figsize=(12, 6))# this works 
    #plt.figure(figsize=(12, 4.8))# this works 
    data=slice1.data
    lon=slice1.coord('longitude').points
    lat=slice1.coord('latitude').points
    new_lon=[]
    for k in range(len(lon)):
        if lon[k]>180:
            #temp=lon[k]
            temp=lon[k]-360
            new_lon=np.append(new_lon,temp)
        else:
            new_lon=np.append(new_lon,lon[k])
    #lon=lon-180    #basemap correction 
    #new_lon=temp

#..............basemap requires lat and lon to be in increasing order
    
    data_1=data[:,0:96]
    data_2=data[:,96:]
    data_21=np.hstack((data_2,data_1))

    new_lon_1=new_lon[0:96]
    new_lon_2=new_lon[96:]
    new_lon_21 = np.hstack((new_lon_2, new_lon_1))


    data_final=data_21
    new_lon_final=new_lon_21

    
    ticks=np.arange(lim1,lim2,step)
    ticks6 = [1e0,1e3,1e5,1e8,1e10]
    ticks6_label = ['1e0','1e3','1e5','1e8','1e10']
    ticks6 = [1e-2,1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9]
    ticks6_label = ['-1e2','1e0','1e1','1e2','1e3','1e4','1e5','1e6','1e7','1e8','1e9']
    ticks6 = [1e-33,1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9]
    ticks6_label = ['-1e2','1e0','1e1','1e2','1e3','1e4','1e5','1e6','1e7','1e8','1e9']
    ticks6 = [1e-35,1e-30,1e-25,1e-20,1e-15]
    ticks6_label = ['1e-35','1e-30','1e-25','1e-20','1e-15']
    ticks6 = [1e-35,1e-32,1e-29,1e-26,1e-23,1e-20,1e-17,1e-14]
    ticks6_label = ['1e-35','1e-32','1e-29','1e-26','1e-23','1e-20','1e-17','1e-14']
    
    ticks6 = [1e-20,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0]
    
    ticks6 = [1e-20,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0]
    ticks6 = [-300,-250,-200,-150,-100,-50,0,50] # this is a percentage change plot
    ticks6 = [-50,-40,-30,-20,-10,0,10,20,30,40,50] # this is a percentage change plot
    ticks6 = [1e-10,1e-6,1e-5,1e-4,1e-3,1e-2]
    ticks6 = [1e-6,5e-6,1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2]
    ticks6 = [5e-6,1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2]
    #ticks6 = np.arange(0,601,50) 
    #ticks6 = [1e-20,1e0,5e0,1e1,5e1,1e2,5e2,1e3,5e3,1e4,5e4,1e5,5e5,1e6,5e6]
    ticks6 = np.arange(lim1,lim2,step)
    ticks6_label = [str(o) for o in ticks6]
    #ticks6_label = ['1e-35','1e-32','1e-29','1e-26','1e-23','1e-20','1e-17','1e-14']
    #ticks6_label = ['1e-35','1e-30','1e-25','1e-20','1e-15']
    #ticks6_label = ['-1e2','1e0','1e1','1e2','1e3','1e4','1e5','1e6','1e7','1e8','1e9']
   # ticks6 = [-1e-22,1e-30,1e-16]
   # ticks6_label = ['-1e-22','1e-30','1e-16']
    
    ax = plt.axes(projection=ccrs.PlateCarree())
    #data_final = np.where(data_final>1.0,data_final,1e-1)
    #x=plt.contourf(new_lon_final,lat,data_final,transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks)
    #x=plt.contourf(new_lon_final,lat,data_final,norm=colors.LogNorm(vmin=np.min(ticks6),vmax=np.max(ticks6)),transform=ccrs.PlateCarree(),cmap='magma',levels=ticks6)
    x=plt.contourf(new_lon_final,lat,data_final,vmin=np.min(ticks6),vmax=np.max(ticks6),transform=ccrs.PlateCarree(),cmap='seismic',levels=ticks6)
    #x=plt.contourf(new_lon_final,lat,data_final,vmin=0,vmax=5,transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks6,extendfrac=0)
    #x=plt.contourf(new_lon_final,lat,data_final,vmin=0,vmax=600,transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks6)
    #x=plt.contourf(new_lon_final,lat,data_final,vmin=-50,vmax=50,transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks6)
    #x=plt.contourf(new_lon_final,lat,data_final,norm=colors.LogNorm(vmin=pow(10,-35),vmax=1e-15),transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks6)
    norm = mpl.colors.Normalize(vmin=lim1, vmax=lim2)
    #plt.title(title,fontdict={'fontsize':16})
    
    plt.title(title,fontsize=24)
    
    ax.coastlines()

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=2, color='black', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    #gl.xlines = False
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([-90, -45, 0, 45, 90])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.xlabel_style = {'color': 'black', 'weight': 'bold'}
    
    gl.ylabel_style = {'size': 15, 'color': 'gray'}
    gl.ylabel_style = {'color': 'black', 'weight': 'bold'}
    
    """
    cbar=plt.colorbar(x,ax=ax,ticks=ticks)
    cbar.ax.tick_params(labelsize=16)
    ax.set_aspect('auto')
    plt.savefig(title+'.png')
   
    """
    #ax.set_extent([.min(), x.max(), y.min(), y.max()], crs=proj_lcc)
    cbar = plt.colorbar(x,ax=ax)
    cbar.set_ticks(ticks6)
    cbar.set_ticklabels(ticks6_label)
    cbar.ax.tick_params(labelsize=16)
    ax.set_aspect('auto')
#________________________________


sigma=[1.59,1.59,1.40,2.0,1.59,1.59,2.0]
cutoff=2
pref=1.013e5
tref=293.0
zboltz=1.3807e-23
staird=pref/(tref*287.058) 




def get_model_data(mpath,data_level):
    
    ccn_file = 'All_months_m01s50i082_lightning_flashrate_per_column.nc'  
    ccn_data = iris.load(mpath+ccn_file)[0]
    mcopy = ccn_data.copy() 
    model_n = ccn_data
    model_n=model_n.collapsed('time',iris.analysis.MEAN)
    model_n = model_n*(365*24*60) # number of lightning flashes annually
    model_slice = model_n  # level 46 is 15km and level 64 is 30kmis cloud base
    
    model_n.long_name='lightning (cm-3)'
    model_n.units='cm-3'
    return model_slice,mcopy




var1=[]
var2=[]
var3=[]
var4=[]
var5=[]
var6=[]
var7=[]
var8=[]
#f=plt.figure()
indx=[10,20,30,35,40,45,50,55]
alt=[3.4,5.1,7.8,9.8,12.1,14.8,18,21.7]


data_level=12 # cloud base level = 12  altitude 15km

for i in range(1):
#for i in range(len(alt_data)):
    print('\n File STARTED ')
    #if i==0:

    p_day1='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz001/All_months/'  # baseline model run
    p_day2='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bs407/All_months/'  # no sulphut model run

    if i%3==0 and i<49:
        
        dat1,mcopy1=get_model_data(p_day1,data_level)
        dat2,mcopy2=get_model_data(p_day2,data_level)
        
        a=dat1.copy()
        b=dat2.copy()
        c = a
        c = ((b-a)/a)*100

        print('the highest is  = ', np.max(c.data))
        print('the lowest is   = ', np.min(c.data))
        print('total lightning = ', np.sum(c.data))
        temp_str='Percentage increase in Lightning strikes per year ' # note the superscript is done 
        #c.data=np.where(c.data>=0,c.data,1e-20) #IMPORTANT LINE TO MASK UNNCESSARY DATA
        plot_diff(c,temp_str,-60,101,20)
        plt.savefig('lightning_slice_nosulphur.eps', dpi = 500) 
        
        plt.show()

        # ------------------------------------------------------------------
        


        
