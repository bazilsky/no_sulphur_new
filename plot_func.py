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

from new_colors import ruths_colors,ruths_colors_r

def plot_diff(slice1,title,lim1,lim2,col_bar):
    #plt.figure()
    flag = abs(lim1)>abs(lim2)
    
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
    
    if flag == True:
        ticks7 = np.linspace(lim1,-lim1,10)
    else:
        ticks7 = np.linspace(-lim2,lim2,10)
    ticks6 = ticks7.copy()
    ticks6 = np.around(ticks6,decimals = 2)
    ticks6_label = [str(o) for o in ticks6]
            
    """    
    ticks7 = np.linspace(lim1,lim2,10)
    ticks6 = ticks7.copy()
    ticks6 = np.around(ticks6,decimals = 0)
    ticks6_label = [str(o) for o in ticks6]
    """
    ax = plt.axes(projection=ccrs.PlateCarree())
    #data_final = np.where(data_final>1.0,data_final,1e-1)
    #x=plt.contourf(new_lon_final,lat,data_final,transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks)
    #x=plt.contourf(new_lon_final,lat,data_final,norm=colors.LogNorm(vmin=1e-8,vmax=1e0),transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks6)
    #x=plt.contourf(new_lon_final,lat,data_final,vmin=-50,vmax=50,transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks6)
    x=plt.contourf(new_lon_final,lat,data_final,vmin=np.min(ticks7),vmax=np.max(ticks7),transform=ccrs.PlateCarree(),cmap=col_bar,levels=ticks7)
    #x=plt.contourf(new_lon_final,lat,data_final,norm=colors.LogNorm(vmin=pow(10,-35),vmax=1e-15),transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks6)
    norm = mpl.colors.Normalize(vmin=lim1, vmax=lim2)
    #plt.title(title,fontdict={'fontsize':16})
    
    plt.title(title,fontsize=15)
    
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
    cbar = plt.colorbar(x,ax=ax)
    cbar.set_ticks(ticks6)
    cbar.set_ticklabels(ticks6_label)
    cbar.ax.tick_params(labelsize=16)
    ax.set_aspect('auto')
#________________________________
def plot_diff_2(slice1,title,lim1,lim2,col_bar):
    #plt.figure()
    flag = abs(lim1)>abs(lim2)
    
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
    
    if flag == True:
        ticks7 = np.linspace(lim1,-lim1,15)
    else:
        ticks7 = np.linspace(-lim2,lim2,15)
    ticks6 = ticks7.copy()
    ticks6 = np.around(ticks6,decimals = 0)
    ticks6_label = [str(o) for o in ticks6]
            
    """    
    ticks7 = np.linspace(lim1,lim2,10)
    ticks6 = ticks7.copy()
    ticks6 = np.around(ticks6,decimals = 0)
    ticks6_label = [str(o) for o in ticks6]
    """
    ax = plt.axes(projection=ccrs.PlateCarree())
    #data_final = np.where(data_final>1.0,data_final,1e-1)
    #x=plt.contourf(new_lon_final,lat,data_final,transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks)
    #x=plt.contourf(new_lon_final,lat,data_final,norm=colors.LogNorm(vmin=1e-8,vmax=1e0),transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks6)
    #x=plt.contourf(new_lon_final,lat,data_final,vmin=-50,vmax=50,transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks6)
    x=plt.contourf(new_lon_final,lat,data_final,vmin=np.min(ticks7),vmax=np.max(ticks7),transform=ccrs.PlateCarree(),cmap=col_bar,levels=ticks7)
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
    cbar = plt.colorbar(x,ax=ax)
    cbar.set_ticks(ticks6)
    cbar.set_ticklabels(ticks6_label)
    cbar.ax.tick_params(labelsize=16)
    ax.set_aspect('auto')
#________________________________
def plot_diff_3(slice1,title,lim1,lim2,col_bar):
    #plt.figure()
    flag = abs(lim1)>abs(lim2)
    
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
    
    if flag == True:
        ticks7 = np.linspace(lim1,-lim1,10)
    else:
        ticks7 = np.linspace(-lim2,lim2,10)
    ticks6 = ticks7.copy()
    ticks6 = np.around(ticks6,decimals = 1)
    ticks6_label = [str(o) for o in ticks6]
            
    """    
    ticks7 = np.linspace(lim1,lim2,10)
    ticks6 = ticks7.copy()
    ticks6 = np.around(ticks6,decimals = 0)
    ticks6_label = [str(o) for o in ticks6]
    """
    ax = plt.axes(projection=ccrs.PlateCarree())
    #data_final = np.where(data_final>1.0,data_final,1e-1)
    #x=plt.contourf(new_lon_final,lat,data_final,transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks)
    #x=plt.contourf(new_lon_final,lat,data_final,norm=colors.LogNorm(vmin=1e-8,vmax=1e0),transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks6)
    #x=plt.contourf(new_lon_final,lat,data_final,vmin=-50,vmax=50,transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks6)
    x=plt.contourf(new_lon_final,lat,data_final,vmin=np.min(ticks7),vmax=np.max(ticks7),transform=ccrs.PlateCarree(),cmap=col_bar,levels=ticks7)
    #x=plt.contourf(new_lon_final,lat,data_final,norm=colors.LogNorm(vmin=pow(10,-35),vmax=1e-15),transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks6)
    norm = mpl.colors.Normalize(vmin=lim1, vmax=lim2)
    #plt.title(title,fontdict={'fontsize':16})
    
    plt.title(title,fontsize=13)
    
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
    cbar = plt.colorbar(x,ax=ax)
    cbar.set_ticks(ticks6)
    cbar.set_ticklabels(ticks6_label)
    cbar.ax.tick_params(labelsize=16)
    ax.set_aspect('auto')

##########--------------------------------------------------------
def plot_diff_4(slice1,data_arr,title,lim1,lim2,col_bar):
    #plt.figure()
    flag = abs(lim1)>abs(lim2)
    
    plt.figure(figsize=(12, 6))# this works 
    #plt.figure(figsize=(12, 4.8))# this works 
    data=data_arr
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
    
    if flag == True:
        ticks7 = np.linspace(lim1,-lim1,15)
    else:
        ticks7 = np.linspace(-lim2,lim2,15)
    ticks7 = np.arange(lim1,lim2,10)
    ticks6 = ticks7.copy()
    #ticks6 = np.around(ticks6,decimals = 1)
    ticks6_label = [str(o)+'%' for o in ticks6]
            
    """    
    ticks7 = np.linspace(lim1,lim2,10)
    ticks6 = ticks7.copy()
    ticks6 = np.around(ticks6,decimals = 0)
    ticks6_label = [str(o) for o in ticks6]
    """
    ax = plt.axes(projection=ccrs.PlateCarree())
    #data_final = np.where(data_final>1.0,data_final,1e-1)
    #x=plt.contourf(new_lon_final,lat,data_final,transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks)
    #x=plt.contourf(new_lon_final,lat,data_final,norm=colors.LogNorm(vmin=1e-8,vmax=1e0),transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks6)
    #x=plt.contourf(new_lon_final,lat,data_final,vmin=-50,vmax=50,transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks6)
    x=plt.contourf(new_lon_final,lat,data_final,vmin=np.min(ticks7),vmax=np.max(ticks7),transform=ccrs.PlateCarree(),cmap=col_bar,levels=ticks7)
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
    cbar = plt.colorbar(x,ax=ax)
    cbar.set_ticks(ticks6)
    cbar.set_ticklabels(ticks6_label)
    cbar.ax.tick_params(labelsize=16)
    ax.set_aspect('auto')

##########--------------------------------------------------------

def plot_diff_delT(slice1,data_arr,title,lim1,lim2,col_bar):
    #plt.figure()
    flag = abs(lim1)>abs(lim2)
    
    plt.figure(figsize=(12, 6))# this works 
    #plt.figure(figsize=(12, 4.8))# this works 
    data=data_arr
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
    
    if flag == True:
        ticks7 = np.linspace(lim1,-lim1,15)
    else:
        ticks7 = np.linspace(-lim2,lim2,15)
    
    ticks7 = np.arange(lim1,lim2+1,1)
    ticks7 = np.arange(0,lim2,1)
    ticks6 = ticks7.copy()
    #ticks6 = np.around(ticks6,decimals = 0)
    #u'\u207B\u00B3'
    ticks6_label = [str(o)+u'\u2070'+'C' for o in ticks6]
            
    """    
    ticks7 = np.linspace(lim1,lim2,10)
    ticks6 = ticks7.copy()
    ticks6 = np.around(ticks6,decimals = 0)
    ticks6_label = [str(o) for o in ticks6]
    """
    ax = plt.axes(projection=ccrs.PlateCarree())
    #data_final = np.where(data_final>1.0,data_final,1e-1)
    #x=plt.contourf(new_lon_final,lat,data_final,transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks)
    #x=plt.contourf(new_lon_final,lat,data_final,norm=colors.LogNorm(vmin=1e-8,vmax=1e0),transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks6)
    #x=plt.contourf(new_lon_final,lat,data_final,vmin=-50,vmax=50,transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks6)
    x=plt.contourf(new_lon_final,lat,data_final,vmin=np.min(ticks7),vmax=np.max(ticks7),transform=ccrs.PlateCarree(),cmap=ruths_colors('lajolla'),levels=ticks7)
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
    cbar = plt.colorbar(x,ax=ax)
    cbar.set_ticks(ticks6)
    cbar.set_ticklabels(ticks6_label)
    cbar.ax.tick_params(labelsize=16)
    ax.set_aspect('auto')


def plot_diff_a(slice1,title,lim1,lim2,col_bar):
    #plt.figure()
    flag = abs(lim1)>abs(lim2)
    
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
    
    if flag == True:
        ticks7 = np.linspace(lim1,-lim1,10)
    else:
        ticks7 = np.linspace(-lim2,lim2,10)
    ticks6 = ticks7.copy()
    ticks6 = np.around(ticks6,decimals = 2)
    ticks6_label = [str(o) for o in ticks6]
            
    """    
    ticks7 = np.linspace(lim1,lim2,10)
    ticks6 = ticks7.copy()
    ticks6 = np.around(ticks6,decimals = 0)
    ticks6_label = [str(o) for o in ticks6]
    """
    ax = plt.axes(projection=ccrs.PlateCarree())
    #data_final = np.where(data_final>1.0,data_final,1e-1)
    #x=plt.contourf(new_lon_final,lat,data_final,transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks)
    #x=plt.contourf(new_lon_final,lat,data_final,norm=colors.LogNorm(vmin=1e-8,vmax=1e0),transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks6)
    #x=plt.contourf(new_lon_final,lat,data_final,vmin=-50,vmax=50,transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks6)
    x=plt.contourf(new_lon_final,lat,data_final,vmin=np.min(ticks7),vmax=np.max(ticks7),transform=ccrs.PlateCarree(),cmap=ruths_colors('vik'),levels=ticks7)
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
    cbar = plt.colorbar(x,ax=ax)
    cbar.set_ticks(ticks6)
    cbar.set_ticklabels(ticks6_label)
    cbar.ax.tick_params(labelsize=16)
    ax.set_aspect('auto')
