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
import plot_func as pltfunc

from new_colors import ruths_colors

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
    ticks6_label = [str(o) for o in ticks6]
    
    #ticks6 = np.arange(-15,40,2)/10.0
    ticks6 = np.arange(lim1,lim2,step)/10.0
    ticks6_label = [str(o) for o in ticks6]
    #ticks6_label = ['1e-35','1e-32','1e-29','1e-26','1e-23','1e-20','1e-17','1e-14']
    #ticks6_label = ['1e-35','1e-30','1e-25','1e-20','1e-15']
    #ticks6_label = ['-1e2','1e0','1e1','1e2','1e3','1e4','1e5','1e6','1e7','1e8','1e9']
   # ticks6 = [-1e-22,1e-30,1e-16]
   # ticks6_label = ['-1e-22','1e-30','1e-16']
    
    ax = plt.axes(projection=ccrs.PlateCarree())
    #data_final = np.where(data_final>1.0,data_final,1e-1)
    #x=plt.contourf(new_lon_final,lat,data_final,transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks)
    #x=plt.contourf(new_lon_final,lat,data_final,norm=colors.LogNorm(vmin=1e-8,vmax=1e0),transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks6)
    #x=plt.contourf(new_lon_final,lat,data_final,vmin=-1.5,vmax=4.0 ,transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks6)
    x=plt.contourf(new_lon_final,lat,data_final,vmin=lim1/10,vmax=lim2/10 ,transform=ccrs.PlateCarree(),cmap='bwr',levels=ticks6)
    #x=plt.contourf(new_lon_final,lat,data_final,norm=colors.LogNorm(vmin=pow(10,-35),vmax=1e-15),transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks6)
    norm = mpl.colors.Normalize(vmin=lim1, vmax=lim2)
    #plt.title(title,fontdict={'fontsize':16})
    
    plt.title(title,fontsize=18)
    
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



for i in range(1):
#for i in range(len(alt_data)):
    print('\n File STARTED ')
    p_day='/group_workspaces/jasmin2/asci/eeara/model_runs/u-by114/All_months/' #new run
    p_ind='/group_workspaces/jasmin2/asci/eeara/model_runs/u-by412/All_months/' #baseline run 
    
    p_day='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bf829/All_months/' #vn10.8 baseline run
    p_ind='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bf830/All_months/' #vn10.8 pre-industrial run
    p_day='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz049/All_months/' #vn10.8 baseline run
    p_ind='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bf830/All_months/' #vn10.8 pre-industrial run
    p_day='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz965/All_months/' #vn10.8 baseline run
    p_ind='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz966/All_months/' #vn10.8 pre-industrial run
    # updated suites below 
    #p_day='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz049/All_months/' #preindustrial run
    #p_ind='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz831/All_months/' #baseline run 
    
    #p_day='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz871/All_months/' # run
    #p_ind='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz050/All_months/' #baseline run 
    
    #p_day='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz899/All_months/' #new run
    #p_ind='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz900/All_months/' #baseline run 
    if i%3==0 and i<49:
 
        cube1=iris.load(p_day+'All_months_m01s01i217_UPWARD_SW_FLUX_ON_LEVELS____________.nc')[1]
        cube2=iris.load(p_day+'All_months_m01s01i517_CLEAN-AIR_UPWARD_SW_FLUX_ON_LEVELS__.nc')[0]
        c1   =iris.load(p_day+'All_months_m01s01i519_CLEAR-CLEAN_UPWARD_SW_FLUX_ON_LEVELS.nc')[0]

        cube3_copy = iris.load(p_ind+'All_months_m01s01i217_UPWARD_SW_FLUX_ON_LEVELS____________.nc')[1]
        cube4_copy = iris.load(p_ind+'All_months_m01s01i517_CLEAN-AIR_UPWARD_SW_FLUX_ON_LEVELS__.nc')[0]
        c2_copy    = iris.load(p_ind+'All_months_m01s01i519_CLEAR-CLEAN_UPWARD_SW_FLUX_ON_LEVELS.nc')[0]
        
        cube3 = cube1.copy()
        cube4 = cube1.copy()
        c2 = cube1.copy()
        cube3.data = cube3_copy.data
        cube4.data = cube4_copy.data
        c2.data = c2_copy.data
        
        cube5=iris.load(p_day+'All_months_m01s02i217_UPWARD_LW_FLUX_ON_LEVELS____________.nc')[1]
        cube6=iris.load(p_day+'All_months_m01s02i517_CLEAN-AIR_UPWARD_LW_FLUX_ON_LEVELS__.nc')[0]
        c3   =iris.load(p_day+'All_months_m01s02i519_CLEAR-CLEAN_UPWARD_LW_FLUX_ON_LEVELS.nc')[0]
        
        cube7_copy = iris.load(p_ind+'All_months_m01s02i217_UPWARD_LW_FLUX_ON_LEVELS____________.nc')[1]
        cube8_copy = iris.load(p_ind+'All_months_m01s02i517_CLEAN-AIR_UPWARD_LW_FLUX_ON_LEVELS__.nc')[0]
        c4_copy    = iris.load(p_ind+'All_months_m01s02i519_CLEAR-CLEAN_UPWARD_LW_FLUX_ON_LEVELS.nc')[0]
      
        
        cube7 = cube1.copy()
        cube8 = cube1.copy()
        c4 = cube1.copy()
        cube7.data = cube7_copy.data
        cube8.data = cube8_copy.data
        c4.data = c4_copy.data
        
        
        cube1.units='W m-2'
        cube2.units='W m-2'
        cube3.units='W m-2'
        cube4.units='W m-2'
        cube5.units='W m-2'
        cube6.units='W m-2'
        cube7.units='W m-2'
        cube8.units='W m-2'
        
        c1.units='W m-2'
        c2.units='W m-2'
        c3.units='W m-2'
        c4.units='W m-2'
       
        
        cube3_copy = cube1.copy()
        cube4_copy = cube1.copy()

        cube3_copy.data = cube3.data
        cube4_copy.data = cube4.data
        


        net_sw      = (cube1-cube2)-(cube3-cube4)
        cloud_sw    = (cube2-c1)-(cube4-c2)
        net_lw      = (cube5-cube6)-(cube7-cube8)
        cloud_lw    = (cube6-c3)-(cube8-c4)
        surf_alb_sw = (c1-c2)
        surf_alb_lw = (c3-c4)
        
        di_eff_2 = cube1-cube3
          
        #di_eff.units = 'W m-2'

        di_eff = net_sw.collapsed('time',iris.analysis.MEAN)
        in_eff = cloud_sw.collapsed('time',iris.analysis.MEAN)
        di_eff_2 = di_eff_2.collapsed('time',iris.analysis.MEAN)
       
        #di_eff.units = 'W m-2'

        di_eff = di_eff*-1
        in_eff = in_eff*-1
        dir_title = 'Direct effect (W/m2) '
        ind_title = 'Indirect effect (W/m2)'
        di_eff.units = 'W m-2'
        in_eff.units = 'W m-2'
        
        
        print('max value dir effect = ', np.max(di_eff.data))
        print('min value dir effect = ', np.min(di_eff.data))
        print('mean value dir effect = ', np.mean(di_eff.data))
        #plot_diff(di_eff,dir_title,-120,121,10)
        pltfunc.plot_diff(di_eff,dir_title,-2,2,'seismic')
        #pltfunc.plot_diff(di_eff,dir_title,np.min(di_eff.data),np.max(di_eff.data),'seismic')
        print('max value indir effect = ', np.max(in_eff.data))
        print('min value indir effect = ', np.min(in_eff.data))
        print('mean value indir effect = ', np.mean(in_eff.data))
        plt.savefig('dir_effect_nosulphur_2.eps',dp1 = 500)
        #plot_diff(in_eff,ind_title,-550,551,100)
        #pltfunc.plot_diff(in_eff,ind_title,np.min(in_eff.data),np.max(in_eff.data),'seismic')
        pltfunc.plot_diff_3(in_eff,ind_title,-2,2,'seismic')
        plt.savefig('indir_effect_nosulphur_2.eps',dp1 = 500)

        
        print('Direct radiative forcing = ',np.mean(di_eff.data))
        print('Indirect radiative forcing = ',np.mean(in_eff.data))
        #print 'Surface albedo forcing  -- SHORTWAVE  , ',np.mean(surf_alb_sw)
        #print 'Surface albedo forcing  -- LONGWAVE   , ',np.mean(surf_alb_lw) 
         
        plt.show()

        # ------------------------------------------------------------------
        


        


