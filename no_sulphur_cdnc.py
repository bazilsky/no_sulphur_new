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
import plot_func as pltfunc

from new_colors import ruths_colors
iris.FUTURE.cell_datetime_objects=True


trial1=iris.cube.CubeList()
trial2=iris.cube.CubeList()

def find_nearest_index(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def cat_func_lat(position):
    return asl.coord('latitude').nearest_neighbour_index(position)

def cat_func_lon(position):
    return asl.coord('longitude').nearest_neighbour_index(position)    

def lognormal_cummulative(N,r,rbar,sigma):
    total=(N/2)*(1+sp.special.erf(np.log(r/rbar)/np.sqrt(2)/np.log(sigma)))
    return total

sigma=[1.59,1.59,1.40,2.0,1.59,1.59,2.0]

pref=1.013e5
tref=293.0
zboltz=1.3807e-23
staird=pref/(tref*287.058) 


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
    ticks6 = [-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0] # this is a percentage change plot
    ticks6_label = [str(o)+'%' for o in ticks6]
    #ticks6_label = ['1e-35','1e-32','1e-29','1e-26','1e-23','1e-20','1e-17','1e-14']
    #ticks6_label = ['1e-35','1e-30','1e-25','1e-20','1e-15']
    #ticks6_label = ['-1e2','1e0','1e1','1e2','1e3','1e4','1e5','1e6','1e7','1e8','1e9']
   # ticks6 = [-1e-22,1e-30,1e-16]
   # ticks6_label = ['-1e-22','1e-30','1e-16']
    
    ax = plt.axes(projection=ccrs.PlateCarree())
    #data_final = np.where(data_final>1.0,data_final,1e-1)
    #x=plt.contourf(new_lon_final,lat,data_final,transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks)
    #x=plt.contourf(new_lon_final,lat,data_final,norm=colors.LogNorm(vmin=1e-8,vmax=1e0),transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks6)
    x=plt.contourf(new_lon_final,lat,data_final,vmin=-100,vmax=0,transform=ccrs.PlateCarree(),cmap=ruths_colors('nuuk'),levels=ticks6)
    #x=plt.contourf(new_lon_final,lat,data_final,vmin=-100,vmax=0,transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks6)
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


sigma=[1.59,1.59,1.40,2.0,1.59,1.59,2.0]
cutoff=2
pref=1.013e5
tref=293.0
zboltz=1.3807e-23
staird=pref/(tref*287.058) 




def get_model_data(mpath,data_level):
    
    #ccn_file = 'L1_ccn0.2_Cloud_condensation_nuclei_at_a_supersaturation_of_0.2000.nc'
    ccn_file = 'All_months_m01s34i968_cloud_drop_number_conc.nc'
    ccn_data = iris.load(mpath+ccn_file)[0]

    #model_n = model_secorg
     
    #model_n = soa1+soa2+soa3+soa4+soa5+soa6+soa7
    
    model_n = ccn_data
    model_n=model_n.collapsed('time',iris.analysis.MEAN)
    #model_n=model_n.collapsed('model_level_number',iris.analysis.SUM) 
    model_slice = model_n[12,:,:]
    #model_n = model_n*16*1e-12*(365*24*3600)
    #model_n=model_n[data_level,:,:]
    #model_n = model_n/1000000
    #print model_n
    
    model_n.long_name='Particle Number Concentration at STP (cm-3)'
    model_n.units='cm-3'
    return model_slice




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

    p_day1='/group_workspaces/jasmin2/asci/eeara/model_runs/u-by412/All_months/'  # baseline model run
    p_day2='/group_workspaces/jasmin2/asci/eeara/model_runs/u-by114/All_months/'  # new model run

    if i%3==0 and i<49:
        
        dat1=get_model_data(p_day1,data_level)
        dat2=get_model_data(p_day2,data_level)
        
        a=dat1
        b=dat2
        #a = dat1*(29.0/150.0)/1e-12 # convert it to ppt
        #b = dat2*(29.0/150.0)/1e-12
        
        c=(b-a)*100.0/a
        #c = b
        #c = a
        #c = b-a
        #d=dat3
        #e=(d-a)*100/a
         
        print('the highest is = ', np.max(c.data))
        print('the lowest is = ', np.min(c.data))
        print('mean = ', np.mean(c.data))
        temp_str='Percentage decrease in CDNC at 1km - present day'
        #c.data=np.where(c.data>=0,c.data,1e-20) #IMPORTANT LINE TO MASK UNNCESSARY DATA
        plot_diff(c,temp_str,1e-20,1e0,20)
        #pltfunc.plot_diff(c,temp_str,0,100,'seismic')
        
        plt.savefig('cndnc_nosulphur_2.eps',dpi = 500) 
        
        plt.show()

        # ------------------------------------------------------------------
        


        
