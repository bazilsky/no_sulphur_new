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

    
    #...........................................................


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
    #lat=c.coord('latitude').points
    #lon=c.coord('longitude').points
    #data=c.data
    ax = plt.axes(projection=ccrs.PlateCarree())
    #im = ax.imshow(data_final)
    #x=plt.contourf(new_lon_final,lat,data_final,transform=ccrs.PlateCarree(),cmap=plt.cm.jet,levels=np.arange(data.min(),data.max(),1))
    #x=plt.contourf(new_lon_final,lat,data_final,transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks)
    #x=plt.contourf(new_lon_final,lat,data_final,transform=ccrs.PlateCarree(),cmap='RdYlBu',levels=ticks,vmin = -100, vmax = 30)
    x=plt.contourf(new_lon_final,lat,data_final,transform=ccrs.PlateCarree(),cmap='RdYlBu_r',levels=ticks,vmin = lim1, vmax = lim2)
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
    
    #plt.colorbar(x,fraction = 0.046, pad=0.04)
    #formatter = LogFormatter(10, labelOnlyBase=False)
    cbar=plt.colorbar(x,ax=ax,ticks=ticks)
    cbar.ax.tick_params(labelsize=16)
    #cbar.set_ticks(ticks)
    #cbar.set_ticklabels(ticks)
    ax.set_aspect('auto')
    #plt.colorbar(im,fraction=0.046, pad=0.04)
    plt.savefig(title+'.png')
   

sigma=[1.59,1.59,1.40,2.0,1.59,1.59,2.0]
cutoff=2
pref=1.013e5
tref=293.0
zboltz=1.3807e-23
staird=pref/(tref*287.058) 




def get_model_data(mpath,data_level):
    model_aird = iris.load(mpath+'L1_air_density_Density of air.nc')
    model_aird = model_aird[0]
    
    model_acc = iris.load(mpath+'L1_n_accsol_number_of_particles_per_air_molecule_of_soluble_accumulation_mode_aerosol_in_air.nc')#particles/m3
    model_nuc = iris.load(mpath+'L1_n_nucsol_number_of_particles_per_air_molecule_of_soluble_nucleation_mode_aerosol_in_air.nc')
    model_aitins = iris.load(mpath+'L1_n_aitins_number_of_particles_per_air_molecule_of_insoluble_aitken_mode_aerosol_in_air.nc')
    model_ait = iris.load(mpath+'L1_n_aitsol_number_of_particles_per_air_molecule_of_soluble_aitken_mode_aerosol_in_air.nc')
    #print model_ait,'\n'
    model_cor=iris.load(mpath+'L1_n_corsol_number_of_particles_per_air_molecule_of_soluble_coarse_mode_aerosol_in_air.nc')
    
    model_nucrad = iris.load(mpath+'L1_rad_nucsol_Radius_of_mode_nucsol.nc')
    model_aitrad = iris.load(mpath+'L1_rad_aitsol_Radius_of_mode_aitsol.nc')
    model_aitirad = iris.load(mpath+'L1_rad_aitins_Radius_of_mode_aitins.nc')

    model_acc=model_acc[0]
    model_nuc=model_nuc[0]
    model_aitins=model_aitins[0]
    model_ait=model_ait[0]
    model_cor=model_cor[0]
    model_nucrad=model_nucrad[0]
    model_aitrad=model_aitrad[0]
    model_aitirad=model_aitirad[0]

    model_acc_stp = model_acc*staird/model_aird
    model_ait_stp = model_ait*staird/model_aird
    model_cor_stp = model_cor*staird/model_aird
    model_nuc_stp = model_nuc*staird/model_aird
    model_aitins_stp = model_aitins*staird/model_aird

    model_nuc_thl = model_nuc_stp - model_nuc_stp.copy(lognormal_cummulative(model_nuc_stp.data, cutoff*1.0e-9, model_nucrad.data, sigma[0]))
    model_ait_thl = model_ait_stp - model_ait_stp.copy(lognormal_cummulative(model_ait_stp.data, cutoff*1.0e-9, model_aitrad.data, sigma[1]))
    model_aitins_thl = model_aitins_stp - model_aitins_stp.copy(lognormal_cummulative(model_aitins_stp.data, cutoff*1.0e-9, model_aitirad.data, sigma[4]))
    
    model_n = model_nuc_thl+model_ait_thl+model_aitins_thl+model_acc_stp+model_cor_stp
    model_n=model_n.collapsed('time',iris.analysis.MEAN)
    model_n=model_n[data_level,:,:]
    model_n = model_n/1000000
    print (model_n)
    
    model_n.long_name='Particle Number Concentration at STP (cm-3)'
    model_n.units='cm-3'
    return model_n


#--------------------------------


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


data_level=12 # cloud base level = 1.3km
#data_level=26 # cloud base level = 5.2km
#data_level=38 # cloud base level = 11.2kmkm

for i in range(1):
#for i in range(len(alt_data)):
    print ('\n File STARTED ')
    #if i==0:
    #p_day1='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bw810/L1/' # this is the baseline
    #p_day2='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bw929/L1/' # new run  
    
    p_day1='/group_workspaces/jasmin2/asci/eeara/model_runs/u-by921/All_months/' # this is the baseline
    p_day2='/group_workspaces/jasmin2/asci/eeara/model_runs/u-by920/All_months/' # new run  
    
    p_day1='/gws/nopw/j04/asci/eeara/model_runs/u-ca440/All_months/' # this is the baseline
    p_day2='/gws/nopw/j04/asci/eeara/model_runs/u-cb602/All_months/' # new run  


    #p_ind1='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bf521/L1/'
    #p_ind2='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bf586/L1/'
    #p_ind3='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bf577/L1/'
    if i%3==0 and i<49:
        
        #dat1=get_model_data(p_day1,data_level)
        #dat2=get_model_data(p_day2,data_level)
        
        #cn_file = 'All_months_m01s38i437_CN__NUMBER_CONCENTRATION____________.nc'
        cn_file = '*m01s38i437*' 
        #a=dat1
        #b=dat2
        a = iris.load(p_day1+cn_file)[0]
        b = iris.load(p_day2+cn_file)[0]
        a=a.collapsed('time',iris.analysis.MEAN)
        a=a[data_level,:,:]
        
        b=b.collapsed('time',iris.analysis.MEAN)
        b=b[data_level,:,:]
        #c=(b-a)*100/a
        #d=dat3
        #e=(d-a)*100/a
        c= b-a 
        print ('the highest is = ', np.max(c.data))
        print ('the lowest is = ', np.min(c.data))
        
        #temp_str='Change in Ntot at 11km (cm'+u'\u207B\u00B3'+')'
        temp_str='Change in Ntot at 1km (cm'+u'\u207B\u00B3'+')'
        #plot_diff(c,temp_str,-100,31,10)
        #plot_diff(c,temp_str,-4000,4000,500) # for cloud base
        #plot_diff(c,temp_str,-5000,5001,500) # for 5km
        #plot_diff(c,temp_str,-10000,10001,1000) # for 11km
        #plot_diff(c,temp_str,-600,601,100) # for 11km
        plot_diff(c,temp_str,-100,101,20) # for 1km
        print ('max = ', np.max(c)) 
        print ('min = ', np.min(c))
        image_filepath = '/home/users/eeara/no_sulphur_new/images/'
        plt.savefig(image_filepath+'ntot_slice_1km.png',dpi = 500)
        #plt.show()

        # ------------------------------------------------------------------
        


        
