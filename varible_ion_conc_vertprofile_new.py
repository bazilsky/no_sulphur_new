import iris
import glob
import datetime
import time
import re
import sys
import numpy as np
import scipy as sp
import numpy.ma as ma
import matplotlib.pyplot as plt
import iris.coord_categorisation
import cf_units
import iris.quickplot as qplt
import pylab as pl

from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
from matplotlib.ticker import LogFormatter
import matplotlib.ticker as ticker

from new_colors import ruths_colors

iris.FUTURE.cell_datetime_objects=True


trial1=iris.cube.CubeList()
trial2=iris.cube.CubeList()

path = '/group_workspaces/jasmin2/gassp/eeara/CARIBIC/'#ANANTH

mpath1='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz047/' # biogenic nucelation branch
#mpath1='/group_workspaces/jasmin2/asci/eeara/model_runs/u-by920/' # biogenic nucelation branch
#mpath2='/group_workspaces/jasmin2/asci/eeara/model_runs/u-by921/' # baseline simulation branch
#mpath1='/gws/nopw/j04/gassp/hgordon/u-be424-noColinFix/L1/'

alt_path='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bf834/L1/'
#alt_path='/gws/nopw/j04/gassp/hgordon/u-be424-noColinFix/L1/'
#mpath2='/group_workspaces/jasmin2/gassp/eeara/model_runs/u-bc244/L1/'
cutoff=2 # this is a radius...12nm is a diameter!!

altitude_path=alt_path+'L1_rad_accsol_Radius_of_mode_accsol.nc'
cube=iris.load(altitude_path)
cube=cube[0]
alt_data=cube.coord('altitude').points
alt_data_2=cube.coord('altitude').points
alt_data=alt_data[:,72,96]
alt_data=alt_data/1000


sigma=[1.59,1.59,1.40,2.0,1.59,1.59,2.0]

pref=1.013e5
tref=293.0
zboltz=1.3807e-23
staird=pref/(tref*287.058) 


#f=plt.figure()
indx=[10,20,30,35,40,45,50,55]
alt=[3.4,5.1,7.8,9.8,12.1,14.8,18,21.7]

#for i in range(1):
flag=[30]

def get_temp():
    filepath = '/group_workspaces/jasmin2/asci/eeara/model_runs/u-bs405/All_months/'
    potential_temperature = iris.load(filepath+'All_months_m01s00i004_THETA_AFTER_TIMESTEP________________.nc')[0]
    air_pressure = iris.load(filepath+'All_months_m01s00i408_PRESSURE_AT_THETA_LEVELS_AFTER_TS___.nc')[0]
    
    p0 = iris.coords.AuxCoord(1000.0,long_name = 'reference_pressure',units = 'hPa')
    p0.convert_units(air_pressure.units)
    Rd = 287.05
    cp = 1005.46
    Rd_cp = Rd/cp
    temperature=potential_temperature*(air_pressure/p0)**(Rd_cp)
    iris.save(temperature,'model_temperature.nc') 
    return temperature

for i in flag:
#for i in range(len(alt_data)):
    print('\n File STARTED ')
    #if i==0:
    if i%3==0:
        #new1=get_model_data(i,alt_data[i],i,mpath1)
        #cond_sink_path = '/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz147/All_months/'+'All_months_m01s38i440_CCN_NO._CONCENTRN._(ACC+COR+AIT>35r).nc'
        cond_sink_path = '/gws/nopw/j04/asci/eeara/model_runs/u-cb815/2017jan/2017jan_m01s38i438_CCN_NO._CONCENTRN._(ACC+COR)________.nc'
        filepath = '/group_workspaces/jasmin2/asci/eeara/model_runs/u-bs405/All_months/'
        #filepath = '/group_workspaces/jasmin2/asci/eeara/model_runs/u-bs405/All_months/'
        temperature = iris.load('/home/users/eeara/CARIBIC_CODE/model_temperature.nc')[0]
        trop_height = iris.load(filepath + 'All_months_m01s30i453_Height_at_Tropopause_Level__________.nc')[0]
        trop_height = trop_height.collapsed('time',iris.analysis.MEAN)
        trop_height = trop_height.collapsed('longitude',iris.analysis.MEAN)
        trop_height = trop_height/1000.0
        max_arr = []
        model_temp = temperature.collapsed('time',iris.analysis.MEAN)
        model_temp = model_temp.collapsed('longitude',iris.analysis.MEAN)
        model_temp = model_temp[0:50,:]
        #model_temp.coord('model_level_number').points = alt_data[0:66]
        model_lat = model_temp.coord('latitude').points
        plt.figure()

        new1 = iris.load(cond_sink_path)[0]
        print(new1)
        #new1 = new1*(29.0/64.0)/1.0e-12 # converting mass fraction to pptv
        new1=new1.collapsed('longitude',iris.analysis.MEAN)
        #new1=new1.collapsed('time',iris.analysis.MEAN)
        
        #print(new1)
        new1=new1[0:50,:] # 66 index is to an alitutde of close to 32km 

        
        data=new1.data
        lnum1=new1.coord('model_level_number').points
        lnum_alt=alt_data[0:50]
        lat1=new1.coord('latitude').points
        x,y=np.meshgrid(lat1,lnum_alt)
        ticks=[1,5,10,20,50,100,200,500,1000,2000,5000,10000,30000,60000,100000,200000]
        #ticks3=[1e0,5e0,1e1,2e1,5e1,1e2,2e2,5e2,1e3,2e3,5e3,1e4,3e4,6e4,1e5,2e5]
        #ticks3       = [1e0,1e1,5e1,1e2,2.5e2,5e2,7.5e2,1e3,2.5e3,5e3,7.5e3,1e4,2.5e4,5e4,7.5e4,1e5,2.5e5]
        ticks3       = [1e0,1e1,5e1,1e2,2.5e2,5e2,7.5e2,1e3,2.5e3,5e3,7.5e3,1e4,2.5e4,5e4,7.5e4,1e5,2.5e5]
        ticks3_label = ['1','10','50',u'10\u00b2',u'2.5 x 10\u00b2',u'5 x 10\u00b2',u'7.5 x 10\u00b2',u'10\u00b3',u'2.5 x 10\u00b3',u'5 x 10\u00b3',u'7.5 x 10\u00b3',u'10\u2074',u'2.5 x 10\u2074',u'5 x 10\u2074',u'7.5 x 10\u2074',u'10\u2075',u'2 x 10\u2075']
        
        flag2=ticks
        ticks5 =  [1e-8,5e-8,1e-7,5e-7,1e-6,5e-6,1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2]
        ticks5 = np.arange(0,650,50)
        #ticks5 =  [1e-2,5e-2,1e-1,5e-1,1e0,5e0,1e1,5e1,1e2,5e2,1e3,5e3]
        ticks5 =  [1e-2,5e-2,1e-1,5e-1,1e0,5e0,1e1,5e1,1e2,5e2,1e3]
        #ticks5 =  [1e-1,5e-1,1e0,5e0,1e1,5e1,1e2,5e2,1e3,5e3]
        ticks5 = [1e2,5e2,1e3,5e3,1e4,5e4,1e5]
       #ticks5_label = ['1e-6','5e-6','1e-5','5e-5','1e-4','5e-4','1e-3','5e-3','1e-2']
        ticks5_label = [str(o) for o in ticks5]
        #plt.contourf(x,y,data,ticks3,vmin=ticks3[0],vmax=ticks3[len(ticks3)-1],cmap=plt.cm.jet,resolution='c')
        #plt.contourf(x,y,data,ticks4,vmin=0,vmax=30,cmap=plt.cm.jet,resolution='c', format = '%.0e')
        #plt.contourf(x,y,data,ticks5,norm=colors.LogNorm(vmin=1e-2,vmax=5e3),cmap='RdYlBu_r',resolution='c', format = '%.0e')
        plt.contourf(x,y,data,ticks5,norm=colors.LogNorm(vmin=1e2,vmax=1e5),cmap='RdYlBu_r',resolution='c', format = '%.0e')
        #plt.contourf(x,y,data,ticks5,norm=colors.LogNorm(vmin=1e-2,vmax=5e3),cmap='RdYlBu_r',resolution='c', format = '%.0e')
        #plt.contourf(x,y,data,ticks5,norm=colors.LogNorm(vmin=np.min(ticks5),vmax=np.max(ticks5)),cmap='RdYlBu_r',resolution='c', format = '%.0e')
        


        #plt.plot(reduced_lat,reduced_trop_height,'k--')
        plt.plot(lat1,trop_height.data,'k--')
        #plt.contourf(x,y,data,ticks5,vmin=0,vmax=2200,cmap='RdYlBu_r',resolution='c', format = '%.0e')
        
        
        
        #plt.contourf(x,y,data,ticks3,norm=colors.LogNorm(vmin=pow(10,1),vmax=pow(10,6)),cmap=plt.cm.jet,resolution='c', format = '%.0e')
        #plt.clim((pow(10,3),pow(10,7)))
        mn=0.1
        mx=100000
        md=(mx-mn)/2
        print("\nThe maximum value is      = ", np.max(data))
        print("\nThe minimum value is      = ", np.min(data))
        #print '\nThe mean value of particle number at 11.5km  = ', np.mean(data[39,:]),'\n'
        
        formatter = LogFormatter(10, labelOnlyBase=False)
        cbar=plt.colorbar()
        #cbar=plt.colorbar(ticks=ticks3, format=formatter)
        cbar.set_ticks(ticks5)



        cbar.set_ticklabels(ticks5_label)
        
        plt.xlabel('latitude')
        plt.ylabel('altitude (km)')
        plt.title('Ion concentration (per cm3)')
        #plt.title('SO'+u'\u2082'+' mixing ratio(pptv)')
        plt.savefig('so2_vertical_baseline_my_paper.eps',dpi = 500)
        plt.show()
        
        #print new1       
        


