import iris,glob,datetime, time,re,sys
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


iris.FUTURE.cell_datetime_objects=True


trial1=iris.cube.CubeList()
trial2=iris.cube.CubeList()

path = '/group_workspaces/jasmin2/gassp/eeara/CARIBIC/'#ANANTH

#mpath1='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bf829/L1/'
#mpath1='/group_workspaces/jasmin2/asci/eeara/model_runs/u-by920/L1/' # with biogenic nucleation ON
#mpath2='/group_workspaces/jasmin2/asci/eeara/model_runs/u-by921/L1/'

#mpath1='/group_workspaces/jasmin2/asci/eeara/model_runs/u-by920/All_months/' # with biogenic nucleation ON .. variable ion concentration
mpath1='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz050/All_months/' # sulphurless planet
mpath2='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz049/All_months/' # baseline simulation

alt_path='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bf829/L1/'
#alt_path='/gws/nopw/j04/gassp/hgordon/u-be424-noColinFix/L1/'
#mpath2='/group_workspaces/jasmin2/gassp/eeara/model_runs/u-bc244/L1/'
cutoff=2 # this is a radius...12nm is a diameter!!

altitude_path=alt_path+'L1_rad_accsol_Radius_of_mode_accsol.nc'
cube=iris.load(altitude_path)
cube=cube[0]
alt_data=cube.coord('altitude').points
alt_data=alt_data[:,72,96]
alt_data=alt_data/1000


sigma=[1.59,1.59,1.40,2.0,1.59,1.59,2.0]

pref=1.013e5
tref=293.0
zboltz=1.3807e-23
staird=pref/(tref*287.058) 




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

level_no = 50
#level_no = 66
#level_n0 = 33 # this is altitude 8.1km


#for i in range(1):
flag=[30]
for i in flag:
#for i in range(len(alt_data)):
    print('\n File STARTED ')
    filepath = '/group_workspaces/jasmin2/asci/eeara/model_runs/u-bs405/All_months/'
    trop_height = iris.load(filepath+'All_months_m01s30i453_Height_at_Tropopause_Level__________.nc')[0]
    trop_height = trop_height.collapsed('time',iris.analysis.MEAN)
    trop_height = trop_height.collapsed('longitude',iris.analysis.MEAN)
    trop_height = trop_height/1000.0
    
    cn_file = 'All_months_m01s00i254_QCL_AFTER_TIMESTEP__________________.nc'
    if i%3==0:
        
        new1 = iris.load(mpath1+cn_file)[0]
        new2 = iris.load(mpath2+cn_file)[0]
       
        new1 = new1.collapsed('time',iris.analysis.MEAN)
        new1 = new1.collapsed('longitude',iris.analysis.MEAN)
        new2 = new2.collapsed('time',iris.analysis.MEAN)
        new2 = new2.collapsed('longitude',iris.analysis.MEAN)

         

        print('this is longitudenal mean\n\n', new1,'\n\n')
        new1=new1[0:level_no,:]
        new2=new2[0:level_no,:]
        diff = new1.data-new2.data # biogenic - baseline simulation
        #diff = (new1.data-new2.data)*100/new2.data # biogenic - baseline simulation
         
        data=new1.data
        data = diff.data
        lnum1=new1.coord('model_level_number').points
        
        lnum_alt=alt_data[0:level_no]
        lat1=new1.coord('latitude').points
        x,y=np.meshgrid(lat1,lnum_alt)
        
        ticks5 = [-1e5,-5e4,-1e4,-5e3,-1e3,-5e2,-1e2,-5e1,-1e1,1e1]
        #ticks5 = [-100,-80,-60,-40,-20,0-5e4,-1e4,-5e3,-1e3,-5e2,-1e2,-5e1,-1e1,1e1]
        ticks5 = np.arange(-0.05,0.05,0.01)
        ticks5 = [-0.05,-0.04,-0.03,-0.02,-0.01,0,0.01,0.02,0.03,0.04,0.05]
        ticks5 =[-1e-4,-1e-6,-1e-8,-1e-10,-1e-12,1e-12,1e-10,1e-8,1e-6,1e-4] 
        #ticks5_label = [str(o) for o in ticks5]
        
        #plt.contourf(x,y,data,ticks4,norm=colors.SymLogNorm(linthresh = 1e1,linscale = 1e0,vmin=-1e3,vmax=1e3),cmap='RdYlBu_r', format = '%.0e')
        plt.contourf(x,y,data,ticks5,norm=colors.SymLogNorm(linthresh = 1e-12,linscale = 1e-13,vmin=-1e-4,vmax=1e-4),cmap='RdYlBu_r', format = '%.0e')
        #plt.contourf(x,y,data,ticks5,vmin=np.min(ticks5),vmax=np.max(ticks5),cmap='RdYlBu', format = '%.0e')
        #plt.contourf(x,y,data,ticks3,norm=colors.LogNorm(vmin=1e-5,vmax=8e4),cmap='RdYlBu_r', format = '%.0e')
        #plt.contourf(x,y,data,ticks3,norm=colors.LogNorm(vmin=pow(10,1),vmax=pow(10,6)),cmap=plt.cm.jet,resolution='c', format = '%.0e')
        #plt.clim((pow(10,3),pow(10,7)))
        plt.plot(lat1,trop_height.data,'k--')
        mn=0.1
        mx=100000
        md=(mx-mn)/2
        print('\nThe maximum value of particle number is      = ', np.max(data))
        print('\nThe minimum value of particle number   = ',       np.min(data),'\n')
        print('\nThe mean value of particle number at 11.5km  = ', np.mean(data),'\n')
        
        #plt.contourf(x,y,data,cmap=,resolution='c')
        formatter = LogFormatter(10, labelOnlyBase=False)
        cbar=plt.colorbar()
        #cbar=plt.colorbar(ticks=ticks3, format=formatter)
        #cbar.set_ticks(ticks4)
        cbar.set_ticks(ticks5)



        #cbar.set_ticklabels(ticks5_label)
        #cbar.set_ticklabels(ticks3_label)
        #plt.yticks(['1','10']) 
        #plt.clim((pow(10,3),pow(10,7)))
        plt.xlabel('latitude')
        plt.ylabel('altitude (km)')
        #plt.title('Total Particle number concentration (cm'+u'\u207B\u00B3'+')')
        plt.title('Change in liquid water content (kg/kg_air)  - PD' )
        plt.savefig('ntot_vertical_profile.eps',dpi = 500)
        plt.show()
        
        print(new1)       
        


