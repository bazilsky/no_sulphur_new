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
import time
import plot_func as pltfunc

iris.FUTURE.cell_datetime_objects=True


trial1=iris.cube.CubeList()
trial2=iris.cube.CubeList()

path = '/group_workspaces/jasmin2/gassp/eeara/CARIBIC/'#ANANTH

#mpath1='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bf829/L1/'
#mpath1='/group_workspaces/jasmin2/asci/eeara/model_runs/u-by920/L1/' # with biogenic nucleation ON
#mpath2='/group_workspaces/jasmin2/asci/eeara/model_runs/u-by921/L1/'

#mpath1='/group_workspaces/jasmin2/asci/eeara/model_runs/u-by920/All_months/' # with biogenic nucleation ON .. variable ion concentration
mpath1='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz050/' # sulphurless planet
mpath2='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz049/' # baseline simulation

#mpath3='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz050/L1/' # sulphurless planet
#mpath4='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz049/L1/' # baseline simulation


alt_path='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bf829/L1/'
#alt_path='/gws/nopw/j04/gassp/hgordon/u-be424-noColinFix/L1/'
#mpath2='/group_workspaces/jasmin2/gassp/eeara/model_runs/u-bc244/L1/'
cutoff=2 # this is a radius...12nm is a diameter!!

altitude_path=alt_path+'L1_rad_accsol_Radius_of_mode_accsol.nc'
cube=iris.load(altitude_path)
cube=cube[0]
alt_data=cube.coord('altitude').points
#alt_data=alt_data[:,72,96]
alt_data=alt_data/1000 # convert to km


sigma=[1.59,1.59,1.40,2.0,1.59,1.59,2.0]

pref=1.013e5
tref=293.0
zboltz=1.3807e-23
staird=pref/(tref*287.058) 

def calc_liq_water_path(qcl,mpath):
    qcl = qcl.collapsed('time',iris.analysis.MEAN)
    aird_file = 'L1_air_density_Density of air.nc'
    aird = iris.load(mpath+'L1/'+aird_file)[0]
    aird = aird.collapsed('time',iris.analysis.MEAN)
    
    val = 0
    alt_diff =[]
    alt_temp = alt_data * 1000.0
    
    qcl_temp = qcl[:84,:,:].data
    water_path = qcl_temp.copy()
    water_path[:,:,:] = 0
    aird_temp = aird[:84,:,:].data
    #for k in [0]:
    start_time = time.time()
    for k in range(len(alt_temp)-1):
        temp_diff = alt_temp[k+1,:,:]-alt_temp[k,:,:]
        water_path[k,:,:] = water_path[k,:,:] + qcl_temp[k,:,:]*aird_temp[k,:,:]*temp_diff
        #alt_diff = np.append(alt_diff,temp_diff)
           
    print('shape of alt_temp = ', np.shape(temp_diff))

    print('time to run the loop = ', (time.time()-start_time))
    #alt_file = iris.load(altitude_path)[0] 
    #alt_data = (alt_file.coord('altitude').points)
    water_path_final = np.sum(water_path,axis = 0) 
    return water_path_final


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
level_n0 = 33 # this is altitude 8.1km


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
    print('hello there') 
    cn_file = 'All_months_m01s00i254_QCL_AFTER_TIMESTEP__________________.nc'
    if i%3==0:
        
        qcl1 = iris.load(mpath1+'All_months/'+cn_file)[0]
        qcl2 = iris.load(mpath2+'All_months/'+cn_file)[0]
        print('function call') 
        new1 = calc_liq_water_path(qcl1,mpath1)
        new2 = calc_liq_water_path(qcl2,mpath2) # paths are to L1 forlder to get air density value

        #new1 = new1.collapsed('time',iris.analysis.MEAN)
        #new1 = new1.collapsed('longitude',iris.analysis.MEAN)
        #new2 = new2.collapsed('time',iris.analysis.MEAN)
        #new2 = new2.collapsed('longitude',iris.analysis.MEAN)

        print('this is longitudenal mean\n\n', new1,'\n\n')
        #new1=new1[0:level_no,:]
        #new2=new2[0:level_no,:]
        
        #diff = new1.data-new2.data # biogenic - baseline simulation
        diff = (new1-new2)*100/new2 # liquid water path (new-old)/(old)*100
        pltfunc.plot_diff_4(qcl1,diff,'percentage change in liquid water path',-100,101,'RdYlBu_r') 
        plt.show()


