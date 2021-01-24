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
    
    cn_file = 'All_months_m01s38i437_CN__NUMBER_CONCENTRATION____________.nc'
    nuc_file = 'All_months_m01s38i504_number_concentration_nucleation_mode.nc'
    aitsol_file = 'All_months_m01s38i505_number_concentration_aitken_mode.nc'
    acc_file = 'All_months_m01s38i506_number_concentration_accumulation_mode.nc'
    cor_file = 'All_months_m01s38i507_number_concentration_coarse_mode_insoluble.nc'
    aitinsol_file = 'All_months_m01s38i508_number_concentration_aitken_insoluble.nc'

    #cn_file = nuc_file

     
    #if i==0:
    if i%3==0:
        #new1=get_model_data(i,alt_data[i],i,mpath1)
        #new2=get_model_data(i,alt_data[i],i,mpath2)
        
        new1 = iris.load(mpath1+cn_file)[0]
        new2 = iris.load(mpath2+cn_file)[0]
       
        new1 = new1.collapsed('time',iris.analysis.MEAN)
        new1 = new1.collapsed('longitude',iris.analysis.MEAN)
        new2 = new2.collapsed('time',iris.analysis.MEAN)
        new2 = new2.collapsed('longitude',iris.analysis.MEAN)

         

        print('this is longitudenal mean\n\n', new1,'\n\n')
        new1=new1[0:level_no,:]
        new2=new2[0:level_no,:]
 #       old1=get_model_data(i,alt_data[i],i,mpath2)
 #       diff=((new1-old1)/old1)*100
      #  data, lat, lon = file_save(new1,i,alt_data[i])
        
        #qplt.contourf(new1)
        #plt.show()
        diff = new1.data-new2.data # biogenic - baseline simulation
        diff = (new1.data-new2.data)*100.0/new2.data # biogenic - baseline simulation
         
        data=new1.data
        data = diff.data
        lnum1=new1.coord('model_level_number').points
        
        #data= np.where(data>0,data,1e-1)
        #data= np.where((data<-1e1 & data>0 ,data,1e-1))
           
            
        lnum_alt=alt_data[0:level_no]
        lat1=new1.coord('latitude').points
        """
        x,y=np.meshgrid(lat1,lnum_alt)
        plt
        #fig, axes= pl.plot(1, figsize=(12, 8))
       

        """
        x,y=np.meshgrid(lat1,lnum_alt)
        #plt.figure()
        #im=plt.imshow(data)   
        #plt.contourf(x,y,data)
        ticks=[1,5,10,20,50,100,200,500,1000,2000,5000,10000,30000,60000,100000,200000]
        #ticks3=[1e0,5e0,1e1,2e1,5e1,1e2,2e2,5e2,1e3,2e3,5e3,1e4,3e4,6e4,1e5,2e5]
        #ticks3       = [1e0,1e1,5e1,1e2,2.5e2,5e2,7.5e2,1e3,2.5e3,5e3,7.5e3,1e4,2.5e4,5e4,7.5e4,1e5,2.5e5]
        #ticks3       = [1e0,1e1,5e1,1e2,2.5e2,5e2,7.5e2,1e3,2.5e3,5e3,7.5e3,1e4,2.5e4,5e4,7.5e4,1e5,2.5e5]
        ticks3       = [1e-2,1e1,5e1,1e2,2.5e2,5e2,7.5e2,1e3,2.5e3,5e3,7.5e3,1e4,2.5e4,5e4,8e4]
        #ticks3_label = ['1','10','50',u'10\u00b2',u'2.5 x 10\u00b2',u'5 x 10\u00b2',u'7.5 x 10\u00b2',u'10\u00b3',u'2.5 x 10\u00b3',u'5 x 10\u00b3',u'7.5 x 10\u00b3',u'10\u2074',u'2.5 x 10\u2074',u'5 x 10\u2074',u'7.5 x 10\u2074',u'10\u2075',u'2 x 10\u2075']
        ticks3_label = ['negative','10','50',u'10\u00b2',u'2.5 x 10\u00b2',u'5 x 10\u00b2',u'7.5 x 10\u00b2',u'10\u00b3',u'2.5 x 10\u00b3',u'5 x 10\u00b3',u'7.5 x 10\u00b3',u'10\u2074',u'2.5 x 10\u2074',u'5 x 10\u2074',u'8 x 10\u2074']
        #ticks3_label = [str(o) for o in ticks3]
        ticks3       = [1e-5,1e-4,1e-3,1e-2,1e-1,1e1,5e1,1e2,2.5e2,5e2,7.5e2,1e3,2.5e3,5e3,7.5e3,1e4,2.5e4,5e4,8e4]
        #ticks3_label = ['1','10','50',u'10\u00b2',u'2.5 x 10\u00b2',u'5 x 10\u00b2',u'7.5 x 10\u00b2',u'10\u00b3',u'2.5 x 10\u00b3',u'5 x 10\u00b3',u'7.5 x 10\u00b3',u'10\u2074',u'2.5 x 10\u2074',u'5 x 10\u2074',u'7.5 x 10\u2074',u'10\u2075',u'2 x 10\u2075']
        ticks3_label = ['-1e-5','-1e-4','-1e-3','-1e-2','-1e-1','10','50',u'10\u00b2',u'2.5 x 10\u00b2',u'5 x 10\u00b2',u'7.5 x 10\u00b2',u'10\u00b3',u'2.5 x 10\u00b3',u'5 x 10\u00b3',u'7.5 x 10\u00b3',u'10\u2074',u'2.5 x 10\u2074',u'5 x 10\u2074',u'8 x 10\u2074']
        flag2=[1,5,10,20,50,100,200,500,1000,2000,5000,10000,30000,60000,100000,200000]
        #plt.contourf(x,y,data,cmap=,resolution='c')
        
       # ticks=[1e0,5e0,1e1,2e1,5e1,1e2,2e2,5e2,1e3,2e3,5e3,1e4,3e4,6e4,1e5]
        ticks4 = [-1e3,-1e2,-1e1,1e1,1e2,1e3] 
        ticks4 = [-1e3,-5e2,-1e2,-5e1,-1e1,1e1,5e1,1e2,5e2,1e3] 
        ticks4_label = [str(o) for o in ticks4]
        
        ticks5 = [-1e5,-5e4,-1e4,-5e3,-1e3,-5e2,-1e2,-5e1,-1e1,1e1]
        ticks5 = [-1e2,-9.5e1,-9e1,-8.5e1,-8e1,-6e1,-4e1,-2e1,0]
        #ticks5 = [-100,-80,-60,-40,-20,0-5e4,-1e4,-5e3,-1e3,-5e2,-1e2,-5e1,-1e1,1e1]
        ticks5 = np.arange(-100,1,10)
        #ticks5 = [-1e2,1e2]
        ticks5_label = [str(o)+'%' for o in ticks5]
        flag2=ticks
        
        #plt.contourf(x,y,data,ticks4,norm=colors.SymLogNorm(linthresh = 1e1,linscale = 1e0,vmin=-1e3,vmax=1e3),cmap='RdYlBu_r', format = '%.0e')
        #plt.contourf(x,y,data,ticks5,norm=colors.SymLogNorm(linthresh = 10,linscale = 0.1,vmin=-1e2,vmax=0),cmap='RdYlBu_r', format = '%.0e')
        plt.contourf(x,y,data,ticks5,vmin=np.min(ticks5),vmax=np.max(ticks5),cmap='RdYlBu_r', format = '%.0e')
        #plt.contourf(x,y,data,ticks3,norm=colors.LogNorm(vmin=1e-5,vmax=8e4),cmap='RdYlBu_r', format = '%.0e')
        #plt.contourf(x,y,data,ticks3,norm=colors.LogNorm(vmin=pow(10,1),vmax=pow(10,6)),cmap=plt.cm.jet,resolution='c', format = '%.0e')
        #plt.clim((pow(10,3),pow(10,7)))
        plt.plot(lat1,trop_height.data,'k--')
        mn=0.1
        mx=100000
        md=(mx-mn)/2
        print('\nThe maximum value of particle number is      = ', np.max(data))
        print('\nThe mean value of particle number at 11.5km  = ', np.mean(data[39,:]),'\n')
        
        #plt.contourf(x,y,data,cmap=,resolution='c')
        formatter = LogFormatter(10, labelOnlyBase=False)
        cbar=plt.colorbar()
        #cbar=plt.colorbar(ticks=ticks3, format=formatter)
        #cbar.set_ticks(ticks4)
        cbar.set_ticks(ticks5)


        #ticks3=['1','5','10','20','50','1E2','2e2','5e2','1e3','2e3','5e3','1e4','3e4','6e4','1e5','2e5']

        cbar.set_ticklabels(ticks5_label)
        #cbar.set_ticklabels(ticks3_label)
        #plt.yticks(['1','10']) 
        #plt.clim((pow(10,3),pow(10,7)))
        plt.xlabel('latitude')
        plt.ylabel('altitude (km)')
        #plt.title('Total Particle number concentration (cm'+u'\u207B\u00B3'+')')
        plt.title('Change in Particle number concentration (cm'+u'\u207B\u00B3'+') - PD' )
        plt.savefig('ntot_vertical_profile.eps',dpi = 500)
        plt.show()
        
        print(new1)       
        


