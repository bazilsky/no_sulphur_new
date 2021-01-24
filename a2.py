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


#mpath1='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz871/All_months/' # sulphurless with biogenic nucleation ON
#mpath2='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz050/All_months/' # sulphurless planet

mpath1='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz983/2017jan/' # sulphurless with biogenic nucleation ON
mpath2='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz980/2017jan/' # sulphurless planet

mpath1='/group_workspaces/jasmin2/asci/eeara/model_runs/u-ca206/2017jan/' # sulphurless with biogenic nucleation ON
#mpath2='/group_workspaces/jasmin2/asci/eeara/model_runs/u-ca179/2017jan/' # new basline sulphurless planet
mpath2='/group_workspaces/jasmin2/asci/eeara/model_runs/u-ca179/2017jan/' # sulphurless planet

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


def get_model_data(indx,alt,num,mpath): #add variable for cutoff and do logcumtor (easy!)

    #new_mpath = mpath.copy()
    mpath = mpath[:-11]+'L1/'

    model_aird = iris.load(mpath+'L1_air_density_Density of air.nc')
    model_aird = model_aird[0]
    #model_aird = iris.load_cube(mpath+'L1_air_density_Density of air.nc') #kgm-3
    
    
    #model_acc = iris.load(mpath+'L1_n_accsol_number_of_particles_per_air_molecule_of_soluble_accumulation_mode_aerosol_in_air.nc')#particles/m3
    #model_nuc = iris.load(mpath+'L1_n_nucsol_number_of_particles_per_air_molecule_of_soluble_nucleation_mode_aerosol_in_air.nc')
    #model_aitins = iris.load(mpath+'L1_n_aitins_number_of_particles_per_air_molecule_of_insoluble_aitken_mode_aerosol_in_air.nc')
    #model_ait = iris.load(mpath+'L1_n_aitsol_number_of_particles_per_air_molecule_of_soluble_aitken_mode_aerosol_in_air.nc')
    #print model_ait,'\n'
    #model_cor=iris.load(mpath+'L1_n_corsol_number_of_particles_per_air_molecule_of_soluble_coarse_mode_aerosol_in_air.nc')
    
    model_acc = iris.load(mpath+'L1_n_accsol_number_*')#particles/m3
    model_nuc = iris.load(mpath+'L1_n_nucsol_number_*')
    model_aitins = iris.load(mpath+'L1_n_aitins_number_*')
    model_ait = iris.load(mpath+'L1_n_aitsol_number_*')
    #print model_ait,'\n'
    model_cor=iris.load(mpath+'L1_n_corsol_number_*')
    
    
    #model_cor = iris.load(mpath,'L1_n_corsol_number_of_particles_per_air_molecule_of_soluble_coarse_mode_aerosol_in_air.nc')
    #print model_cor,'\n'
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
    #print model_n
    model_n = model_n.collapsed('time',iris.analysis.MEAN)
    model_n = model_n/1000000
    #print(model_n)
    #model
    model_n.long_name='Particle Number Concentration at STP (cm-3)'
    model_n.units='cm-3'
    #take longitudenale mean 
    long_mean=model_n.collapsed('longitude',iris.analysis.MEAN)
    slice1=long_mean
    return slice1

####################################################################
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
    
    #cn_file = 'All_months_m01s38i437_CN__NUMBER_CONCENTRATION____________.nc'
    cn_file = '2017jan_m01s38i437_CN__NUMBER_CONCENTRATION____________.nc'
    #if i==0:
    if i%3==0:
        #new1=get_model_data(i,alt_data[i],i,mpath1)
        #new2=get_model_data(i,alt_data[i],i,mpath2)
         
        #new1 = iris.load(mpath1+cn_file)[0]
        #new2 = iris.load(mpath2+cn_file)[0]
        
        secorg_path='/gws/nopw/j04/asci/eeara/model_runs/u-cb395/'
        secorg_path2='/gws/nopw/j04/asci/eeara/model_runs/u-cb448/2017jan/' # only secorg monthly output
        
        secorgfromoho3_393 = 'secorgfromoho3_393.nc'
        secorgfromoho3_395 = 'secorgfromoho3_395.nc'
        sec_org_dummy = 'sec_org_dummy.nc'
        
        sec_org_dummy_2 = '2017jan_m01s38i439_CCN_NO._CONCENTRN._(ACC+COR+AIT>25r).nc'
        secorgfromoho3_2 = '2017jan_m01s38i440_CCN_NO._CONCENTRN._(ACC+COR+AIT>35r).nc' 

        new1 = iris.load(secorg_path+secorgfromoho3_393)[0]
        new2 = iris.load(secorg_path+secorgfromoho3_395)[0]
        new3 = iris.load(secorg_path+sec_org_dummy)[0]
        
        new3_dummy = iris.load(secorg_path2+sec_org_dummy_2)[0]
        new4 = iris.load(secorg_path2+secorgfromoho3_2)[0]
        
        
        o3_oh_path = '/gws/nopw/j04/asci/eeara/model_runs/u-cb393/'

        o3_file = 'o3.nc'
        oh_file = 'oh.nc'
        o3_file_new = 'o3_calcnucrate.nc'
        oh_file_new = 'oh_calcnucrate.nc'
        

        o3 = iris.load(o3_oh_path+o3_file)[0]
        oh = iris.load(o3_oh_path+oh_file)[0]
        oh_new = iris.load(o3_oh_path+oh_file_new)[0]
        o3_new = iris.load(o3_oh_path+o3_file_new)[0]


        #diff_dummy = new2 - new1
        #diff_dummy = new2/new1
        #diff_dummy = new3-new1
        #diff_dummy = new4-new3_dummy
        #diff_dummy = new3_dummy-new4
        #diff_dummy = new3_dummy/new4
        #diff_dummy = new4 
        #diff_dummy_2 = o3_new.data[2,:,:,:]/o3.data[2,:,:,:]
        diff_dummy_2 = oh_new.data[2,:,:,:] / oh.data[2,:,:,:]
        
       # diff_dummy_2 = oh_new.data[2,:,:,:] - oh.data[2,:,:,:]
        
        dummy_cube = new1.copy()[0,:,:,:]
        dummy_cube.data = diff_dummy_2
        diff_dummy = dummy_cube.copy()
        diff_dummy = diff_dummy.collapsed('longitude',iris.analysis.MEAN)
        #diff_dummy = diff_dummy.collapsed('time',iris.analysis.MEAN)

        
        new1 = new1.collapsed('longitude',iris.analysis.MEAN)
        new1 = new1.collapsed('time',iris.analysis.MEAN)
        new2 = new2.collapsed('longitude',iris.analysis.MEAN)
        new2 = new2.collapsed('time',iris.analysis.MEAN)
        
         

        #print('this is longitudenal mean\n\n', new1,'\n\n')
        #new1=new1[0:66,:]
        #new2=new2[0:66,:]
        
        #qplt.contourf(new1)
        #plt.show()
        #diff = new2.data-new1.data # biogenic - baseline simulation
        diff = diff_dummy.data[0:52,:] 
        #data=new1.data
        data = diff.data
        lnum1=new1.coord('model_level_number').points
        
        #data= np.where(data>0,data,1e-1)
        #data= np.where((data<-1e1 & data>0 ,data,1e-1))
                    
           
            
        lnum_alt=alt_data[0:52]
        #lnum_alt=alt_data[0:66]
        lat1=new1.coord('latitude').points
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
        ticks4 = [-1e4,-5e3,-1e3,-5e2,-1e2,-5e1,-1e1,1e1,5e1,1e2,5e2,1e3,5e3,1e4] 
        ticks4 = [-1e7,-1e6,-1e5,-1e4,-1e3,-1e2,-1e1,1e1,1e2,1e3,1e4,1e5,1e6,1e7] 
        #ticks4 = [-1e200,-1e100,-1e50,-1e30,-1e10,-1e1,1e1,1e10,1e30,1e50,1e100,1e200] 
        ticks4 = np.arange(0,1.1,0.1)
        ticks4 = [0.1,0.4,0.8,1e0,1e1,1e2,1e3,1e4,1e5]
        #ticks4 = [-1,-0.7,-0.4,-0.1,0.1,0.4,0.7,1]
        ticks4_label = [str(o) for o in ticks4]
        flag2=ticks
        
        #plt.contourf(x,y,data,ticks4,norm=colors.SymLogNorm(linthresh = 1e1,linscale = 1e0,vmin=-1e7,vmax=1e7),cmap='RdYlBu_r', format = '%.0e')
        #plt.contourf(x,y,data,ticks4,vmin = np.min(ticks4), vmax = np.max(ticks4) ,cmap='RdYlBu_r', format = '%.0e')
        plt.contourf(x,y,data,ticks4,norm=colors.LogNorm(vmin=0.1,vmax=1e5),cmap='RdYlBu_r', format = '%.0e')
        #plt.contourf(x,y,data,ticks4,norm=colors.SymLogNorm(linthresh = 1e1,linscale = 1e0,vmin=-1,vmax=1),cmap='RdYlBu_r', format = '%.0e')
       # plt.contourf(x,y,data,ticks4,norm=colors.SymLogNorm(linthresh = 1e1,linscale = 1e0,vmin=-1e200,vmax=1e200),cmap='RdYlBu_r', format = '%.0e')
        #plt.clim((pow(10,3),pow(10,7)))
        plt.plot(lat1,trop_height.data,'k--')
        mn=0.1
        mx=100000
        md=(mx-mn)/2
        print('\nThe maximum value of particle number is      = ', np.max(data))
        print('\nThe minimum value of particle number is      = ', np.min(data))
        print('\nThe mean value of particle number is         = ', np.mean(data))
        print('\nThe max value at surface is                  = ', np.max(data[0,:]))
        print('\nThe min value of surface is                  = ', np.min(data[0,:]))
        print('\nThe mean value of surface is                 = ', np.mean(data[0,:]))
        
        #plt.contourf(x,y,data,cmap=,resolution='c')
        formatter = LogFormatter(10, labelOnlyBase=False)
        cbar=plt.colorbar()
        #cbar=plt.colorbar(ticks=ticks3, format=formatter)
        cbar.set_ticks(ticks4)


        #ticks3=['1','5','10','20','50','1E2','2e2','5e2','1e3','2e3','5e3','1e4','3e4','6e4','1e5','2e5']

        cbar.set_ticklabels(ticks4_label)
        #cbar.set_ticklabels(ticks3_label)
        
        #plt.clim((pow(10,3),pow(10,7)))
        plt.xlabel('latitude')
        plt.ylabel('altitude (km)')
        #plt.title('Total Particle number concentration (cm'+u'\u207B\u00B3'+')')
        plt.title('oh_calcnucrate/oh_chem')
        #plt.title('(sec_org) concentration molecules/cc')
        #plt.title('(HOM) concentration molecules/cc')
        #plt.title('N_Total change (sulphurless+purebiogenic) - (sulphurless)(cm'+u'\u207B\u00B3'+')' )
        plt.savefig('ntot_vertical_profile.eps',dpi = 500)
        plt.show()
        
        #print(new1)       
        


