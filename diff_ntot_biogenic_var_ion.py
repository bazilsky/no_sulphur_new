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
mpath1='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz147/All_months/' # with biogenic nucleation ON
mpath2='/group_workspaces/jasmin2/asci/eeara/model_runs/u-by921/All_months/'

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
    print(model_n)
    #model
    model_n.long_name='Particle Number Concentration at STP (cm-3)'
    model_n.units='cm-3'
    #take longitudenale mean 
    long_mean=model_n.collapsed('longitude',iris.analysis.MEAN)
    slice1=long_mean
    return slice1

def file_save(slice1,num,alt):
    slice1=slice1.collapsed('longitude',iris.analysis.MEAN)
    data=slice1.data
    
   # lon=slice1.coord('longitude').points
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
#..............i
    
    

    fig = plt.figure(num=None, figsize=(12, 8) )
    m = Basemap(projection='tmerc',llcrnrlat=-80,urcrnrlat=80, resolution ='c')
    
    #m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,resolution='c')
    
    x,y=m(*np.meshgrid(new_lon_final,lat))
    m.drawcoastlines()
    m.drawmapboundary(fill_color='lightblue')

    m.pcolormesh(x,y,data_final,norm=colors.LogNorm(vmin=data.min(),vmax=data.max()),cmap=plt.cm.jet)
    m.colorbar(location='right')
    plt.title('JANE(So2 correction) - Number Conc (cm-3) || ALTITUDE = '+str(alt)+'km');

    file_name='file'+str(num)+'.png'
    #plt.show()
    plt.savefig(file_name)
    #return data,lat,new_lon 
    
    
    
    #...........................................................
def file_save2(slice1,num,alt):
    plt.figure()
    qplt.contourf(slice1,25)
    plt.gca().coastlines()
    plot_name='(new-old/old)*100--uax424 (%) || ALT = '+str(alt)+'km'
    print('\n',plot_name)
    plt.title(plot_name)
    file_name='file'+str(num)+'.png'
    plt.show()
    #plt.savefig(file_name)
    
    
    #plt.show()
    #return slice1
    #model_n = model_nuc_stp+model_ait_stp+model_aitins_stp+model_acc_stp+model_cor_stp
   # model_n=model_nuc+model_ait+model_aitins+model_acc+model_cor



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
    
    cn_file = 'All_months_m01s38i437_CN__NUMBER_CONCENTRATION____________.nc'
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
        new1=new1[0:66,:]
        new2=new2[0:66,:]
 #       old1=get_model_data(i,alt_data[i],i,mpath2)
 #       diff=((new1-old1)/old1)*100
      #  data, lat, lon = file_save(new1,i,alt_data[i])
        
        #qplt.contourf(new1)
        #plt.show()
        diff = new1.data-new2.data # biogenic - baseline simulation
         
        data=new1.data
        data = diff.data
        lnum1=new1.coord('model_level_number').points
        
        #data= np.where(data>0,data,1e-1)
        #data= np.where((data<-1e1 & data>0 ,data,1e-1))
        """ 
        for q in range(len(data[:,1])):
            for w in range(len(data[1,:])):
                dummy = data[q,w]
                if dummy>=-1e1 and dummy <0.0:
                    data[q,w] = 1e-1
                elif dummy>=-1e2 and dummy <-1e1:
                    data[q,w] = 1e-2
                elif dummy>=-1e3 and dummy <-1e2:
                    data[q,w] = 1e-3
                elif dummy>=-1e4 and dummy <-1e3:
                    data[q,w] = 1e-4
                elif dummy>=-1e5 and dummy <-1e4:
                    data[q,w] = 1e-5
        """
                    
           
            
        lnum_alt=alt_data[0:66]
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
        flag2=ticks
        
        plt.contourf(x,y,data,ticks4,norm=colors.SymLogNorm(linthresh = 1e1,linscale = 1e0,vmin=-1e3,vmax=1e3),cmap='RdYlBu_r', format = '%.0e')
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
        cbar.set_ticks(ticks4)


        #ticks3=['1','5','10','20','50','1E2','2e2','5e2','1e3','2e3','5e3','1e4','3e4','6e4','1e5','2e5']

        cbar.set_ticklabels(ticks4_label)
        #cbar.set_ticklabels(ticks3_label)
        
        #plt.clim((pow(10,3),pow(10,7)))
        plt.xlabel('latitude')
        plt.ylabel('altitude (km)')
        #plt.title('Total Particle number concentration (cm'+u'\u207B\u00B3'+')')
        plt.title('Change in Particle number concentration (cm'+u'\u207B\u00B3'+')- variable ion concentration' )
        plt.savefig('ntot_vertical_profile.eps',dpi = 500)
        plt.show()
        
        print(new1)       
        


