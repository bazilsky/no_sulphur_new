import numpy as np
import iris
import matplotlib.pyplot as plt
import matplotlib

#files_path = '/home/users/eeara/emis_files/'
#files_path = '/home/users/eeara/CARIBIC_CODE/'
#so2_file_1 = 'SO2_all_low_anthropogenic_2014_time_slice.nc'

o3_file = 'All_months_m01s34i001_mass_fraction_of_ozone_in_air.nc'


so2_file_1 = '/group_workspaces/jasmin2/asci/eeara/model_runs/u-bt717/All_months/'+o3_file # new run
so2_file_2 = '/group_workspaces/jasmin2/asci/eeara/model_runs/u-bt676/All_months/'+o3_file # baseline simulation


#so2_file_2 = 'ukca_emiss_SO2_nat_grg.nc' # this file has only one time dimension
#dms_file_1 = 'DMS_biomass_low_2014_time_slice.nc'
#dms_file_2 = 'DMS_land_spiro1992.nc'

legend_id = ['SO2_all_low','DMS_biomass_low','DMS_land_spiroo']
legend_id = ['Sulphurless Planet ','baseline']

cube1 = iris.load(so2_file_1)[0]
cube2 = iris.load(so2_file_2)[0]
#cube2 = iris.load(files_path+so2_file_2)[0]
#cube3 = iris.load(files_path+dms_file_1)[0]
#cube4 = iris.load(files_path+dms_file_2)[0]

def cube_mean(cube):
    cube = cube.collapsed('model_level_number',iris.analysis.MEAN)
    #cube = cube[:,12,:,:]
    cube = cube.collapsed('latitude',iris.analysis.MEAN)
    cube = cube.collapsed('longitude',iris.analysis.MEAN)
    cube = cube*(29.0/48.0)/1e-12/1e6
    return cube 

dat1 = cube_mean(cube1)
dat2 = cube_mean(cube2)
#dat3 = cube_mean(cube3)
#dat4 = cube_mean(cube4)

time_val = np.arange(1,13,1)
month=['J','F','M','A','M','J','J','A','S','O','N','D']
matplotlib.style.use('ggplot')
#plt.figure(figsize=(12,12),dpi = 500)
plt.figure()
plt.plot(time_val,dat1.data,'ro-')
plt.plot(time_val,dat2.data,'bo-')
#plt.plot(time_val,dat3.data,'go-')
#plt.plot(time_val,dat4.data,'ko-')
plt.legend(legend_id)
#plt.yscale('log')
plt.xticks(time_val,month)
plt.xlabel('Month')
plt.ylabel('O3 concentration (ppb)')
plt.title('Seasonal cycle Ozone concentration')
plt.savefig('O3_seasonal_nosulphur_2.eps',dpi = 500)
"""
plt.figure()

plt.plot(time_val,dat1.data,'ro-')
plt.title('SO2 emission flux (kg/m2/s)')
plt.xticks(time_val,month)
plt.legend(['SO2_all_low'])
plt.xlabel('Month')
plt.ylabel('Emission flux (kg/m2/s)')
plt.savefig('so2_seasonal.eps',dp1=500)
"""


plt.show()

