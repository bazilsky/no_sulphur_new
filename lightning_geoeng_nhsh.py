import numpy as np
import iris
import matplotlib.pyplot as plt
import matplotlib

files_path = '/group_workspaces/jasmin2/asci/eeara/model_runs/u-by412/All_months/' # baseline simulation
files_path2= '/group_workspaces/jasmin2/asci/eeara/model_runs/u-by413/All_months/'
so2_file_1 = 'All_months_m01s50i082_lightning_flashrate_per_column.nc'
#so2_file_2 = 'All_months_m01s50i082_lightning_flashrate_per_column.nc'
#so2_file_2 = 'ukca_emiss_SO2_nat_grg.nc' # this file has only one time dimension
#dms_file_1 = 'DMS_biomass_low_2014_time_slice.nc'
#dms_file_2 = 'DMS_land_spiro1992.nc'

legend_id = ['baseline_NH','baseline_SH','geoeng_NH','geoeng_SH']

cube1 = iris.load(files_path+so2_file_1)[0]
cube2 = iris.load(files_path2+so2_file_1)[0]
#cube3 = iris.load(files_path+dms_file_1)[0]
#cube4 = iris.load(files_path+dms_file_2)[0]

def cube_mean(cube):
    #cube = cube.collapsed('model_level_number',iris.analysis.MEAN)
    
    cube_sh = cube[:,0:72,:]
    cube_nh = cube[:,72:,:]
    cube_nh = cube_nh.collapsed('latitude',iris.analysis.MEAN)
    cube_nh = cube_nh.collapsed('longitude',iris.analysis.MEAN)
    cube_nh = cube_nh * (30*24*60) # calculate number of striked per month  
    
    cube_sh = cube_sh.collapsed('latitude',iris.analysis.MEAN)
    cube_sh = cube_sh.collapsed('longitude',iris.analysis.MEAN)
    cube_sh = cube_sh * (30*24*60) # calculate number of striked per month  
    return cube_sh,cube_nh 

dat1_sh,dat1_nh = cube_mean(cube1)
dat2_sh,dat2_nh = cube_mean(cube2)

time_val = np.arange(1,13,1)
month=['J','F','M','A','M','J','J','A','S','O','N','D']
matplotlib.style.use('ggplot')
plt.figure()
#plt.figure(figsize=(8,6), dpi = 300)
plt.plot(time_val,dat1_nh.data,'ro-')
plt.plot(time_val,dat1_sh.data,'bo-')
plt.plot(time_val,dat2_nh.data,'ro--')
plt.plot(time_val,dat2_sh.data,'bo--')
#plt.plot(time_val,dat3.data,'go-')
#plt.plot(time_val,dat4.data,'ko-')
plt.legend(legend_id)

#plt.yscale('log')
plt.xticks(time_val,month)
plt.xlabel('Month')
plt.ylabel('number of lightning strikes (per month)')
plt.title('Lightning seasonal cycle')
plt.savefig('lightning_seasonal_nosulphur_shnh_2.eps',dpi = 500)

"""
plt.figure(figsize = (6,6), dpi = 300)

plt.plot(time_val,dat1.data,'ro-')
plt.title('lightning seasonal cycle ')
plt.xticks(time_val,month)
plt.legend(['baseline'])
plt.xlabel('Month')
plt.ylabel('lightning flashes per month')
plt.savefig('lightning_seasonal_noforest_2.eps',dpi=500)
"""


plt.show()

