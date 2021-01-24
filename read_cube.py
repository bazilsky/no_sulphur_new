import iris
import numpy as np


def cube_avg(cube):
    avg = cube.collapsed('latitude',iris.analysis.MEAN)
    return avg

filename = 'All_months_m01s38i437_CN__NUMBER_CONCENTRATION____________.nc'
mpath1='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz693/All_months/' # sulphurless planet + pure biogenic
#mpath1='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz603/All_months/' # sulphurless planet + pure biogenic
mpath2='/group_workspaces/jasmin2/asci/eeara/model_runs/u-bz050/All_months/' # just sulphurless plant 

cube1 = iris.load(mpath1+filename)[0]
cube2 = iris.load(mpath2+filename)[0]
lat = cube1.coord('latitude').points

#cube1 =cube_avg(cube1)
#cube2 =cube_avg(cube2)


diff = (cube1-cube2).data[:,0,:,:] 
#diff = (cube1-cube2).data 

#for i in range(12):
#    print('month',i)   
#    print('max =', np.max(diff[i,:,:,:]))
    #print('min =', np.min(diff[i,:,:,:]))
    #print('mean =', np.mean(diff[i,:,:,:]))

print('max =', np.max(diff))
print('min =', np.min(diff))
print('mean =', np.mean(diff))
#print('max =', np.max(diff[,:,:,:]))
