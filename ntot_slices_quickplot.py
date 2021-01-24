import matplotlib.pyplot as plt
import iris
import iris.quickplot as qplt

from new_colors import ruths_colors,ruths_colors_r

mpath1 ='/group_workspaces/jasmin2/asci/eeara/model_runs/u-ca206/All_months/' # sulphurless + pure biogenic
mpath2 ='/group_workspaces/jasmin2/asci/eeara/model_runs/u-ca179/All_months/' # baseline sulphueless sim

#mpath1 ='/group_workspaces/jasmin2/asci/eeara/model_runs/u-ca250/All_months/' # PD + purebiogenic
#mpath2 ='/group_workspaces/jasmin2/asci/eeara/model_runs/u-ca123/All_months/' # baseline Present day

#filename = 'All_months_m01s38i437_CN__NUMBER_CONCENTRATION____________.nc'
filename = 'All_months_m01s38i506_number_concentration_accumulation_mode.nc'

level_no = 50 #level 50 is 17 km altitide, and level 66 is 31km 
# level 30 = 6.8km, level 40 is 11.75km, level35 = 9.12km
def plot_cube(cube,title):
    plt.figure()
    qplt.contourf(cube,10,cmap = ruths_colors_r('roma'))
    plt.gca().coastlines()
    plt.title(title)

def diff_cube(cube1,cube2):
    cube1 = iris.load(mpath1+filename)[0]
    cube2 = iris.load(mpath2+filename)[0]

    cube1 = cube1[:,0:level_no]
    cube2 = cube2[:,0:level_no]

    cube1 = cube1.collapsed('time',iris.analysis.MEAN)
    cube1 = cube1.collapsed('model_level_number',iris.analysis.MEAN)
    cube2 = cube2.collapsed('time',iris.analysis.MEAN)
    cube2 = cube2.collapsed('model_level_number',iris.analysis.MEAN)

    diff = cube1-cube2
    return diff

def diff_cube_level(cube1,cube2,level):
    cube1 = iris.load(mpath1+filename)[0]
    cube2 = iris.load(mpath2+filename)[0]


    cube1 = cube1.collapsed('time',iris.analysis.MEAN)
    cube1 = cube1[level,:,:]
    #cube1 = cube1.collapsed('model_level_number',iris.analysis.MEAN)
    cube2 = cube2.collapsed('time',iris.analysis.MEAN)
    cube2 = cube2[level,:,:]
    #cube2 = cube2.collapsed('model_level_number',iris.analysis.MEAN)

    diff = cube1-cube2
    return diff

def diff_cube_units(cube1,cube2):
    cube1 = iris.load(mpath1+filename)[0]
    cube2 = iris.load(mpath2+filename)[0]

    cube1 = cube1[:,0:level_no]
    cube2 = cube2[:,0:level_no]

    cube1 = cube1.collapsed('time',iris.analysis.MEAN)
    cube1 = cube1.collapsed('model_level_number',iris.analysis.MEAN)
    cube2 = cube2.collapsed('time',iris.analysis.MEAN)
    cube2 = cube2.collapsed('model_level_number',iris.analysis.MEAN)

    diff = (cube1-cube2)/1.0e6
    return diff

def diff_cube_level_units(cube1,cube2,level):
    cube1 = iris.load(mpath1+filename)[0]
    cube2 = iris.load(mpath2+filename)[0]


    cube1 = cube1.collapsed('time',iris.analysis.MEAN)
    cube1 = cube1[level,:,:]
    #cube1 = cube1.collapsed('model_level_number',iris.analysis.MEAN)
    cube2 = cube2.collapsed('time',iris.analysis.MEAN)
    cube2 = cube2[level,:,:]
    #cube2 = cube2.collapsed('model_level_number',iris.analysis.MEAN)

    diff = (cube1-cube2)/1.0e6
    return diff


diff1 = diff_cube_units(mpath1,mpath2)

plot_cube(diff1,'Change in Accumulation mode (mean over time and model_level)')



diff2 = diff_cube_level_units(mpath1,mpath2,12)
plot_cube(diff2, 'change in Accumulation mode at 1km')

diff3 = diff_cube_level_units(mpath1,mpath2,40)
plot_cube(diff3, 'change in Accumulation mode at 12km')

#qplt.contourf(diff,10,cmap = ruths_colors_r('roma'))
#plt.gca().coastlines()
#plt.title('Change in N_total')
plt.show()


