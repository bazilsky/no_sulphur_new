import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

###Function for Colorbars that ruth keeps going on about
def ruths_colors(string):
    sequential_list = ['devon','lajolla','bamako','davos','bilbao','nuuk','oslo','grayC','hawaii','lapaz','tokyo','buda','acton','turku','imola']
    diverging_list = ['broc','cork','vik','lisbon','tofino','berlin']
    special_list = ['batlow','roma','oleron']
    cyclic_list = ['romaO','brocO','corkO','vikO']
    
    total_list = sequential_list+diverging_list+special_list+cyclic_list
    if string not in total_list:
        print('color not found, please choose from the following colors in the image')
        print('sequential_colorbars = ',sequential_list)
        print('diverging_colorbars = ',diverging_list)
        print('special_colorbars = ',special_list)
        print('cyclic_colorbars = ',cyclic_list)
        # output image with color choices to choose from 
        img = Image.open('/group_workspaces/jasmin2/asci/eeara/new_colorbar/ruths_colors.png')
        img.show()
        exit()

    color_path = '/group_workspaces/jasmin2/asci/eeara/new_colorbar/ScientificColourMaps6/'+string+'/'+string+'.txt'
    #color_path = '/home/users/eeara/ScientificColourMaps6/'+string+'/'+string+'.txt'
    cm_data = np.loadtxt(color_path)
    color_map = LinearSegmentedColormap.from_list(str, cm_data)
    return color_map






###Function for Colorbars that ruth keeps going on about -- but in reverse order
def ruths_colors_r(string):
    sequential_list = ['devon','lajolla','bamako','davos','bilbao','nuuk','oslo','grayC','hawaii','lapaz','tokyo','buda','acton','turku','imola']
    diverging_list = ['broc','cork','vik','lisbon','tofino','berlin']
    special_list = ['batlow','roma','oleron']
    cyclic_list = ['romaO','brocO','corkO','vikO']
    
    total_list = sequential_list+diverging_list+special_list+cyclic_list
    if string not in total_list:
        print('color not found, please choose from the following colors')
        print('sequential_colorbars = ',sequential_list)
        print('diverging_colorbars = ',diverging_list)
        print('special_colorbars = ',special_list)
        print('cyclic_colorbars = ',cyclic_list)
        # output image with color choices to choose from 
        img = Image.open('/group_workspaces/jasmin2/asci/eeara/new_colorbar/ruths_colors.png')
        img.show()
        exit()

    color_path = '/group_workspaces/jasmin2/asci/eeara/new_colorbar/ScientificColourMaps6/'+string+'/'+string+'.txt'
    cm_data_2 = np.loadtxt(color_path)
    cm_data = np.flip(cm_data_2,axis=0)
    color_map = LinearSegmentedColormap.from_list(str, cm_data)
    return color_map

