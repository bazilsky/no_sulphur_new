import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def new_colors(string):
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
        exit()

    color_path = '/home/users/eeara/ScientificColourMaps6/'+string+'/'+string+'.txt'
    cm_data = np.loadtxt(color_path)
    color_map = LinearSegmentedColormap.from_list(str, cm_data)
    return color_map
    

