import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from matplotlib.patches import RegularPolygon
import random
from neighbors import *
from hexagon import * 

file_path = sys.argv[1]

LearnMax = 0.5
epochs = 8

#the coordinates of the hexagon grid som map
coord = [[0,4,-4],[-1,4,-3],[0,3,-3],[1,3,-4],[2,2,-4],[1,2,-3],[0,2,-2],[-1,3,-2],
 	[-2,4,-2],[-3,4,-1],[-2,3,-1],[-1,2,-1],[0,1,-1],[1,1,-2],[2,1,-3],
	[3,1,-4],[4,0,-4],[3,0,-3],[2,0,-2],[1,0,-1],[0,0,0],[-1,1,0],[-2,2,0],
	[-3,3,0],[-4,4,0],[-4,3,1],[-3,2,1],[-2,1,1],[-1,0,1],[0,-1,1],
	[1,-1,0],[2,-1,-1],[3,-1,-2],[4,-1,-3],[4,-2,-2],[3,-2,-1],[2,-2,0],
	[1,-2,1],[0,-2,2],[-1,-1,2],[-2,0,2],[-3,1,2],[-4,2,2],
	[-4,1,3],[-3,0,3],[-2,-1,3],[-1,-2,3],[0,-3,3],[1,-3,2],[2,-3,1],[3,-3,0],
	[4,-3,-1],[4,-4,0],[3,-4,1],[2,-4,2],[1,-4,3],[0,-4,4],[-1,-3,4],[-2,-2,4],
	[-3,-1,4],[-4,0,4]]



# the label of the hexagons in the som map
labels = [['0'],['1'],['2'],['3'],['4'],['5'],['6'],['7'],['8'],['9'],
          ['10'],['11'],['12'],['13'],['14'],['15'],['16'],['17'],
          ['18'],['19'],['20'],['21'],['22'],['23'],['24']
         ,['25'],['26'],['27'],['28'],['29'],['30']
          ,['31'],['32'],['33'],['34'],['35']
          ,['36'],['37'],['38'],['39'],['40'],['41'],['42'],['43']
          ,['44'],['45'],['46'],['47'],['48'],['49'],['50']
          ,['51'],['52'],['53'],['54'],['55'],['56']
          ,['57'],['58'],['59'],['60']]



#loading dataset
dataset = pd.read_csv(file_path)

#dividing dataset into their groups to process seperately(deoendant variables, independent variables and labels)
x = dataset.iloc [:,2:].values
y = dataset.iloc [:,0].values
z = dataset.iloc [:,1].values
total_votes = dataset.iloc[:,2].values


#feature scaling
sc = MinMaxScaler(feature_range=(0,1))
x = sc.fit_transform(x)

#makes a dictionary that maps cities to their socio-economic status for better access later
def make_city_socio_econimic_dict():
    city_socio_economic_dict = {}
    for Y,Z in zip(y,z):
        city_socio_economic_dict[Y] = Z
    return city_socio_economic_dict

#initializes the som map with a hexagon grid where each hexagn has a random weight vector
def initialize_som_map():
    hexagon_list = []
    for i in range(61):
        hexagon_list.append(hexagon(coord[i], labels[i]))
    return hexagon_list

'''trains the som, for a finite number of epoch, we check for random cities which hexgon they are mapped to
    after find the vector that has the closest eucledian distance we update the weights of that vector and of its neighbors
    we update the closest one the most and then 1st degree neighbors a little bit less, and second degree neighors even less'''
def train_som_map(y,z,x):
    som_map = initialize_som_map()
    for i in range(epochs):             
        c = list(zip(y, z, x))
        np.random.shuffle(c)
        y,z,x = zip(*c)
        for Y,Z,X in zip(y,z,x):
            minimum = 999  
            counter = 0
            temp = -1
            for hexa in som_map:
                dist = np.linalg.norm(np.array(X)-hexa.weight_vector)
                if dist < minimum:
                    minimum = dist
                    temp = counter
                counter = counter + 1
            #print(temp)
            som_map[temp] = update_hexa(som_map[temp], X, i)
            som_map = update_hexa_fst_deg(som_map, X, som_map[temp].label, i)
            som_map = update_hexa_scnd_deg(som_map, X, som_map[temp].label, i)
    return som_map 

#this function "tests" our model, after it was trained and maps cities to hexagons according to trained som
def test_som_map(som_map,y,z,x):
    dist_sum = 0
    city_counter = 0
    hex_dict = {}
    for Y,Z,X in zip(y,z,x):
        minimum = 999
        counter = 0
        temp = -1
        for hexa in som_map:
            dist = np.linalg.norm(np.array(X)-hexa.weight_vector)
            if dist < minimum:
                minimum = dist
                temp = counter
            counter +=1
        city_counter +=1
        dist_sum += minimum
        som_map[temp].city_list.append(Y)
    print('average: ' + str(dist_sum / city_counter))
    return som_map, str(dist_sum / city_counter)

#update the weight vector of the hexagon that the city was mapped to        
def update_hexa(hexa, X, epoch):
    pct_left = 1.0 - (epoch / epochs)
    curr_rate = pct_left 
    hexa.weight_vector = hexa.weight_vector + curr_rate * (X-hexa.weight_vector)
    return hexa


#update the hexagons that are first degree neighbors(right next)to the hexagon the city was mapped to
def update_hexa_fst_deg(som_map,X, label, epoch):
    pct_left = 1.0 - (epoch / epochs)
    curr_rate = 0.3*(pct_left * epochs)
    for neighbor in NEIGHBOR_DICT_FST[label[0]]:
        som_map[neighbor].weight_vector = som_map[neighbor].weight_vector + curr_rate * (X-som_map[neighbor].weight_vector)
    return som_map

#update the hexagons that are secong degree neighbors to the hexagon the city was mapped to
def update_hexa_scnd_deg(som_map,X, label, epoch):
    pct_left = 1.0 - (epoch / epochs)
    curr_rate = 0.1*(pct_left )
    for neighbor in NEIGHBOR_DICT_SCND[label[0]]:
        som_map[neighbor].weight_vector = som_map[neighbor].weight_vector + curr_rate * (X-som_map[neighbor].weight_vector)
    return som_map

#update the hexagons that are secong degree neighbors to the hexagon the city was mapped to (the non overlapping neighbors neighbors)
def update_hexa_fst_deg2(som_map,X, label, epoch):
    pct_left = 1.0 - (epoch / epochs)
    curr_rate = 0.4*(pct_left )
    label  = int(label[0])
    if label > 0:
        som_map[label-1].weight_vector = som_map[label-1].weight_vector + curr_rate * (X-som_map[label-1].weight_vector)
    if label < 60:
        som_map[label+1].weight_vector = som_map[label+1].weight_vector + curr_rate * (X-som_map[label+1].weight_vector)
    return som_map

#update the hexagons that are secong degree neighbors to the hexagon the city was mapped to
def update_hexa_scnd_deg2(som_map,X, label, epoch):
    pct_left = 1.0 - (epoch / epochs)
    curr_rate = 0.2*(pct_left * epochs)
    label  = int(label[0])
    if label > 1:
        som_map[label-2].weight_vector = som_map[label-2].weight_vector + curr_rate * (X-som_map[label-2].weight_vector)
    if label < 59:
        som_map[label+2].weight_vector = som_map[label+2].weight_vector + curr_rate * (X-som_map[label+2].weight_vector)
    return som_map

#given a list of cities mapped to hexagn on the grid, calculates the average of the socio economic statuses
def find_avg_socio(hexa, economic_dict):
    economic_list = []
    for city in hexa.city_list:
        economic_list.append(economic_dict[city])
    if len(hexa.city_list) == 0:
        return 0
    return int(sum(economic_list)) / int(len(hexa.city_list))

#a colour list similar colours are near each other to show convergance on the som map 
colors = ['w','xkcd:pale blue','xkcd:robin\'s egg blue', 'xkcd:baby blue', 'xkcd:cyan','xkcd:aqua','xkcd:sea green','xkcd:spring green','xkcd:apple green','xkcd:leaf green','xkcd:shamrock']



#function the prints the cities mapped to each hexagon with their respective socio economic status
def print_cities_per_hex(som_map):
    economic_dict = make_city_socio_econimic_dict()
    for h in som_map:
        print(h.label )
        for c in h.city_list:
            print('city: '+c + ', socio-economic status: '+ str(economic_dict[c]))
            
'''function that draws the som, a hexa grid that is coloured 
   by the average of the socio-economic stautus of the cities mapped to the the hexagon'''         
def generate_som_map(som_map, average):
    # Horizontal cartesian coords
    economic_dict = make_city_socio_econimic_dict()
    hcoord = [c[0] for c in coord]

    # Vertical cartersian coords
    vcoord = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) /3. for c in coord]
    plt.ion()
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')

    # Add some coloured hexagons by the average the sum of the socio economic statuses of the cities mapped to it them
    for x, y, l, h in zip(hcoord, vcoord, labels,som_map): 
        hex = RegularPolygon((x, y), numVertices=6, radius=2. / 3., 
                             orientation=np.radians(30), 
                             facecolor=colors[int(find_avg_socio(h, economic_dict))], alpha=1, edgecolor='k')
        ax.add_patch(hex)
        # Also add a text label
        ax.text(x, y+0.2, l[0], ha='center', va='center', size=9)

    # Also add scatter points in hexagon centres
    ax.scatter(hcoord, vcoord, c='b', alpha=0.01)
    plt.title('average:' + average)
    plt.show()
    plt.savefig('som'+average+'.png')
    plt.close('all')

#runs the training them the test and then generates the SOM
def run_model():
    som_map = train_som_map(y,z,x)
    som_map, average = test_som_map(som_map,y,z,x)
    print_cities_per_hex(som_map)
    generate_som_map(som_map, average)
    


run_model()    
    
    
    
    
    
    
   #here bla 