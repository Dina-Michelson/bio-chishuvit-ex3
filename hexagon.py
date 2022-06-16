import numpy as np
from sklearn.preprocessing import MinMaxScaler


# a helper class for storing information about hexagons the som map
class hexagon():
    def __init__(self, coord, label):
        self.label = label # the label of the hexagon in the grid(numeric)
        self.coord = coord # the coordinates of the hexgon in the grid
        self.weight_vector = self.scaler(np.array(np.random.rand(14))) #a random normalized vector(there are 14 atributes in cvs file)
        self.city_list = [] #the list of cities mapped to the hexagon
     
    #functio that normalizes the values
    def scaler(self, vector):
        val_vector = []
        val_vector.append(vector)
        sc = MinMaxScaler(feature_range=(0,1))
        vector = sc.fit_transform(val_vector)
        return vector
        