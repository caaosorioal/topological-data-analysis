import numpy as np
import matplotlib.pyplot as plt
import imageio
import gif
from os.path import join
import os

# Get the linear function passing through point1 and point2 
def linear_function(x, point1, point2):
    ''' This function return the line 
        between point1 and point2'''

    m = (point2 - point1)[1] / (point2 - point1)[0]
    return m * (x - point1[0]) + point1[1]

# Get the quadratic function passing through point1 and point2 
def quadratic_function(x, point1, point2, seed = None):
    ''' This function return the parabola
        passing through point1 and point2. 
        There are infinitely many possible 
        parabolas, and that's why there
        is a seed parameter.'''
    
    np.random.seed(seed)
    point0 = np.random.rand(1, 2)[0]
    
    L0 = ((x - point1[0])*(x - point2[0])) / ((point0[0] - point1[0])*(point0[0] - point2[0]))
    L1 = ((x - point0[0])*(x - point2[0])) / ((point1[0] - point0[0])*(point1[0] - point2[0]))
    L2 = ((x - point0[0])*(x - point1[0])) / ((point2[0] - point0[0])*(point2[0] - point1[0]))
    
    return point0[1] * L0 + point1[1] * L1 + point2[1] * L2

# Get the homotopy function as a convex combination of linear_function and quadratic_funcion
def H(t, x, point1, point2):
    return (1-t) * y_linear + t * y_quadratic

# Generate two random points with no seed
np.random.seed()
point1, point2 = np.random.rand(2, 2)

# Get the horizontal space
x = np.linspace(point1[0], point2[0], 100)

# Get the functions
y_linear = linear_function(x, point1, point2)
y_quadratic = quadratic_function(x, point1, point2, seed = 10)

# Path
path = '/Users/carlosandresosorioalcalde/Documents/GitHub/topological-data-analysis/Images'

# Plot every step in the animation
for i, t in enumerate(np.linspace(0, 1, 100)):
    y_H = H(t, x, point1, point2)
    fig, ax = plt.subplots(figsize = (10, 10))
    plt.box(False)
    
    ax.scatter(
                [point1[0], point2[0]], 
                [point1[1], point2[1]],
                zorder = 2,
                s = 150,
                color = 'red'
               )
    
    ax.plot(x, y_linear, color = 'black', zorder = 1)
    ax.plot(x, y_quadratic, linestyle = '--', color = 'gray', zorder = 1)
    ax.plot(x, y_H, color = 'black', zorder = 0)
    ax.set_xticks([])
    ax.set_yticks([])
    id_image = str(i).zfill(3)
    
    plt.savefig(join(path, f'{id_image}.png'))
    
# Export gif
images_path = sorted([f for f in os.listdir(path) if f.split('.')[1] == 'png'])
ims = [imageio.imread(join(path, f), pilmode = "RGB") for f in images_path]
imageio.mimwrite(join(path, 'homotopy_image.gif'), ims, duration = 0.02)