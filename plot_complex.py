# Simulate Vietoris-Rips complex

import numpy as np
import matplotlib.pyplot as plt
import itertools

# Euclidean distance
def distance(points):
    ''' This function compute the euclidean distance
        between a set of pointsß '''
    return np.sqrt(np.sum((points[0] - points[1])**2))
def plot_balls_complex(x, y, alpha, complex_type = 'vietoris_rips', hide_balls = False):
    ''' This function return the plot of the points (x,y), the open ball with radius alpha 
        around every point and the selected complex '''

    fig, ax = plt.subplots(figsize = (10, 10))
    plt.box(False)
    
    if hide_balls:
        pass
    else:
        for i in range(n):
            circle = plt.Circle(
                                (x[i], y[i]), 
                                alpha, 
                                color = 'gray', 
                                alpha = 0.25, 
                                linestyle = '--', 
                                zorder = 0
                                )

            ax.add_patch(circle)
    
    ax.scatter(
                x, 
                y, 
                color = 'black', 
                zorder = 2
               )

    ax.set_xticks([])
    ax.set_yticks([])
    
    # Pairing points
    points = np.array(list(zip(x, y)))
    pairs = np.array(tuple(itertools.product(points, points)))
    
    # Select the complex type. If complex_type = 'vietoris_rips' then radius = alpha,
    # in the case complex_type = 'cech', then radius = 2*alpha

    if complex_type not in ['vietoris_rips', 'cech']:
        raise Exception("The selected complex type is not valid")
    else:
        radius_type = {
                        'vietoris_rips' : alpha,
                        'cech' : 2*alpha
                       }
    
    # Compute the distance between every pair of points and plot the line
    for pair in pairs:
        if ((distance(pair) <= radius_type[complex_type]) and not (pair[0] == pair[1]).all()):
            ax.plot(
                    pair[:, 0], 
                    pair[:, 1], 
                    color = 'red', 
                    zorder = 1
                    )

if __name__ == '__main__':
    # Generate random points
    alpha = .1
    n = 50
    x, y = np.random.rand(2, n)

    # Plot the result, complex_type = 'vietoris_rips'
    plot_balls_complex(x, y, alpha, hide_balls = False)

    # Plot the result, complex_type = 'cech'
    plot_balls_complex(x, y, alpha, complex_type = 'cech', hide_balls = False)