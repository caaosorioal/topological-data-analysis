# Simulate Vietoris-Rips complex

import numpy as np
import matplotlib.pyplot as plt
import itertools

# Euclidean distance
def distance(points):
    ''' This function compute the euclidean distance
        between a pair of points. '''

    return np.sqrt((points[0][0] - points[1][0])**2 + (points[0][1] - points[1][1])**2)

def plot_balls_complex(x, y, alpha, hide_balls = False):
    ''' This function return the plot of the points (x,y), the open ball with radius alpha 
        around every point and the Vietoris-Rips complex '''

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
    
    # Compute the distance between every pair of points
    edges_complex = []
    for pair in pairs:
        if ((distance(pair) <= alpha) and not (pair[0] == pair[1]).all()):
            edges_complex.append(pair)
    
    for edge in edges_complex:
        ax.plot(
                edge[:, 0], 
                edge[:, 1], 
                color = 'red', 
                zorder = 1
                )

if __name__ == "__main__":
    # Generate random points
    alpha = .1
    n = 50
    x, y = np.random.rand(2, n)

    # Plot the result
    plot_balls_complex(x, y, alpha, hide_balls = False)