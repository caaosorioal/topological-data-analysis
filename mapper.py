import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
import plotly
import networkx as nx

# Function to read the data
def read_data(path, file, chunk_size):
    df_list = []
    with pd.read_csv(join(path, file), chunksize = chunk_size, low_memory = False) as r:
        for chunk in r:
            df_list.append(chunk)
    
    return pd.concat(df_list)

# Function to find the preimage of interval
def condition_preimage(x, tuple):
    if ((x >= tuple[0]) and (x <= tuple[1])):
        return True
    return False

# Find the intersection between two sets of 3-dimensional points
def find_intersection(array1, array2):
    columns = ['d1', 'd2', 'd3']
    df_temp_1 = pd.DataFrame(data = array1, columns = columns)
    df_temp_2 = pd.DataFrame(data = array2, columns = columns)
    
    df_intersection = pd.merge(df_temp_1,
                                df_temp_2,
                                on = columns,
                                how = 'inner')
    
    return df_intersection.shape[0]

# Clustering the cover using DBSCAN
def clustering_cover(U: list, eps = 0.1, min_samples = 25) -> list:
    clusters = []
    for interval in U:
        preimages = [value for value in x if condition_preimage(f(value), interval)]
        clustering = DBSCAN(eps = eps, min_samples = min_samples).fit(preimages)
        etiquetas = clustering.labels_
        
        for i in range(len(preimages)):
            preimages[i] = np.concatenate((preimages[i], [etiquetas[i]]), axis = 0)

        data_clusters = pd.DataFrame(data = preimages,
                                    columns = ['d1', 'd2', 'd3', 'c'])

        for cluster in data_clusters['c'].unique():
            clusters.append(data_clusters[data_clusters['c'] == cluster][['d1', 'd2', 'd3']].values)
        
    return clusters

# f
def f(x):
    return np.dot(coef, x)

# Create the nerve based on the clusters
def create_nerve(clusters):
    nodes = list(range(len(clusters)))
    edges = []
    G = nx.Graph()
    G.add_nodes_from(nodes)
    
    for i, cluster1 in enumerate(clusters):
        for j, cluster2 in enumerate(clusters):
            if (find_intersection(cluster1, cluster2) > 0) and (i != j):
                G.add_edge(i, j)
    return G

# Get the mean value of f for every cluster
def get_mean_value_f(clusters):
    return [np.mean(list(map(f, cluster))) for cluster in clusters]

if __name__ == '__main__':
    # Select the files and the path
    path = '/Users/carlosandresosorioalcalde/'
    file = 'Saber_11__2019-2.csv'

    # Read the data with chunks
    chunk_size = 10**5
    df_icfes = read_data(path, file, chunk_size)

    # Select the columns to deal with (just "PUNT_" columns)
    columns = [column for column in df_icfes.columns if 'PUNT_' in column]

    # Cutted data with columns
    n_sample = 10**5
    df_icfes_cutted = df_icfes[columns].dropna().sample(n_sample)
    values = df_icfes_cutted.values

    # Standarize the data
    scaler = StandardScaler()
    data = scaler.fit_transform(values)

    # Get f as a multivariate linear regression
    x, y = data[:, :3], data[:, 5]
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    coef = regr.coef_

    # Plot the data
    fig = go.Figure(data=[go.Scatter3d(
        x=x[:, 0],
        y=x[:, 1],
        z=x[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=f(x.T),
            colorscale='inferno',  
            opacity=0.6
        )
    )])

    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
        )
    )

    fig.write_html(join(path, 'file.html'))
    fig.show()

    # Create the covering for f(x)
    n_intervals = 10
    min_score = y.min()
    max_score = y.max()
    delta = max_score - min_score
    gain = 0.5

    # This is the covering for f(x)
    U = [(min_score + (gain * i) * (delta) / n_intervals, min_score + (1 + gain * i) * (delta)/n_intervals) for i in range(2 * n_intervals)]

    # Create the clusters for every set of the cover
    eps = .15
    min_samples = 15
    clusters = clustering_cover(U, eps, min_samples)

    # Create edges and nodes
    G = create_nerve(clusters)

    # Plot the resulting graph
    node_colors = get_mean_value_f(clusters)
    cmap = plt.cm.inferno

    plt.figure(figsize = (12, 10))
    nx.draw(
            G, 
            edge_color = 'black',
            node_color = node_colors,
            node_size = 60,
            cmap = cmap,
            linewidths = 2
        )

    sm = plt.cm.ScalarMappable(
                            cmap=cmap, 
                            norm=plt.Normalize(
                                                vmin=min(node_colors), 
                                                vmax=max(node_colors)
                                                )
                            )
    sm.set_array([])
    cbar = plt.colorbar(sm)

