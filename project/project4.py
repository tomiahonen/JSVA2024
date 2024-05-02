import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import cumfreq
import heapq
import powerlaw

"""
1. Use NetworkX to display the corresponding network, suggest appropriate simple labelling
of the nodes to maintain the readability of the network graph as clear as possible. Save the
adjacency matrix of this graph in a separate file
"""

G = nx.read_gml('C:\\Users\\OMISTAJA\\Desktop\\JSVA2024\\JSVA2024\\project\\adjnoun.gml')

# Draw the graph
plt.figure(figsize=(10, 10))
nx.draw(G, with_labels=True, node_size=300, node_color='skyblue')
plt.savefig("network_graph.png")
#plt.show() 

#Adjacency matrix
adjacency_matrix = nx.to_numpy_array(G)
np.savetxt("adjacency_matrix.txt", adjacency_matrix, fmt="%d")

"""
2. Write a script to show whether the graph is bipartie graph or not. 
"""

if nx.is_bipartite(G):      #returns True if G is bipartite, False if not
    print("The graph is bipartite")
else:
    print("The graph is not bipartite")

"""
3. Suggest a script that uses NetworkX functions to identify the nodes of the three highest
degree centrality, three highest closeness centrality and three highest betweenness
centrality. 
"""

degree_centrality = nx.degree_centrality(G)

# get the node with three highest degree centrality
highest_degree_centrality = heapq.nlargest(3, degree_centrality, key=degree_centrality.get)
print("The nodes with the three highest degree centrality are: ", highest_degree_centrality)
# meaning that these three nodes have the most amount of edges out of all nodes

closeness_centrality = nx.closeness_centrality(G)

# get the node with three highest closeness centrality
highest_closeness_centrality = heapq.nlargest(3, closeness_centrality, key=closeness_centrality.get)
print("The nodes with the three highest closeness centrality are: ", highest_closeness_centrality)
# meaning that these three nodes are close to all other nodes in the graph, which means the node can quickly interact with other nodes

betweenness_centrality = nx.betweenness_centrality(G)

# get the node with three highest betweenness centrality
highest_betweenness_centrality = heapq.nlargest(3, betweenness_centrality, key=betweenness_centrality.get)
print("The nodes with the three highest betweenness centrality are: ", highest_betweenness_centrality)
# meaning that these three nodes have the most amount of shortest paths that pass through them

"""
4. Write a script that plots the degree centrality distribution, closeness centrality distribution
and betweenness centrality distribution. Also, write a script to plot the cumulative degree
distribution and the clustering coefficient distribution
"""

# Degree centrality distribution
plt.figure()
plt.hist(list(degree_centrality.values()))
plt.title("Degree centrality distribution")
plt.xlabel("Degree centrality")
plt.ylabel("Frequency")
plt.show()

# Closeness centrality distribution
plt.figure()
plt.hist(list(closeness_centrality.values()))
plt.title('Closeness Centrality Distribution')
plt.xlabel('Closeness Centrality')
plt.ylabel('Frequency')
plt.show()

# Betweenness centrality distribution
plt.figure()
plt.hist(list(betweenness_centrality.values()))
plt.title('Betweenness Centrality Distribution')
plt.xlabel('Betweenness Centrality')
plt.ylabel('Frequency')
plt.show()

# Cumulative degree distribution
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
degree_counts = cumfreq(degree_sequence, numbins=len(set(degree_sequence)))
plt.figure()
plt.plot(degree_counts.lowerlimit + np.linspace(0, degree_counts.binsize*degree_counts.cumcount.size, degree_counts.cumcount.size), degree_counts.cumcount / degree_counts.cumcount[-1])
plt.title('Cumulative Degree Distribution')
plt.xlabel('Degree')
plt.ylabel('Cumulative Frequency')
plt.show()
# meaning that the cumulative degree distribution shows the proportion of nodes that have a degree less than or equal to a certain value

# Clustering coefficient distribution
clustering_coefficients = nx.clustering(G)
plt.figure()
plt.hist(list(clustering_coefficients.values()))
plt.title('Clustering Coefficient Distribution')
plt.xlabel('Clustering Coefficient')
plt.ylabel('Frequency')
plt.show()
# meaning that the clustering coefficient distribution shows the frequency of nodes that have a certain clustering coefficient
# Clusterin coef. of 0 means that none of a node's neighbours are connected to each other, while 1 means that all of a node's neighbours are connected to each other
#The average clustering coefficient of this network is avg_clustering_coefficient = nx.average_clustering(G)

print("##################################################################")
"""
5. We want to test the extent to which the centrality distributions in 4) fit a power law
distribution. You may inspire from the implementation in powerlaw · PyPI of the power-law
distribution, or can use alternative one of your choice. It is important to quantify the
goodness of fit using p-value. Typically, when p-value is greater than 10%, we can state that
power-law is a plausible fit to the (distribution) data.
"""

# Fit a log-normal distribution to the degree centrality distribution
fit_degree = powerlaw.Fit(np.array(list(degree_centrality.values())) + 1, discrete=True)
R, p = fit_degree.distribution_compare('lognormal', 'exponential', normalized_ratio=True)
print("Degree centrality distribution log-normal fit p-value:", p)

# Fit a log-normal distribution to the closeness centrality distribution
fit_closeness = powerlaw.Fit(np.array(list(closeness_centrality.values())) + 1, discrete=True)
R, p = fit_closeness.distribution_compare('lognormal', 'exponential', normalized_ratio=True)
print("Closeness centrality distribution log-normal fit p-value:", p)

# Fit a log-normal distribution to the betweenness centrality distribution
fit_betweenness = powerlaw.Fit(np.array(list(betweenness_centrality.values())) + 1, discrete=True)
R, p = fit_betweenness.distribution_compare('lognormal', 'exponential', normalized_ratio=True)
print("Betweenness centrality distribution log-normal fit p-value:", p)

#log-normal distributuin fits better here than powerlaw distribution
print("##################################################################")
"""
6. We want to use exponentially truncated power-law instead of power law distribution.
Suggest a script that quantifies the goodness of fit for degree-centrality, closeness centrality
and betweenness centrality distributions.
"""

from scipy.stats import exponweib, kstest

# Fit an exponentially truncated power-law distribution to the degree centrality distribution
params = exponweib.fit(list(degree_centrality.values()), floc=0, f0=1)
D, p = kstest(list(degree_centrality.values()), 'exponweib', args=params)
print("Degree centrality distribution exponentially truncated power-law fit p-value:", p)

# Fit an exponentially truncated power-law distribution to the closeness centrality distribution
params = exponweib.fit(list(closeness_centrality.values()), floc=0, f0=1)
D, p = kstest(list(closeness_centrality.values()), 'exponweib', args=params)
print("Closeness centrality distribution exponentially truncated power-law fit p-value:", p)

# Fit an exponentially truncated power-law distribution to the betweenness centrality distribution
params = exponweib.fit(list(betweenness_centrality.values()), floc=0, f0=1)
D, p = kstest(list(betweenness_centrality.values()), 'exponweib', args=params)
print("Betweenness centrality distribution exponentially truncated power-law fit p-value:", p)

print("##################################################################")

"""
7. We want to identify relevant communities from the network graph. For this purpose, use
Label propagation algorithm implementation in NetworkX to identify the main
communities. Write a script that uses different color for each community and visualize the
above graph with the detected communities. Use the appropriate function in NetworkX to
compute the separation among the various communities and any other related quality
measures. Comment on the quality of the partition by taking into account the available
knowledge about the node attributes (0 and 1 values depending whether it accommodates
noun or adjectives). 
"""
from networkx.algorithms import community

communities = list(community.label_propagation_communities(G))
modularity = community.modularity(G, communities)

print("Modularity of the partition: ", modularity)

node_map = {node: i for i, node in enumerate(G.nodes)}

colors = [0] * nx.number_of_nodes(G)
for i, com in enumerate(communities):
    for node in com:
        colors[node_map[node]] = i

nx.draw(G, node_color=colors, with_labels=True)
plt.show()