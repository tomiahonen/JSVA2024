## TOMI AHONEN
## JSVA2024
## PROJECT 4
## Y68756994
## https://github.com/tomiahonen/JSVA2024


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

G = nx.read_gml('C:\\Users\\tomia\\OneDrive\\Työpöytä\\JSVA2024\\JSVA2024\\project\\adjnoun.gml')

plt.figure(figsize=(10, 10))
nx.draw(G, with_labels=True, node_size=150, node_color='skyblue', font_size=7)
plt.savefig("G.png")
#plt.show() 

#diameter of graph
diameter = nx.diameter(G)
print("Diameter of the graph: ", diameter)

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

highest_degree_centrality = heapq.nlargest(3, degree_centrality, key=degree_centrality.get)
print("The nodes with the three highest degree centrality are: ", highest_degree_centrality)
# meaning that these three nodes have the most amount of edges out of all nodes

closeness_centrality = nx.closeness_centrality(G)

highest_closeness_centrality = heapq.nlargest(3, closeness_centrality, key=closeness_centrality.get)
print("The nodes with the three highest closeness centrality are: ", highest_closeness_centrality)
# meaning that these three nodes are close to all other nodes in the graph, which means the node can quickly interact with other nodes

betweenness_centrality = nx.betweenness_centrality(G)

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

avg_clustering_coefficient = nx.average_clustering(G)
print("Average clustering coefficient: ", avg_clustering_coefficient)
print("##################################################################")
"""
5. We want to test the extent to which the centrality distributions in 4) fit a power law
distribution. You may inspire from the implementation in powerlaw · PyPI of the power-law
distribution, or can use alternative one of your choice. It is important to quantify the
goodness of fit using p-value. Typically, when p-value is greater than 10%, we can state that
power-law is a plausible fit to the (distribution) data.
"""
# Fit a power law distribution to the centrality distributions and calculate the p-value
fit_degree = powerlaw.Fit(np.array(list(degree_centrality.values())) + 1, discrete=True)
R, p = fit_degree.distribution_compare('power_law', 'exponential', normalized_ratio=True)
print("Degree centrality distribution power law fit p-value:", p)

fit_closeness = powerlaw.Fit(np.array(list(closeness_centrality.values())) + 1, discrete=True)
R, p = fit_closeness.distribution_compare('power_law', 'exponential', normalized_ratio=True)
print("Closeness centrality distribution power law fit p-value:", p)

fit_betweenness = powerlaw.Fit(np.array(list(betweenness_centrality.values())) + 1, discrete=True)
R, p = fit_betweenness.distribution_compare('power_law', 'exponential', normalized_ratio=True)
print("Betweenness centrality distribution power law fit p-value:", p)
# the p-value is now less than 10% which means that the power law distribution is not a plausible fit to the data

# lets try log-normal distribution

fit_degree = powerlaw.Fit(np.array(list(degree_centrality.values())) + 1, discrete=True)
R, p = fit_degree.distribution_compare('lognormal', 'exponential', normalized_ratio=True)
print("Degree centrality distribution log-normal fit p-value:", p)

fit_closeness = powerlaw.Fit(np.array(list(closeness_centrality.values())) + 1, discrete=True)
R, p = fit_closeness.distribution_compare('lognormal', 'exponential', normalized_ratio=True)
print("Closeness centrality distribution log-normal fit p-value:", p)

fit_betweenness = powerlaw.Fit(np.array(list(betweenness_centrality.values())) + 1, discrete=True)
R, p = fit_betweenness.distribution_compare('lognormal', 'exponential', normalized_ratio=True)
print("Betweenness centrality distribution log-normal fit p-value:", p)

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
from networkx.algorithms.community import modularity
from networkx.algorithms.community import girvan_newman
print("girvan-newman")

communities = girvan_newman(G)
communities = next(communities)

colors = []
for node in G.nodes():
    for i, community in enumerate(communities):
        if node in community:
            colors.append(i)
            break

nx.draw(G, node_color=colors, with_labels=True)
plt.show()

mod = modularity(G, communities)
print(f"Modularity (Girvan-Newman): {mod}")
print(len(communities))

import matplotlib.cm as cm
import community as community_louvain

print("Louvain")
partition = community_louvain.best_partition(G)

plt.figure(figsize=(10, 10))  
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)

nx.draw(G, with_labels=True, node_size=150,
        node_color=[partition.get(node) for node in G.nodes()],
        font_size=7, cmap=cmap, edge_color='k')

plt.show()

modularity = community_louvain.modularity(partition, G)
print(f"Modularity (Louvain): {modularity}")
"""
8. We want to analyze the network in terms of type of cascades available in the network.
Write a script that identifies the number of starts and number of chains (see definition in
Handout 5) attached to liberal and conservative nodes. 
"""

# ?

"""
9. We want to quantify the density of adjective and noun node topology. For this purpose,
write a script that calculates, for each node X, the proportion P(X) of neighbors that have
the same affiliation as X. Compute the average score of P(X) for all noun nodes and average
score for all adjective nodes. Comment on the relationship among noun and adjective nodes. 
"""

def calculate_proportion(G, node):
    neighbors = list(G.neighbors(node))
    if not neighbors:
        return 0
    same_type_count = sum(1 for neighbor in neighbors if G.nodes[neighbor]['value'] == G.nodes[node]['value'])
    return same_type_count / len(neighbors)

proportions = {node: calculate_proportion(G, node) for node in G.nodes}

noun_proportions = [p for node, p in proportions.items() if G.nodes[node]['value'] == 1]
adjective_proportions = [p for node, p in proportions.items() if G.nodes[node]['value'] == 0]

average_noun_proportion = sum(noun_proportions) / len(noun_proportions) if noun_proportions else 0
average_adjective_proportion = sum(adjective_proportions) / len(adjective_proportions) if adjective_proportions else 0

print(f"Average Proportion for Noun Nodes: {average_noun_proportion}")
print(f"Average Proportion for Adjective Nodes: {average_adjective_proportion}")


