"""
B)
Consider the eggo-Facebook dataset, available at https://snap.stanford.edu/data/ego-Facebook.html

Write a program that

a) Calculates the degree, closeness and in-betweeness centrality of each node of the network, and displays the corresponding distribution (histogram)

b) Calculates the shortest distance between node (s) of highest centrality score and node (s) of second highest centrality score (for both degree, closeness, in-betweeness centraility measures)

c) Displays the subgraph where the nodes are most connected (in terms of degree centrality).. Can use your own reasoning for this issue

d) Calculates the local clustering coefficients and the shortest distance among the nodes with highest and second highest clustering coefficient.

d) Checks whether Power-law distribution is fitted
"""
import networkx as nx
import matplotlib.pyplot as plt

# Load the data
G = nx.read_edgelist('JSVA2024\\lab2\\facebook_combined.txt')
#G=nx.read_edgelist(r'JSVA2024\lab2\facebook_combined.txt')
###########
#Note: takes about 45 seconds to run#######

# Calculate the centralities
degree_centrality = dict(G.degree())

# Use a subset of nodes for the closeness centrality calculation
subset_nodes = list(G.nodes())[::10]  # Adjust the step size based on the size of the graph
closeness_centrality = {node: nx.closeness_centrality(G, node) for node in subset_nodes}

# Use approximation for the betweenness centrality calculation
subset_nodes = list(G.nodes())[::10]  # Adjust the step size based on the size of the graph
betweenness_centrality = nx.betweenness_centrality_subset(G, subset_nodes, subset_nodes)

# Plot the degree centrality distribution
plt.figure(figsize=(10, 6))
plt.hist(list(degree_centrality.values()), bins=20)
plt.title('Degree Centrality Distribution')
plt.show()

# Plot the closeness centrality distribution
plt.figure(figsize=(10, 6))
plt.hist(list(closeness_centrality.values()), bins=20)
plt.title('Closeness Centrality Distribution')
plt.show()

# Plot the betweenness centrality distribution
plt.figure(figsize=(10, 6))
plt.hist(list(betweenness_centrality.values()), bins=20)
plt.title('Betweenness Centrality Distribution')
plt.show()

#b) Calculates the shortest distance between node (s) of highest centrality score and node (s) of second highest centrality score (for both degree, closeness, in-betweeness centraility measures)

# Function to get the nodes with the highest and second highest centrality scores
def get_top_two_nodes(centrality_dict):
    sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes[0][0], sorted_nodes[1][0]

# Get the nodes with the highest and second highest degree centrality scores
degree_node1, degree_node2 = get_top_two_nodes(degree_centrality)

# Get the nodes with the highest and second highest closeness centrality scores
closeness_node1, closeness_node2 = get_top_two_nodes(closeness_centrality)

# Get the nodes with the highest and second highest betweenness centrality scores
betweenness_node1, betweenness_node2 = get_top_two_nodes(betweenness_centrality)

# Calculate the shortest distances
degree_distance = nx.shortest_path_length(G, degree_node1, degree_node2)
closeness_distance = nx.shortest_path_length(G, closeness_node1, closeness_node2)
betweenness_distance = nx.shortest_path_length(G, betweenness_node1, betweenness_node2)

# Print the shortest distances
print(f"Shortest distance between nodes with highest and second highest degree centrality: {degree_distance}")
print(f"Shortest distance between nodes with highest and second highest closeness centrality: {closeness_distance}")
print(f"Shortest distance between nodes with highest and second highest betweenness centrality: {betweenness_distance}")

# c) Displays the subgraph where the nodes are most connected (in terms of degree centrality)

# Function to get the nodes with the highest degree centrality
def get_top_nodes(centrality_dict, num_nodes):
    sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)
    return [node[0] for node in sorted_nodes[:num_nodes]]

# Get the nodes with the highest degree centrality
top_degree_nodes = get_top_nodes(degree_centrality, 10)  # Adjust the number of nodes based on your needs

# Create a subgraph with these nodes
subgraph = G.subgraph(top_degree_nodes)

# Draw the subgraph
nx.draw(subgraph, with_labels=True)
plt.show()

#d) Calculates the local clustering coefficients and the shortest distance among the nodes with highest and second highest clustering coefficient.

# Calculate the local clustering coefficients
clustering_coefficients = nx.clustering(G)

# Get the nodes with the highest and second highest clustering coefficients
clustering_node1, clustering_node2 = get_top_two_nodes(clustering_coefficients)

# Calculate the shortest distance
clustering_distance = nx.shortest_path_length(G, clustering_node1, clustering_node2)

# Print the shortest distance
print(f"Shortest distance between nodes with highest and second highest clustering coefficient: {clustering_distance}")

#e) Checks whether Power-law distribution is fitted
import powerlaw

# Get the degree distribution
degree_values = list(dict(G.degree()).values())

# Fit the power-law distribution
fit = powerlaw.Fit(degree_values)

# Check the goodness of fit
R, p = fit.distribution_compare('power_law', 'lognormal')

if p < 0.05:
    print("The degree distribution follows a power-law distribution")
else:
    print("The degree distribution does not follow a power-law distribution")