"""
Labratory Session 2

 Manipulation using NetworkX and other python libraries

A)
  Use the karateClubDataset used in Laboratory session 1 or downloaded elsewhere https://docs.dgl.ai/en/2.0.x/generated/dgl.data.KarateClubDataset.html
   
   Write a program that 
   
  a) Displays the whole graph 
  b) Displays the degree centrality, eigenvector centrality, Katz centrality, page rank centrality of each node of the network. Draw the network graph where the node with the highest centrality is highlighted (use different color for
        each centrality type)
  c) Draws the distribution (histogram) for each centrality measure
  d) Repeat b) and c) when using closeness centrality, betweenness centrality 
  e) Displays the local clustering coefficient of each node, and draws the corresponding distribution function. Then compare possible link between clustering coefficient values and some centrality measures when scrutinizing the values
     of clustering coefficient/centrality measures of individual nodes.
  f) Calculates the global clustering coefficient of the overall graph (or its largest connected componenty). 
  g) Identify smallest subgraph that has a global clustering coefficient close to the one of the whole graph.  
  h) Identify a subgraph, which is bipartie graph 
"""

import networkx as nx
import matplotlib.pyplot as plt

#a) Displays the whole graph

# Load the Karate Club dataset
G = nx.karate_club_graph()

# Draw the graph
nx.draw(G, with_labels=True)

# Show the plot
plt.show()

#b) Displays the degree centrality, eigenvector centrality, Katz centrality, page rank centrality of each node of the network. Draw the network graph where the node with the highest centrality is highlighted (use different color for

# Compute the centralities
degree_centrality = nx.degree_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G)
katz_centrality = nx.katz_centrality(G)
pagerank = nx.pagerank(G)

# Find the nodes with the highest centralities
max_degree_node = max(degree_centrality, key=degree_centrality.get)
max_eigenvector_node = max(eigenvector_centrality, key=eigenvector_centrality.get)
max_katz_node = max(katz_centrality, key=katz_centrality.get)
max_pagerank_node = max(pagerank, key=pagerank.get)

# Draw the graph with the nodes with the highest centralities highlighted
node_colors = ["red" if node == max_degree_node else "blue" if node == max_eigenvector_node else "green" if node == max_katz_node else "yellow" if node == max_pagerank_node else "black" for node in G.nodes()]
nx.draw(G, node_color=node_colors, with_labels=True)

# Show the plot
plt.show()

#c) Draws the distribution (histogram) for each centrality measure

# Draw the histogram for degree centrality
plt.figure(figsize=(10, 6))
plt.hist(list(degree_centrality.values()))
plt.title('Degree Centrality Distribution')
plt.show()

# Draw the histogram for eigenvector centrality
plt.figure(figsize=(10, 6))
plt.hist(list(eigenvector_centrality.values()))
plt.title('Eigenvector Centrality Distribution')
plt.show()

# Draw the histogram for Katz centrality
plt.figure(figsize=(10, 6))
plt.hist(list(katz_centrality.values()))
plt.title('Katz Centrality Distribution')
plt.show()

# Draw the histogram for PageRank
plt.figure(figsize=(10, 6))
plt.hist(list(pagerank.values()))
plt.title('PageRank Distribution')
plt.show()

#d) Repeat b) and c) when using closeness centrality, betweenness centrality

# Compute the centralities
closeness_centrality = nx.closeness_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

# Find the nodes with the highest centralities
max_closeness_node = max(closeness_centrality, key=closeness_centrality.get)
max_betweenness_node = max(betweenness_centrality, key=betweenness_centrality.get)

# Draw the graph with the nodes with the highest centralities highlighted
node_colors = ["red" if node == max_closeness_node else "blue" if node == max_betweenness_node else "black" for node in G.nodes()]
nx.draw(G, node_color=node_colors, with_labels=True)

# Show the plot
plt.show()

# Draw the histogram for closeness centrality
plt.figure(figsize=(10, 6))
plt.hist(list(closeness_centrality.values()))
plt.title('Closeness Centrality Distribution')
plt.show()

# Draw the histogram for betweenness centrality
plt.figure(figsize=(10, 6))
plt.hist(list(betweenness_centrality.values()))
plt.title('Betweenness Centrality Distribution')
plt.show()

#e) Displays the local clustering coefficient of each node, and draws the corresponding distribution function. Then compare possible link between clustering coefficient values and some centrality measures when scrutinizing the values

# Compute the local clustering coefficient of each node
clustering_coefficient = nx.clustering(G)

# Display the local clustering coefficient of each node
for node, coeff in clustering_coefficient.items():
    print(f'Node {node}: {coeff}')

# Draw the distribution function for the local clustering coefficient
plt.figure(figsize=(10, 6))
plt.hist(list(clustering_coefficient.values()))
plt.title('Local Clustering Coefficient Distribution')
plt.show()

#f) Calculates the global clustering coefficient of the overall graph (or its largest connected componenty).

# Calculate the global clustering coefficient of the overall graph
global_clustering_coefficient = nx.average_clustering(G)
print(f'Global clustering coefficient of the overall graph: {global_clustering_coefficient}')

# Find the largest connected component
largest_connected_component = max(nx.connected_components(G), key=len)

# Create a subgraph for the largest connected component
largest_component_subgraph = G.subgraph(largest_connected_component)

# Calculate the global clustering coefficient of the largest connected component
global_clustering_coefficient_largest_component = nx.average_clustering(largest_component_subgraph)
print(f'Global clustering coefficient of the largest connected component: {global_clustering_coefficient_largest_component}')

#g) Identify smallest subgraph that has a global clustering coefficient close to the one of the whole graph. 

from itertools import combinations

# Calculate the global clustering coefficient of the overall graph
global_clustering_coefficient = nx.average_clustering(G)

# Initialize the smallest subgraph and its size
smallest_subgraph = None
smallest_subgraph_size = len(G)

# Iterate over all possible subgraph sizes
for size in range(1, len(G)):
    # Iterate over all possible subgraphs of the current size
    for nodes in combinations(G.nodes(), size):
        # Create a subgraph
        subgraph = G.subgraph(nodes)
        # Calculate the global clustering coefficient of the subgraph
        subgraph_clustering_coefficient = nx.average_clustering(subgraph)
        # If the global clustering coefficient of the subgraph is close to the one of the whole graph
        if abs(subgraph_clustering_coefficient - global_clustering_coefficient) < 0.01:
            # If the size of the subgraph is smaller than the current smallest subgraph
            if size < smallest_subgraph_size:
                # Update the smallest subgraph and its size
                smallest_subgraph = subgraph
                smallest_subgraph_size = size
                # Break the loop as we have found a smaller subgraph
                break
    # If we have found a smallest subgraph, break the loop
    if smallest_subgraph is not None:
        break

# Print the nodes of the smallest subgraph
print(smallest_subgraph.nodes()) 

#h) Identify a subgraph, which is bipartie graph 

from networkx.algorithms import bipartite

# Initialize the bipartite subgraph
bipartite_subgraph = None

# Iterate over all possible subgraph sizes
for size in range(1, len(G)):
    # Iterate over all possible subgraphs of the current size
    for nodes in combinations(G.nodes(), size):
        # Create a subgraph
        subgraph = G.subgraph(nodes)
        # If the subgraph is bipartite
        if bipartite.is_bipartite(subgraph):
            # Update the bipartite subgraph
            bipartite_subgraph = subgraph
            # Break the loop as we have found a bipartite subgraph
            break
    # If we have found a bipartite subgraph, break the loop
    if bipartite_subgraph is not None:
        break

# Print the nodes of the bipartite subgraph
if bipartite_subgraph is not None:
    print(bipartite_subgraph.nodes())
else:
    print("No bipartite subgraph found.")
