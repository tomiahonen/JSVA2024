"""

 2) Use the provided dataset karate_club_coords.pkl.   
   Write a program that 
   
  a) Inputs the above dataset 
  b) Displays the adjacency matrix of this graph and the network associated to this dataset
  c) Calculates the degree centrality of each node and store them in an array vector
  d) Identifies potential regular graphs in the network. 
  e) Uses appropriate NetworkX functions to identify the largest component of the graph, and smallest component. 
  f) Draw the degree distribution of this component (subgraph of d)). 
  g) Use appropriate NetworkX functions to compute the diameter of the whole network and diameter of the largest component.
"""
# a) Inputs the above dataset 

import pickle

with open('C:\\Users\\OMISTAJA\\Desktop\\JSVA2024\\JSVA2024\\lab1\\karate_club_coords.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')
# b) Displays the adjacency matrix of this graph and the network associated to this dataset
    
import networkx as nx
import matplotlib.pyplot as plt

# Create a graph from the adjacency matrix
G = nx.karate_club_graph()

# Display the adjacency matrix
print(nx.adjacency_matrix(G).todense())

# Draw the network
nx.draw(G, with_labels=True, font_size=3, node_size=300)
plt.show()

# c) Calculates the degree centrality of each node and store them in an array vector
import numpy as np

# Assuming 'G' is your graph
degree_centrality_dict = nx.degree_centrality(G)

# Get the degree centrality values and convert them to an array
degree_centrality_array = np.array(list(degree_centrality_dict.values()))

# d) Identifies potential regular graphs in the network.

# Get the degrees of all nodes
degrees = [degree for node, degree in G.degree()]

# Check if all degrees are the same
is_regular = len(set(degrees)) == 1

print("Is the graph regular?", is_regular)

# e) Uses appropriate NetworkX functions to identify the largest component of the graph, and smallest component.

# Get the connected components
components = nx.connected_components(G)

# Convert to a list so we can use it multiple times
components = list(components)
print(components)

# Find the largest component
largest_component = max(components, key=len)
# Find the smallest component
smallest_component = min(components, key=len)

print("Largest component:", len(largest_component))
print("Smallest component:", len(smallest_component))

# f) Draw the degree distribution of this component (subgraph of d)).
import collections

# Count the frequency of each degree value
degreeCount = collections.Counter(degrees) 
deg, cnt = zip(*degreeCount.items()) # Unzip the degree and count values

fig, ax = plt.subplots()
plt.bar(deg, cnt, width=0.80, color="b")

plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
ax.set_xticks([d for d in deg])
ax.set_xticklabels(deg)

# Draw the plot
plt.show()

# g) Use appropriate NetworkX functions to compute the diameter of the whole network and diameter of the largest component.
diameter_network = nx.diameter(G)
print("Diameter of the whole network", diameter_network)

diameter_largest = nx.diameter(nx.subgraph(G, largest_component))
print("Diameter of the largest component", diameter_largest)
