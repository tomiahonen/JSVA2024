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


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

G=nx.read_edgelist('facebook_combined.txt', create_using=nx.Graph(),nodetype=int)

#print(nx.info(g))'


#sp=nx.spring_layout(g)

#plt.axes('off')

nx.draw(G,with_labels=False,node_size=35)
plt.show()

degree_centrality_dict = nx.degree_centrality(G)
degree_centrality_array = np.array(list(degree_centrality_dict.values()))
highest_degree_node = max(degree_centrality_dict, key=degree_centrality_dict.get)
#print(degree_centrality_array)

closeness_centrality_dict = nx.closeness_centrality(G)
closeness_centrality_array = np.array(list(closeness_centrality_dict.values()))
highest_cc = max(closeness_centrality_dict, key=closeness_centrality_dict.get)
#print(degree_centrality_dict)

betweenes_centrality_dict = nx.betweenness_centrality(G)
betweenes_centrality_array = np.array(list(betweenes_centrality_dict.values()))
highest_bc = max(betweenes_centrality_dict, key=betweenes_centrality_dict.get)
#print(betweenes_centrality_dict)


plt.figure(figsize=(8, 4))

plt.subplot(2, 2, 1)
plt.hist(degree_centrality_array, bins=10, color='skyblue', edgecolor='black')
plt.title('Degree Centrality Histogram')
plt.xlabel('Degree Centrality')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
plt.hist(closeness_centrality_array, bins=10, color='salmon', edgecolor='black')
plt.title('Eigenvector Centrality Histogram')
plt.xlabel('Eigenvector Centrality')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
plt.hist(betweenes_centrality_array, bins=10, color='green', edgecolor='black')
plt.title('Katz Centrality Histogram')
plt.xlabel('Katz Centrality')
plt.ylabel('Frequency')


plt.tight_layout()
plt.show()