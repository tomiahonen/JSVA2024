"""
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

import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

with open('C:\\Users\\Tuomas\\Documents\\JVSNA\JSVA2024\\lab2\\karate_club_coords.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

G = nx.karate_club_graph()
nx.draw(G, with_labels=True, font_size=3, node_size=300)
plt.show()

#calculating degreen centrality, eigenvector centrality, Katz centraility, page rank centrality of each node

#same as lab 1, degree centrality
degree_centrality_dict = nx.degree_centrality(G)
degree_centrality_array = np.array(list(degree_centrality_dict.values()))
highest_degree_node = max(degree_centrality_dict, key=degree_centrality_dict.get)
print(degree_centrality_array)

#eigenvector
eigenvector_centrality_dict = nx.eigenvector_centrality(G)
eigenvector_centrality_array = np.array(list(eigenvector_centrality_dict.values()))
highest_eigenvector_node = max(eigenvector_centrality_dict, key=eigenvector_centrality_dict.get)
print(eigenvector_centrality_array)

#Katz centrality
katz_centrality_dict = nx.katz_centrality(G)
katz_centrality_array = np.array(list(katz_centrality_dict.values()))
highest_katz_node = max(katz_centrality_dict, key=katz_centrality_dict.get)

#Page rank centrality
page_centrality_dict = nx.pagerank(G)
page_centrality_array = np.array(list(page_centrality_dict.values()))
highest_page_node = max(page_centrality_dict, key=page_centrality_dict.get)

#highlight 
color_list=['green', 'blue', 'black', 'yellow']

pos = nx.spring_layout(G)
#node_colors = ['red' if node == highest_degree_node  else 'red' for node in G.nodes()]
color_map=[]
for node in G.nodes():
    if node == highest_degree_node:
        color_map.append('red')
    elif node == highest_eigenvector_node:
        color_map.append('green')
    elif node == highest_katz_node:
        color_map.append('yellow')
    elif node == highest_page_node:
        color_map.append('black')
    else:
        color_map.append('blue')



nx.draw(G, with_labels=True, font_size=8, node_size=300, node_color=color_map)
#nx.draw(G, pos, with_labels=True, node_size=300)


#nx.draw_networkx_nodes(G, pos, nodelist=[node], )
plt.show()


#C Drawing histograms

plt.figure(figsize=(10, 5))

plt.subplot(2, 2, 1)
plt.hist(degree_centrality_array, bins=10, color='skyblue', edgecolor='black')
plt.title('Degree Centrality Histogram')
plt.xlabel('Degree Centrality')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
plt.hist(eigenvector_centrality_array, bins=10, color='salmon', edgecolor='black')
plt.title('Eigenvector Centrality Histogram')
plt.xlabel('Eigenvector Centrality')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
plt.hist(katz_centrality_array, bins=10, color='green', edgecolor='black')
plt.title('Katz Centrality Histogram')
plt.xlabel('Katz Centrality')
plt.ylabel('Frequency')

plt.subplot(2, 2, 4)
plt.hist(page_centrality_array, bins=10, color='orange', edgecolor='black')
plt.title('Page Centrality Histogram')
plt.xlabel('Page Centrality')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


#closeness centrality and betweeness centrality

closeness_centrality_dict = nx.closeness_centrality(G)
closeness_centrality_array = np.array(list(closeness_centrality_dict.values()))
highest_cc = max(closeness_centrality_dict, key=closeness_centrality_dict.get)



betweenes_centrality_dict = nx.betweenness_centrality(G)
betweenes_centrality_array = np.array(list(betweenes_centrality_dict.values()))
highest_bc = max(betweenes_centrality_dict, key=betweenes_centrality_dict.get)

#node_colors = ['blue' if ( node != highest_degree_node and node != highest_bc ) else 'red' for node in G.nodes()]
color_map2=[]
for node in G.nodes():
    if node == highest_cc:
        color_map2.append('maroon')
    elif node == highest_bc:
        color_map2.append('green')
    else:
        color_map2.append('blue')

nx.draw(G, with_labels=True, font_size=8, node_size=300, node_color=color_map2)

plt.show()

#clustering
clustering_dict = nx.clustering(G)


#sets
set = nx.is_bipartite(G)
print(set)