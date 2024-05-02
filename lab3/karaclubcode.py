"""
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community.centrality import girvan_newman

G = nx.karate_club_graph()
communities = girvan_newman(G)

node_groups = []  
for com in next(communities):
    node_groups.append(list(com))
    
print(node_groups)

color_map = []
for node in G:
    if node in node_groups[0]:
        color_map.append('blue')
    else:
        color_map.append('green')

nx.draw(G, node_color=color_map, with_labels=True)
plt.show() """

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community.centrality import girvan_newman

G = nx.karate_club_graph()
communities = girvan_newman(G)

node_groups = []  
for com in next(communities):
    node_groups.append(list(com))

# Create a color map and size map (larger sizes for nodes with higher degree)
color_map = []
size_map = []
for node in G:
    if node in node_groups[0]:
        color_map.append('blue')
    else:
        color_map.append('green')
    size_map.append(100 * G.degree(node))

# Draw the graph with node labels
pos = nx.spring_layout(G)  # compute graph layout
nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=size_map)  # draw nodes with color and size
nx.draw_networkx_edges(G, pos, alpha=0.5)  # draw edges
nx.draw_networkx_labels(G, pos)  # draw node labels/names

plt.axis('off')  # turn off the axis
plt.show()  # display the graph

#1. This alternative version uses the spring_layuout function to compute the position of the node, which 
# results in a more readable graph.

#2. We want to study the result of community detection using Ratio Cut Method.

from networkx.algorithms import community

# Perform community detection using the Ratio Cut method
communities = community.kernighan_lin_bisection(G)

# Save the results in separate variables
community1, community2 = communities

# Create a color map (different colors for different communities)
color_map = []
for node in G:
    if node in community1:
        color_map.append('blue')
    else:  # node in community2
        color_map.append('green')

# Draw the graph with node labels
pos = nx.spring_layout(G)  # compute graph layout
nx.draw_networkx_nodes(G, pos, node_color=color_map)  # draw nodes with color
nx.draw_networkx_edges(G, pos, alpha=0.5)  # draw edges
nx.draw_networkx_labels(G, pos)  # draw node labels/names

plt.axis('off')  
plt.show() 

#3. Repeat 2) when using the Louvain community detection algorithm, which has inbuilt function in 
#networkX
import community as community_louvain
import time

# Perform community detection using the Louvain method
partition = community_louvain.best_partition(G)

# Create a color map (different colors for different communities)
color_map = [partition[node] for node in G.nodes]

# Draw the graph with node labels
pos = nx.spring_layout(G)  # compute graph layout
nx.draw_networkx_nodes(G, pos, node_color=color_map, cmap=plt.cm.jet)  # draw nodes with color
nx.draw_networkx_edges(G, pos, alpha=0.5)  # draw edges
nx.draw_networkx_labels(G, pos)  # draw node labels/names

plt.axis('off')
plt.show()  

#4. Repeat 2) when using label-propagation community detection algorithm. 

# Perform community detection using the label propagation method
communities = list(community.label_propagation_communities(G))

# Save the results in separate variables
community1, community2, *_ = communities

# Create a color map (different colors for different communities)
color_map = []
for node in G:
    if node in community1:
        color_map.append('blue')
    else:  # node in community2
        color_map.append('green')

# Draw the graph with node labels
pos = nx.spring_layout(G)  # compute graph layout
nx.draw_networkx_nodes(G, pos, node_color=color_map)  # draw nodes with color
nx.draw_networkx_edges(G, pos, alpha=0.5)  # draw edges
nx.draw_networkx_labels(G, pos)  # draw node labels/names

plt.axis('off')  # turn off the axis
plt.show()  # display the graph

#5. We want to evaluate the quality of the communities detected in 1)-4). 

# Girvan-Newman method
start_time = time.time()
communities = community.girvan_newman(G)
top_level_communities = next(communities)
girvan_newman_modularity = community.modularity(G, top_level_communities)
print(f'Girvan-Newman modularity: {girvan_newman_modularity}')
end_time = time.time()
print(f'Girvan-Newman execution time: {end_time - start_time} seconds')

# Ratio Cut method (Kernighan–Lin bisection)
start_time = time.time()
communities = community.kernighan_lin_bisection(G)
ratio_cut_modularity = community.modularity(G, communities)
print(f'Ratio Cut modularity: {ratio_cut_modularity}')
end_time = time.time()
print(f'Ratio Cut execution time: {end_time - start_time} seconds')

# Label propagation method
start_time = time.time()
communities = list(community.label_propagation_communities(G))
label_propagation_modularity = community.modularity(G, communities)
print(f'Label propagation modularity: {label_propagation_modularity}')
end_time = time.time()
print(f'Label propagation execution time: {end_time - start_time} seconds')

# Louvain method
start_time = time.time()
partition = community_louvain.best_partition(G)
louvain_communities = list(set(partition.values()))
louvain_modularity = community_louvain.modularity(partition, G)
print(f'Louvain modularity: {louvain_modularity}')
end_time = time.time()
print(f'Louvain execution time: {end_time - start_time} seconds')

#6. We want to compare the performance of the algorithms in 1)-4) in terms of algorithmic 
#complexity. It is implemented above.

#7. We want to seek a random graph that has close characteristic to Karate graph in terms of 
#communities. 

# Detect communities in the Karate Club graph using the Girvan-Newman method
communities = community.girvan_newman(G)
top_level_communities = next(communities)
num_communities_karate = len(top_level_communities)

# Get the number of nodes and edges in the Karate Club graph
n = G.number_of_nodes()
m = G.number_of_edges()

closest_graph = None
closest_num_communities = None

# Generate various random graphs by varying the seed value
for seed in range(100):
    # Generate a random graph with the same number of nodes and edges
    random_G = nx.gnm_random_graph(n, m, seed)

    # Detect communities in the random graph using the Girvan-Newman method
    communities = community.girvan_newman(random_G)
    top_level_communities = next(communities)
    num_communities_random = len(top_level_communities)

    # Check if this random graph is closer to the Karate Club graph in terms of number of communities
    if closest_graph is None or abs(num_communities_random - num_communities_karate) < abs(closest_num_communities - num_communities_karate):
        closest_graph = random_G
        closest_num_communities = num_communities_random

# Now closest_graph is the random graph that is closest to the Karate Club graph in terms of number of communities
from networkx.algorithms.community import LFR_benchmark_graph

# Generate an LFR benchmark graph with overlapping communities
mu = 0.1
tau1 = 3
tau2 = 1.5
n = 1000
k = 20
minc = 20
maxc = 50
G = LFR_benchmark_graph(n, tau1, tau2, mu, min_degree=k, 
                         max_degree=k, min_community=minc, 
                         max_community=maxc, seed=10)

# Detect communities in the LFR graph using the Girvan-Newman method
communities = community.girvan_newman(G)
top_level_communities = next(communities)
num_communities_lfr = len(top_level_communities)

# Get the number of nodes and edges in the LFR graph
n = G.number_of_nodes()
m = G.number_of_edges()

closest_graph = None
closest_num_communities = None

# Generate various random graphs by varying the seed value
for seed in range(100):
    # Generate a random graph with the same number of nodes and edges
    random_G = nx.gnm_random_graph(n, m, seed)

    # Detect communities in the random graph using the Girvan-Newman method
    communities = community.girvan_newman(random_G)
    top_level_communities = next(communities)
    num_communities_random = len(top_level_communities)

    # Check if this random graph is closer to the LFR graph in terms of number of communities
    if closest_graph is None or abs(num_communities_random - num_communities_lfr) < abs(closest_num_communities - num_communities_lfr):
        closest_graph = random_G
        closest_num_communities = num_communities_random

# Now closest_graph is the random graph that is closest to the LFR graph in terms of number of communities