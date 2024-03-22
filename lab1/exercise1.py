"""
1) Write a full graph of 50 nodes where from each node, there is a link (either in-going or out-going links) to one to 4 other nodes, taken at random, of your choice. 
    Use a labelling of your choice to label each node of the network.
 
 a) -  Use a visualization tool to display the graph     
 
 b) -  Use a visualization of your choice to display the nodes each node is linked to

 c) -  Calculate the degree centrality of each node and the average degree of the graph  (use appropriate functions in 
    NetworkX) and display their values

 d) - Draw the degree distribution plot and comment on whether the power-law distribution is fit 

 e) - Test other centrality measures available in NetworkX and display their values, and store the centrality values 
      in a vector

 f) - write a script that randomly removes one node from the above graph

     - Repeat the process a)-d)  until the number of nodes in the graph is equal to one.

 g ) Display a graph showing the variations of the various centrality measures as a function of the number of 
edges in the graph.
"""
import networkx as nx
import matplotlib.pyplot as plt
import random

# Create an empty directed graph
G = nx.DiGraph()

# Add 50 nodes to the graph
G.add_nodes_from(range(1, 51))

# For each node, add edges to 1 to 4 other nodes chosen randomly
for node in G.nodes:
    num_edges = random.randint(1, 4)
    edges = random.sample(list(G.nodes), num_edges)
    for edge in edges:
        if edge != node:  # Prevent self-loops
            G.add_edge(node, edge)

# Draw the graph with node labels
nx.draw(G, with_labels=True, node_color='skyblue', node_size=300)

# Display the graph using matplotlib
plt.show()

# c) -  Calculate the degree centrality of each node and the average degree of the graph 
degrees = [deg for node, deg in G.degree()]     # Get the degree of each node
average_degree = sum(degrees) / len(degrees)
print(f"The average degree of the graph is {average_degree}")


 # d) - Draw the degree distribution plot and comment on whether the power-law distribution is fit 
"""
If the plot is a straight line on a log-log scale, then the degree distribution follows a power-law and
the network can be considered a scale-free network.
If it doesn't, then the network does not follow a power-law distribution.
"""
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

#e) - Test other centrality measures available in NetworkX and display their values, and store the centrality values 
      #in a vector

# Closeness Centrality
closeness_centrality = nx.closeness_centrality(G)
print("Closeness Centrality:")
for node, centrality in closeness_centrality.items():
    print(f"Node {node}: {centrality}")

# Betweenness Centrality
betweenness_centrality = nx.betweenness_centrality(G)
print("\nBetweenness Centrality:")
for node, centrality in betweenness_centrality.items():
    print(f"Node {node}: {centrality}")

# Eigenvector Centrality
eigenvector_centrality = nx.eigenvector_centrality(G)
print("\nEigenvector Centrality:")
for node, centrality in eigenvector_centrality.items():
    print(f"Node {node}: {centrality}")

#f) - write a script that randomly removes one node from the above graph

     #- Repeat the process a)-d)  until the number of nodes in the graph is equal to one.

def calculate_and_display_centrality(G):
    degrees = [deg for node, deg in G.degree()]     # Get the degree of each node
    average_degree = sum(degrees) / len(degrees)

def draw_degree_distribution(G):
    import collections

    # Count the frequency of each degree value
    degreeCount = collections.Counter(degrees) 
    deg, cnt = zip(*degreeCount.items()) # Unzip the degree and count values

# Main loop
while len(G.nodes) > 1:
    # Step a) - Remove a random node
    node_to_remove = random.choice(list(G.nodes))
    G.remove_node(node_to_remove)

    # Step c) - Calculate and display centrality
    calculate_and_display_centrality(G)

    # Step d) - Draw degree distribution and check for power-law distribution
    draw_degree_distribution(G)

nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500)
plt.show()

 #g ) Display a graph showing the variations of the various centrality measures as a function of the number of 
    #edges in the graph.
import networkx as nx
import matplotlib.pyplot as plt
import random

# Initialize the graph G here

# Initialize lists to store values
num_edges = []
closeness_values = []
betweenness_values = []
eigenvector_values = []

# Main loop
while len(G.nodes) > 1:
    # Remove a random node
    node_to_remove = random.choice(list(G.nodes))
    G.remove_node(node_to_remove)

    # Check if the graph is connected
    if nx.is_connected(G):
        # Store number of edges
        num_edges.append(G.number_of_edges())

        # Calculate centrality measures
        closeness_centrality = nx.closeness_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G)

        # Store average centrality measures
        closeness_values.append(sum(closeness_centrality.values()) / len(closeness_centrality))
        betweenness_values.append(sum(betweenness_centrality.values()) / len(betweenness_centrality))
        eigenvector_values.append(sum(eigenvector_centrality.values()) / len(eigenvector_centrality))

# Plot centrality measures
plt.figure(figsize=(10, 6))
plt.plot(num_edges, closeness_values, label='Closeness Centrality')
plt.plot(num_edges, betweenness_values, label='Betweenness Centrality')
plt.plot(num_edges, eigenvector_values, label='Eigenvector Centrality')
plt.xlabel('Number of Edges')
plt.ylabel('Average Centrality')
plt.title('Centrality Measures as a Function of the Number of Edges')
plt.legend()
plt.show()