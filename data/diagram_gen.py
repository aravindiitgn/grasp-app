import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Define nodes and their labels
nodes = {
    "start": "Start",
    "input_features": "Input Features for\nObjects A and B",
    "feature_vectors": "Feature Vectors\n(x1 and x2)",
    "normalize": "Normalize Features\nusing MinMax Scaler",
    "predict": "Predict Remaining\nWorkspace Volumes\n(GRB Model)",
    "calculate_volumes": "Calculate Total Volumes\nV_total^1 and V_total^2",
    "compare_volumes": "Compare Volumes\nV_total^1 and V_total^2",
    "optimal_sequence": "Select Optimal\nGrasp Sequence"
}

# Add nodes to the graph
for node_id, label in nodes.items():
    G.add_node(node_id, label=label)

# Define edges (connections between nodes)
edges = [
    ("start", "input_features"),
    ("input_features", "feature_vectors"),
    ("feature_vectors", "normalize"),
    ("normalize", "predict"),
    ("predict", "calculate_volumes"),
    ("calculate_volumes", "compare_volumes"),
    ("compare_volumes", "optimal_sequence")
]

# Add edges to the graph
G.add_edges_from(edges)

# Set node positions for layout
pos = {
    "start": (0, 6),
    "input_features": (0, 5),
    "feature_vectors": (0, 4),
    "normalize": (0, 3),
    "predict": (0, 2),
    "calculate_volumes": (0, 1),
    "compare_volumes": (0, 0),
    "optimal_sequence": (0, -1)
}

# Draw the nodes and edges
plt.figure(figsize=(10, 8))

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='#A0CBE2', edgecolors='black')

# Draw edges
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=30, edge_color='black')

# Draw labels
labels = nx.get_node_attributes(G, 'label')
nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color='black')

# Title and layout adjustments
plt.title("Grasp Sequence Prediction Flowchart", fontweight="bold", fontsize=12)
plt.axis('off')  # Turn off the axis

# Show the flowchart
plt.show()
