

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import json

# Create a research group network with the actual members
# Joseph Nii Lante Lamptey is the team lead, with 4 other research members

G = nx.Graph()

# Define research group members
members = {
    'Joseph Nii Lante Lamptey': 'Team Lead',
    'Isaac Frimpong Asante': 'Research Member',
    'Abigail The Obeng-Asamoah': 'Research Member',
    'Raphael Anaafi': 'Research Member',
    'Godfred Kwabena Lumor': 'Research Member',
}

# Add nodes with attributes
for name, role in members.items():
    G.add_node(name, role=role)

# Add edges representing collaboration relationships
# Creating a realistic collaboration network
collaborations = [
    # Team lead connections with all members (as project coordinator)
    ('Joseph Nii Lante Lamptey', 'Isaac Frimpong Asante', 'project collaboration'),
    ('Joseph Nii Lante Lamptey', 'Abigail The Obeng-Asamoah', 'project collaboration'),
    ('Joseph Nii Lante Lamptey', 'Raphael Anaafi', 'project collaboration'),
    ('Joseph Nii Lante Lamptey', 'Godfred Kwabena Lumor', 'project collaboration'),
    
    # Peer collaborations among research members
    ('Isaac Frimpong Asante', 'Abigail The Obeng-Asamoah', 'peer collaboration'),
    ('Isaac Frimpong Asante', 'Raphael Anaafi', 'peer collaboration'),
    ('Isaac Frimpong Asante', 'Godfred Kwabena Lumor', 'peer collaboration'),
    
    ('Abigail The Obeng-Asamoah', 'Godfred Kwabena Lumor', 'peer collaboration'),
    ('Abigail The Obeng-Asamoah', 'Raphael Anaafi', 'peer collaboration'),
    
    ('Raphael Anaafi', 'Godfred Kwabena Lumor', 'peer collaboration'),
]

# Add edges to graph
for person1, person2, rel_type in collaborations:
    G.add_edge(person1, person2, relationship=rel_type)

# Compute network statistics
print("=" * 80)
print("RESEARCH GROUP NETWORK ANALYSIS")
print("=" * 80)

# 1. Number of nodes and edges
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

print(f"\n1. BASIC NETWORK METRICS")
print(f"   Number of nodes (members): {num_nodes}")
print(f"   Number of edges (collaborations): {num_edges}")
print(f"   Network density: {nx.density(G):.3f}")
print(f"   Maximum possible edges: {num_nodes * (num_nodes - 1) // 2}")

# 2. Degree distribution
degrees = dict(G.degree())
degree_values = list(degrees.values())
degree_counts = Counter(degree_values)

print(f"\n2. DEGREE DISTRIBUTION")
print(f"   Average degree: {np.mean(degree_values):.2f}")
print(f"   Median degree: {np.median(degree_values):.1f}")
print(f"   Maximum degree: {max(degree_values)}")
print(f"   Minimum degree: {min(degree_values)}")
print(f"   Standard deviation: {np.std(degree_values):.2f}")

print(f"\n   Degree frequency:")
for degree in sorted(degree_counts.keys()):
    print(f"   Degree {degree}: {degree_counts[degree]} member(s)")

print(f"\n   Individual member connections:")
sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
for i, (member, degree) in enumerate(sorted_degrees, 1):
    role = G.nodes[member]['role']
    print(f"   {i}. {member} ({role}): {degree} connections")

# 3. Check for isolated nodes
isolated_nodes = list(nx.isolates(G))
print(f"\n3. ISOLATED NODES ANALYSIS")
if isolated_nodes:
    print(f"   ⚠ Number of isolated nodes: {len(isolated_nodes)}")
    for node in isolated_nodes:
        role = G.nodes[node]['role']
        print(f"   - {node} ({role})")
else:
    print(f"   ✓ No isolated nodes found")
    print(f"   ✓ All {num_nodes} members are connected to the research network")

# Additional network metrics
print(f"\n4. CONNECTIVITY ANALYSIS")
print(f"   Is network connected: {nx.is_connected(G)}")
if nx.is_connected(G):
    print(f"   Average shortest path length: {nx.average_shortest_path_length(G):.2f}")
    print(f"   Network diameter (max distance): {nx.diameter(G)}")
    print(f"   Network radius (min eccentricity): {nx.radius(G)}")
print(f"   Number of connected components: {nx.number_connected_components(G)}")
print(f"   Average clustering coefficient: {nx.average_clustering(G):.3f}")

# Centrality measures
betweenness = nx.betweenness_centrality(G)
closeness = nx.closeness_centrality(G)
eigenvector = nx.eigenvector_centrality(G)

print(f"\n5. CENTRALITY ANALYSIS")
print(f"\n   Betweenness Centrality (Bridge Connectors):")
sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
for i, (member, score) in enumerate(sorted_betweenness, 1):
    print(f"   {i}. {member}: {score:.3f}")

print(f"\n   Closeness Centrality (Information Spreaders):")
sorted_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
for i, (member, score) in enumerate(sorted_closeness, 1):
    print(f"   {i}. {member}: {score:.3f}")

print(f"\n   Eigenvector Centrality (Influence in Network):")
sorted_eigenvector = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)
for i, (member, score) in enumerate(sorted_eigenvector, 1):
    print(f"   {i}. {member}: {score:.3f}")

# Individual eccentricity (maximum distance to any other node)
if nx.is_connected(G):
    print(f"\n6. ECCENTRICITY ANALYSIS")
    eccentricity = nx.eccentricity(G)
    print(f"   Member eccentricity (max distance to reach any other member):")
    for member, ecc in sorted(eccentricity.items(), key=lambda x: x[1]):
        print(f"   {member}: {ecc}")

# Save statistics to file
stats = {
    'num_nodes': num_nodes,
    'num_edges': num_edges,
    'density': float(nx.density(G)),
    'degree_distribution': dict(degree_counts),
    'average_degree': float(np.mean(degree_values)),
    'median_degree': float(np.median(degree_values)),
    'isolated_nodes': isolated_nodes,
    'is_connected': nx.is_connected(G),
    'num_components': nx.number_connected_components(G),
    'clustering_coefficient': float(nx.average_clustering(G)),
    'members': list(members.keys()),
}

if nx.is_connected(G):
    stats['diameter'] = nx.diameter(G)
    stats['avg_shortest_path'] = float(nx.average_shortest_path_length(G))

with open('network_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("\n" + "=" * 80)

# Create visualization
fig = plt.figure(figsize=(14, 10))

# Define colors for different roles
role_colors = {
    'Team Lead': '#FF6B6B',  # Red
    'Research Member': '#4ECDC4',  # Teal
}

# Get node colors based on roles
node_colors = [role_colors[G.nodes[node]['role']] for node in G.nodes()]

# Use spring layout for better visualization
pos = nx.spring_layout(G, k=1.8, iterations=50, seed=42)

# Draw the network
nx.draw_networkx_nodes(G, pos, 
                       node_color=node_colors,
                       node_size=5500,
                       alpha=0.9,
                       edgecolors='black',
                       linewidths=2.5)

# Adjust label positions and sizes
nx.draw_networkx_labels(G, pos, 
                        font_size=9,
                        font_weight='bold',
                        font_family='sans-serif')

nx.draw_networkx_edges(G, pos, 
                       width=2.5,
                       alpha=0.4,
                       edge_color='gray')

# Add title
plt.title('Research Group Collaboration Network\n5-Member Research Team', 
          fontsize=18, fontweight='bold', pad=20)

# Create legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FF6B6B', edgecolor='black', label='Team Lead'),
    Patch(facecolor='#4ECDC4', edgecolor='black', label='Research Members')
]
plt.legend(handles=legend_elements, 
          loc='upper left',
          fontsize=11,
          title='Member Roles',
          title_fontsize=12,
          framealpha=0.9)

# Add network statistics text
stats_text = f"Nodes: {num_nodes} | Edges: {num_edges} | Density: {nx.density(G):.3f} | "
stats_text += f"Avg Degree: {np.mean(degree_values):.2f} | Connected: {'Yes' if nx.is_connected(G) else 'No'}"
if isolated_nodes:
    stats_text += f" | Isolated: {len(isolated_nodes)}"
    
plt.text(0.5, 0.02, stats_text,
         ha='center',
         transform=fig.transFigure,
         fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.axis('off')
plt.tight_layout()

# Save the visualization to current directory
plt.savefig('research_network_visualization.png', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white')
print(f"\n✓ Main visualization saved: research_network_visualization.png")

# Create degree distribution plot - Individual connections only
fig2 = plt.figure(figsize=(10, 6))

# Bar chart of individual members
members_list = [m for m, _ in sorted_degrees]
degrees_list_sorted = [d for _, d in sorted_degrees]
colors = ['#FF6B6B' if G.nodes[m]['role'] == 'Team Lead' else '#4ECDC4' 
          for m in members_list]

# Use shorter labels for readability
short_labels = []
for m in members_list:
    if 'Lamptey' in m:
        short_labels.append('Lamptey (Lead)')
    else:
        short_labels.append(m.split()[-1])

plt.barh(range(len(members_list)), degrees_list_sorted, 
         color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
plt.yticks(range(len(members_list)), short_labels, fontsize=11)
plt.xlabel('Number of Connections', fontsize=13, fontweight='bold')
plt.ylabel('Team Member', fontsize=13, fontweight='bold')
plt.title('Individual Member Connection Counts', fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x', linestyle='--')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('degree_distribution.png', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white')
print(f"✓ Degree distribution plot saved: degree_distribution.png")

# Create a more detailed network with edge labels
fig3 = plt.figure(figsize=(16, 12))
pos2 = nx.spring_layout(G, k=2.5, iterations=50, seed=42)

nx.draw_networkx_nodes(G, pos2, 
                       node_color=node_colors,
                       node_size=7000,
                       alpha=0.9,
                       edgecolors='black',
                       linewidths=2.5)

nx.draw_networkx_labels(G, pos2, 
                        font_size=10,
                        font_weight='bold')

nx.draw_networkx_edges(G, pos2, 
                       width=2,
                       alpha=0.3,
                       edge_color='gray')

# Add edge labels showing relationship types
edge_labels = nx.get_edge_attributes(G, 'relationship')
nx.draw_networkx_edge_labels(G, pos2, edge_labels, 
                             font_size=7, 
                             font_color='red',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.title('Research Group Network with Collaboration Types', 
          fontsize=18, fontweight='bold', pad=20)
plt.legend(handles=legend_elements, 
          loc='upper left',
          fontsize=11,
          framealpha=0.9)
plt.axis('off')
plt.tight_layout()

plt.savefig('network_with_labels.png', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white')
print(f"✓ Detailed network with labels saved: network_with_labels.png")

# Display all figures (optional - comment out if running in non-interactive mode)
plt.show()

print("\n✓ All visualizations and statistics generated successfully!")
print("=" * 80)