import networkx as nx

graph = nx.Graph()
graph.add_node('decadence')
nx.write_gml(graph, '../database/graph.gml')
