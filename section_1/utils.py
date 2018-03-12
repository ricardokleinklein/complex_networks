from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import networkx as nx

def _rm_hidden(files):
	return [file for file in files if not file.startswith(".")]

def load_graph(path):
	return nx.read_pajek(path)

def compile_graph_libs(in_dir):
	"""Return a dictionary with all the graphs available."""
	dirs = _rm_hidden(os.listdir(in_dir))
	graphs = dict()
	for d in dirs:
		g = _rm_hidden(os.listdir(
			os.path.join(in_dir, d)))
		graphs[d] = g
	return graphs

def get_num_vertices(G):
	return nx.number_of_nodes(G)

def get_num_edges(G):
	return nx.number_of_edges(G)

def draw_graph(G, name):
	nx.draw_networkx(G, with_labels=True, font_weight='bold',
		label=name)