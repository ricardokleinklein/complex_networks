from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import networkx as nx

from networkx.algorithms.approximation.clustering_coefficient import average_clustering
from networkx.algorithms.assortativity import degree_assortativity_coefficient

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


def _get_min_degree(G):
	return min(d[1] for d in G.degree)


def _get_max_degree(G):
	return max(d[1] for d in G.degree)


def _get_mean_degree(G):
	return get_num_edges(G) / get_num_vertices(G)


def get_degree_info(G):
	min_ = _get_min_degree(G)
	max_ = _get_max_degree(G)
	mean_ = _get_mean_degree(G)
	return min_, max_, mean_


def get_ACC(G):
	"""Average Clustering Coefficient."""
	G = nx.Graph(G)
	return nx.average_clustering(G)


def get_assortativity(G):
	return degree_assortativity_coefficient(G)


def draw_graph(G, name):
	nx.draw_networkx(G, with_labels=True, font_weight='bold',
		label=name)


def write_file(table, dst_dir):
	with open(dst_dir, 'w') as f:
		f.write('dir\t' + 'graph\t' + 'order\t' + 'size\t' + 'degrees\t' +
			'ACC\t' + 'assortativity\t' + '\n')
		for line in table:
			parts = [str(obj) for obj in line]
			line = "\t".join(parts)
			f.write(line + "\n")