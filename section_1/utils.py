from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import networkx as nx
from tqdm import tqdm

from networkx.algorithms.approximation.clustering_coefficient import average_clustering
from networkx.algorithms.assortativity import degree_assortativity_coefficient
from networkx.algorithms.distance_measures import diameter
from networkx.algorithms.shortest_paths.generic import average_shortest_path_length

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


def _get_num_vertices(G):
	return nx.number_of_nodes(G)


def _get_num_edges(G):
	return nx.number_of_edges(G)


def _get_min_degree(G):
	return min(d[1] for d in G.degree)


def _get_max_degree(G):
	return max(d[1] for d in G.degree)


def _get_mean_degree(G):
	return get_num_edges(G) / get_num_vertices(G)


def _get_degree_info(G):
	min_ = _get_min_degree(G)
	max_ = _get_max_degree(G)
	mean_ = _get_mean_degree(G)
	return min_, max_, mean_


def _get_ACC(G):
	"""Average Clustering Coefficient."""
	G = nx.Graph(G)
	return nx.average_clustering(G)


def _get_assortativity(G):
	return degree_assortativity_coefficient(G)


def _get_APL(G):
	"""Average Path Length."""
	return average_shortest_path_length(G)


def _get_diameter(G):
	return diameter(G)


def draw_graph(G, name=None):
	nx.draw_networkx(G, with_labels=True, font_weight='bold',
		label=name)


def _write_file(table, dst_dir):
	with open(dst_dir, 'w') as f:
		f.write('dir\t' + 'graph\t' + 'order\t' + 'size\t' + 'degrees\t' +
			'ACC\t' + 'assortativity\t' + 'avg_path_length\t' + 
			'diameter' + '\n')
		for line in table:
			parts = [str(obj) for obj in line]
			line = "\t".join(parts)
			f.write(line + "\n")


def get_table(graphs):
	table = list()
	for key in graphs.keys():
		for G_name in tqdm(graphs[key]):
			G = load_graph(os.path.join(in_dir, key, G_name))
			table.append((key,
					G_name, 
					_get_num_vertices(G),
					_get_num_edges(G),
					_get_degree_info(G), 
					_get_ACC(G),
					_get_assortativity(G),
					_get_APL(G),
					_get_diameter(G)))

	_write_file(table, dst_dir)


def compile_selected_graphs(in_dir, names):
	"""Return a dictionary with selected graphs."""
	graphs = dict()
	for n in names:
		g = os.path.join(in_dir, n)
		graphs[n] = g
	return graphs


def get_pdfs(graphs):
	pdfs = dict()
	for key in graphs.keys():
		G = load_graph(graphs[key])
		pdf = nx.degree_histogram(G)
		pdfs[key] = pdf
	return pdfs
			
