from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import collections
import matplotlib.pyplot as plt
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
	return _get_num_edges(G) / _get_num_vertices(G)


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
	plt.show()


def write_file(table, dst_dir):
	with open(dst_dir, 'w') as f:
		f.write('dir\t' + 'graph\t' + 'order\t' + 'size\t' + 'degrees\t' +
			'ACC\t' + 'assortativity\t' + 'avg_path_length\t' + 
			'diameter' + '\n')
		for line in table:
			parts = [str(obj) for obj in line]
			line = "\t".join(parts)
			f.write(line + "\n")


def get_table(in_dir, graphs):
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
	return table


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
		pdf = [v for _, v in nx.degree(G)]
		pdfs[key] = np.sort(pdf)
	return pdfs


def plot_simple_pdf(pdfs):
	for G in pdfs.keys():
		plt.hist(pdfs[G], normed=True)
		plt.title(G)
		plt.xlabel('Degree k')
		plt.ylabel('Fraction p_k of vertices with degree k')
		plt.show()


def plot_loglog_pdf(pdfs):
	for G in pdfs.keys():
		k_min = np.min(pdfs[G])
		k_max = np.max(pdfs[G])
		p_max = np.log(k_max + 1)
		p = np.log(pdfs[G])
		n_bins = 10
		bins = np.linspace(p[0], p_max, n_bins)
		for i in range(len(bins)):
			for j in range(len(p)):
				if p[j] > bins[i] and p[j] < bins[i+1]:
					p[j] = bins[i]
				if p[j] > bins[-1]:
					p[j] = bins[-1]
		plt.hist(p, normed=True, log=True)
		plt.title(G)
		plt.xlabel('Degree k')
		plt.ylabel('Fraction p_k of vertices with degree k')
		plt.show()

		
		
		

			
