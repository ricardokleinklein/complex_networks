# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import collections
import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig
import zipfile
import itertools
from tqdm import tqdm

from networkx.algorithms import community
import sklearn.metrics as metrics


def _unzip(in_dir, name):
	dirs = _rm_hidden(os.listdir(in_dir))
	zip_ref = zipfile.ZipFile(os.path.join(in_dir, name), 'r')
	zip_ref.extractall(in_dir)
	zip_ref.close()
	

def _rm_hidden(files):
	return [file for file in files if not file.startswith(".")]


def get_paths(in_dir):
	"""Return a dictionary pointing to the graphs' path in `in_dir`."""
	name_dir = 'A3-networks.zip'
	# _unzip(in_dir, name_dir)
	dirs = _rm_hidden(os.listdir(in_dir))
	name_dir = name_dir.replace('.zip', '')
	assert name_dir in dirs

	graphs = dict()
	for d in _rm_hidden(os.listdir(os.path.join(in_dir, name_dir))):
		g = _rm_hidden(os.listdir(os.path.join(in_dir, name_dir, d)))
		graphs[os.path.join(in_dir, name_dir, d)] = g
	return graphs


def read_reference(file):
	if os.path.exists(file):
		with open(file, 'r') as f:
			reference = f.readlines()
		reference = [x.strip() for x in reference][1:]
		reference = [int(x) for x in reference]
		return reference
	return None


def load_graph(G):
	g = ig.Graph()
	g_nx = nx.convert_node_labels_to_integers(nx.read_pajek(G))
	return g_nx, g.Read_Pajek(G)


def get_graph_and_reference(paths):
	"""Given a dicitonary of relative paths, extract both the graph and the
	corresponding reference partition."""
	data = list()
	for section in paths:
		graphs = [g for g in paths[section] if g.endswith('.net')]
		name = [name.replace('.net','.clu') for name in graphs]
		for i in range(len(graphs)):
			g_nx, g_ig = load_graph(os.path.join(section, graphs[i]))
			reference = read_reference(os.path.join(section, name[i]))
			data.append((g_nx, g_ig, reference, name[i].replace('.clu', '')))
	return data

				
def label_propagation(G):
	return nx.algorithms.community.label_propagation.label_propagation_communities(G)


def girvan(G):
	return nx.algorithms.community.centrality.girvan_newman(G)


def multilevel(G):
	return G.community_multilevel()


def convert_nx_to_pred(nx_pred, G):
	N = nx.number_of_nodes(G)
	pred = [None] * N
	for i, component in enumerate(nx_pred):
		for node in component:
			pred[node] = i + 1
	return pred


def convert_ig_to_pred(ig_pred, G):
	N = G.vcount()
	pred = [None] * N
	for i, component in enumerate(ig_pred):
		for node in component:
			pred[node] = i + 1
	return pred


def jaccard(true, pred):
	return metrics.jaccard_similarity_score(true, pred)


def mutual_info(true, pred):
	return metrics.adjusted_mutual_info_score(true, pred)


def draw(G, labels, dst_dir, name):
	plt.subplot(1, 4, 1)
	nx.draw_kamada_kawai(G, node_color=labels[0], node_size=50)
	plt.title('Label-prop')
	plt.subplot(1, 4, 2)
	nx.draw_kamada_kawai(G, node_color=labels[1], node_size=50)
	plt.title('G-N')
	plt.subplot(1, 4, 3)
	nx.draw_kamada_kawai(G, node_color=labels[2], node_size=50)
	plt.title('MultiLevel')
	plt.subplot(1, 4, 4)
	nx.draw_kamada_kawai(G, node_color=labels[3], node_size=50)
	plt.title('Ref')
	plt.savefig(os.path.join(dst_dir, name))