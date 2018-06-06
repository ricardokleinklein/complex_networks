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
import community as CM


def _unzip(in_dir, name):
	dirs = _rm_hidden(os.listdir(in_dir))
	zip_ref = zipfile.ZipFile(os.path.join(in_dir, name), 'r')
	zip_ref.extractall(in_dir)
	zip_ref.close()
	

def _rm_hidden(files):
	return [file for file in files if not file.startswith(".")]


def get_graphs(in_dir):
	"""Return a dictionary pointing to the graphs' path in `in_dir`."""
	name_dir = 'A3-networks.zip'
	_unzip(in_dir, name_dir)
	dirs = _rm_hidden(os.listdir(in_dir))
	name_dir = name_dir.replace('.zip', '')
	assert name_dir in dirs

	graphs = dict()
	for d in _rm_hidden(os.listdir(os.path.join(in_dir, name_dir))):
		g = _rm_hidden(os.listdir(os.path.join(in_dir, name_dir, d)))
		graphs[os.path.join(name_dir, d)] = g
	return graphs


def load_graph(G):
	g = ig.Graph()
	return nx.read_pajek(G), g.Read_Pajek(G)


def girvan_newman(G_nx, G_ig, clusters=None):
	comp = community.centrality.girvan_newman(G_nx)
	if clusters == None:
		c_ig = G_ig.community_edge_betweenness()
		clusters = c_ig.optimal_count
	for communities in itertools.islice(comp, clusters):
		comm = tuple(sorted(c) for c in communities)
	c_nx = comm
	# Convert to dictionary
	prov = dict()
	for step, comm in enumerate(c_nx):
		for node in comm:
			prov[node] = step
	c_nx = prov
	return c_nx, c_ig
		

def label_propagation(G_nx, G_ig):
	c_nx = community.label_propagation.asyn_lpa_communities(G_nx)
	c_ig = G_ig.community_label_propagation()
	# Convert to dictionary
	prov = dict()
	for step, comm in enumerate(c_nx):
		for node in comm:
			prov[node] = step
	c_nx = prov
	return c_nx, c_ig


def louvain(G_nx, G_ig):
	c_nx = CM.best_partition(G_nx, randomize=True)
	c_ig = G_ig.community_multilevel()
	return c_nx, c_ig


def compute_modularity(G_nx, part_nx, G_ig, part_ig):
	return CM.modularity(part_nx, G_nx), G_ig.modularity(part_ig)


def _part_to_nx(part):
	new_part = dict()
	num_clu = len(part)
	for c in range(num_clu):
		for i in part[c]:
			new_part[unicode('*' + str(i) + '*')] = c
	return new_part


def from_ig_to_nx(g, part):
	g = g.get_edgelist()
	g = nx.MultiGraph(g)
	part = _part_to_nx(part)
	return g, part


def community_layout(g, partition):
    """
    Compute the layout for a modular graph.


    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

    pos_communities = _position_communities(g, partition, scale=3.)

    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos

def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos


def draw_nx_graph(g1, g2, part1, part2):
	pos = nx.kamada_kawai_layout(g1)
	plt.title(g1)
	plt.subplot(1,2,1)
	nx.draw(g1, pos, node_color=list(part1.values()))
	plt.subplot(1,2,2)
	pos = nx.kamada_kawai_layout(g2)
	nx.draw(g2, pos, node_color=list(part2.values()))
	plt.show()


def save_graph(name, G_nx, G_ig, dst_dir):
	try:
		os.mkdir(dst_dir)
	except:
		pass
	G_ig.graph.write_pajek(os.path.join(dst_dir, name + '.net'))


