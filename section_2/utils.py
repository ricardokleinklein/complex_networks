from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import collections
import random
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import matplotlib.mlab as mlab

def _new_empty_G(N):
	return nx.empty_graph(N)


def _new_null_G():
	return nx.null_graph()


def _new_ER(G, N, p):
	"""Erdos-Renyi probabilistic network."""
	non_edges = nx.non_edges(G)
	max_n_edges = int(N * (N - 1) / 2)
	edges_p = np.random.random(max_n_edges)
	edges_idx, = np.where(edges_p >= 1 - p)
	for idx, e in enumerate(non_edges):
		if idx in edges_idx:
			G.add_edge(e[0], e[1])
	

def new_ER(N, K=None, p=None):
	"""Erdos-Renyi network."""
	G = _new_empty_G(N)
	if p is None and K is not None:
		non_edges = nx.non_edges(G)
		max_n_edges = int(N * (N - 1) / 2)
		edges_idx = random.sample(range(max_n_edges), K)
		for idx, e in enumerate(non_edges):
			if idx in edges_idx:
				G.add_edge(e[0], e[1])
	elif p is not None:
		_new_ER(G, N, p)
	else:
		raise ValueError('Configuration of edges not specified.')
	return G


def _make_ring_lattice(G, K):
	d = nx.number_of_nodes(G) - 1 - K/2
	nodes = nx.nodes(G)
	for i in nodes:
		for j in nodes:
			op = abs(i - j) % d
			if op > 0 and op <= K/2:
				G.add_edge(i,j)


def _update_G(G, change):
	G.remove_edge(change[0], change[1])
	G.add_edge(change[0], change[2])


def new_WS(N, K, p):
	"""Watts-Strogatz network."""
	# It must satisfy that N >> K >> log(N) >> 1
	G = _new_empty_G(N)
	_make_ring_lattice(G, K)
	nodes = nx.nodes(G)
	for i in nodes:
		neighbors = nx.all_neighbors(G, i)
		forbidden = list()
		forbidden.append(i)
		[forbidden.append(j) for j in neighbors]
		for j in nx.all_neighbors(G,i):
			if i < j:
				prob = np.random.random()
				if prob > 1 - p:
					new_j = np.random.randint(0, N)
					while new_j in forbidden:
						new_j = np.random.randint(0, N)
					_update_G(G, (i, j, new_j))
	return G


def _make_clique(G):
	non_edges = nx.non_edges(G)
	for e in non_edges:
		G.add_edge(e[0], e[1])


def _add_node(G, m):
	pdf = get_pdf(G)
	new_node = len(pdf)
	G.add_node(new_node)
	K = 2 * nx.number_of_edges(G)
	probs_per_node = [np.sum(pdf[:i+1]) for i in range(len(pdf))]
	unvalid = list()
	while len(unvalid) < m:
		prob = np.random.random() * K
		idx = min([probs_per_node.index(s) for s in probs_per_node if s > prob])
		if idx not in unvalid:
			unvalid.append(idx)
		G.add_edge(new_node, idx)


def new_BA(N, m):
	"""Barabasi and Albert network."""
	N_init = 15
	G = _new_empty_G(N_init)
	_make_clique(G)
	while nx.number_of_nodes(G) < N:
		_add_node(G, m)
	return G


def _generate_random_pdf(N):
	pdf = np.random.randint(0, N - 1, size= N)
	while np.sum(pdf) % 2 != 0:
		pdf = np.random.randint(0, N - 1, size=N)
	return pdf


def new_CM(N, pdf=None):
	G = _new_empty_G(N)
	max_n_edges = N - 1
	if pdf is None:
		pdf = _generate_random_pdf(N)
	K = np.sum(pdf)
	stubs_node = [np.sum(pdf[:i+1]) for i in range(len(pdf))]
	print(K, pdf, stubs_node)
	forbidden = list()
	while len(forbidden) != K:
		stub_i = np.random.randint(0, K)
		while stub_i in forbidden:
			stub_i = np.random.randint(0, K)
		i = min([stubs_node.index(s) for s in stubs_node if s > stub_i])
		forbidden.append(stub_i)

		stub_j = np.random.randint(0, K)
		while stub_j in forbidden:
			stub_j = np.random.randint(0, K)
		j = min([stubs_node.index(s) for s in stubs_node if s > stub_j])
		forbidden.append(stub_j)
		G.add_edge(i, j)

	return G


def normal_pdf(mu, var):
	sigma = np.sqrt(var)
	x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
	plt.plot(x,mlab.normpdf(x, mu, sigma))


def get_pdf(G):
	return [v for _, v in nx.degree(G)]


def plt_pdf(G):
	pdf = [v for _, v in nx.degree(G)]
	pdf = np.sort(pdf)
	plt.hist(pdf, normed=True, bins=50)
	plt.title(G)
	plt.xlabel('Degree k')
	plt.ylabel('Fraction p_k of vertices with degree k')
	plt.show()


def draw(G):
	print(nx.info(G))
	nx.draw_networkx(G, node_color='c', alpha=0.85)
	plt.show()