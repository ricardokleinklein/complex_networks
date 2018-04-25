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


def _increase_kmin(pdf, k_min):
	k = np.min(pdf)
	while k < k_min:
		pdf += 1
		k = np.min(pdf)


def _generate_pdf(N, k_min, gamma, is_powerLaw=True):
	if is_powerLaw:
		k_max = np.floor(np.sqrt(N))
		pdf = np.ceil(k_max * np.random.power(gamma + 1, size=N))
	else:
		pdf = np.floor(np.random.poisson(gamma - 1, size=N))
	_increase_kmin(pdf, k_min)
	pdf[-1] = pdf[-1] + 1 if np.sum(pdf) % 2 != 0 else pdf[-1]
	return pdf.astype(int)


def _to_stub_vector(pdf):
	from itertools import chain
	chaini = chain.from_iterable
	return list(chaini([n] * d for n, d in enumerate(pdf)))


def _rm_isolated_nodes(G):
	node_rm = [node for node in nx.nodes(G) if not nx.edges(G, node)]
	G.remove_nodes_from(node_rm)


def new_CM(N, k_min=1, gamma=2, is_powerLaw=True):
	"""Configuration Model."""
	G = _new_empty_G(N)
	pdf = _generate_pdf(N, k_min, gamma, is_powerLaw=is_powerLaw)
	stub = _to_stub_vector(pdf)
	np.random.shuffle(stub)
	n = len(stub) // 2
	for i in range(n):
		stub_a = stub[2*i]
		stub_b = stub[2*i+1]
		is_diff = stub_a != stub_b
		is_connected = stub_b in nx.all_neighbors(G, stub_a)
		if is_diff and not is_connected:
			G.add_edge(stub_a, stub_b)
	_rm_isolated_nodes(G)
	return G


def get_pdf(G):
	return [v for _, v in nx.degree(G)]


def plt_pdf(G):
	pdf = [v for _, v in nx.degree(G)]
	pdf = np.sort(pdf)
	plt.hist(pdf, normed=True, bins=10)
	plt.xlabel('Degree k')
	plt.ylabel('Fraction of p_k of vertices with degree k')
	plt.show()

	
def estimate_gamma(pdf):
	sum_term = 0
	n = len(pdf)
	k_min = np.min(pdf)
	for i in range(len(pdf)):
		sum_term += np.log(pdf[i] / (k_min - 0.5))
	sum_term = 1 / sum_term
	return 1 + n * sum_term


def draw(G, name, dst_dir):
	print(nx.info(G))
	nx.draw_networkx(G, node_color='c', alpha=0.85)
	plt.title(name.replace(".png",""))
	plt.axis('off')
	plt.savefig(os.path.join(dst_dir,name))