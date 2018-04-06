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

def _new_empty_G(N):
	return nx.empty_graph(N)


def _new_null_G():
	return nx.null_graph()


def _new_ER(G, N, p):
	"""Erdos-Renyi probabilistic network."""
	raise NotImplementedError


def new_ER(N, K, p=None):
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


def draw(G):
	print(nx.info(G))
	nx.draw_networkx(G)
	plt.show()