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
	return G


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


def normal_pdf(mu, var):
	sigma = np.sqrt(var)
	x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
	plt.plot(x,mlab.normpdf(x, mu, sigma))


def get_pdf(G):
	pdf = [v for _, v in nx.degree(G)]
	pdf = np.sort(pdf)
	plt.hist(pdf, normed=True, bins=20)
	plt.title(G)
	plt.xlabel('Degree k')
	plt.ylabel('Fraction p_k of vertices with degree k')


def plot():
	plt.show()


def draw(G):
	print(nx.info(G))
	nx.draw_networkx(G, node_color='c', alpha=0.85)
	plt.show()