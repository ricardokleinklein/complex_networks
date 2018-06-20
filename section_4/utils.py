# encoding: utf-8
from __future__ import print_function
from __future__ import division

import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm

np.random.seed(1)

class BaseGraph:
	"""Baseline Graph defined as a base for further Graph classes."""
	def __init__(self):
		self.name = None
		self.N = None
		self.G = None

		self.params = dict()
		self.state = None

	def set_params(self, mu=None, beta=None, 
		Nrep=50, p0=0.2, Tmax=1000, Ttrans=900):
		"""Set the montecarlo parameters of the model to execute later.
		
		Args:
			mu (array<float>): spontaneous recovery probability.
			beta (array<float>): infection probability of a susceptible individual.
			Nrep (int): number of repetitions of the simulation.
			p0 (float): initial fraction of infected nodes.
			Tmax (int): maximum number of time steps of each simulation.
			Ttrans (int): number of steps of the transitory.
		"""
		for key, value in vars().items():
			if key != 'self':
				if (key == 'mu' or key == 'beta') and np.isscalar(value):
					if key == 'mu':
						np.random.poisson(lam=value, size=(self.N))
					else:
						self.params[key] = [value] * self.N
				else:
					self.params[key] = value

	def _is_recovering(self, idx):
		prob = np.random.random()
		if self.params['mu'][idx] > prob:
			return False
		else:
			return True

	def _gets_infected(self, idx):
		nbr = self.G[idx]
		infected_nbr = [i for i in nbr if self.state[i] == True]
		for infected in infected_nbr:
			prob = np.random.random()
			if self.params['beta'][idx] > prob and not self.state[idx]:
				return True

		return False

	def _update_state(self, recovering, new_infected):
		for idx, status in recovering:
			self.state[idx] = status
		for idx, status in new_infected:
			self.state[idx] = status

	def draw(self, dst_dir, display=False):
		nx.draw(self.G)
		if display:
			plt.show()
		else:
			plt.savefig(os.path.join(dst_dir, self.name))
			plt.clf()


class CompleteGraph(BaseGraph):
	"""Complete Graph K_N."""
	def __init__(self, N):
		self.name = 'Complete graph K_%i' % N
		self.N = N
		self.G = nx.complete_graph(N)

		self.params = dict()
		self.state = None


class PathGraph(BaseGraph):
	"""Path Graph L_N."""
	def __init__(self, N):
		self.name = 'Path graph L_%i' % N
		self.N = N
		self.G = nx.path_graph(N)

		self.params = dict()
		self.state = None


class StarGraph(BaseGraph):
	"""Star Graph S_N."""
	def __init__(self, N):
		self.name = 'Star graph S_%i' % N
		self.N = N
		self.G = nx.star_graph(N-1)

		self.params = dict()
		self.state = None


class WheelGraph(BaseGraph):
	"""Wheel Graph W_N."""
	def __init__(self, N):
		self.name = 'Wheel graph W_%i' % N
		self.N = N
		self.G = nx.wheel_graph(N-1)

		self.params = dict()
		self.state = None


class HypercubeGraph(BaseGraph):
	"""Wheel Graph HyperCube_N."""
	def __init__(self, N):
		self.name = 'Hypercube graph HyperCube_%i' % N
		self.N = N
		self.G = nx.hypercube_graph(N)

		self.params = dict()
		self.state = None


class ErdosRenyiGraph(BaseGraph):
	"""Erdos-Renyi random Graph."""
	def __init__(self, N, p):
		self.name = 'Erdos-Renyi random graph ER_%i' % N
		self.N = N
		self.G = nx.fast_gnp_random_graph(N, p)

		self.params = dict()
		self.state = None


class PowerLawGraph(BaseGraph):
	def __init__(self, N, m, p):
		self.name = 'Default Power-Law random graph SF_%i (Poisson distribution)' % N
		self.N = N
		self.G = nx.powerlaw_cluster_graph(N, m, p)

		self.params = dict()
		self.state = None


class BarabasiAlbertGraph(BaseGraph):
	def __init__(self, N, m):
		self.name = 'Barabasi-Albert random graph BA_%i' % N
		self.N = N
		self.G = nx.barabasi_albert_graph(N, m)

		self.params = dict()
		self.state = None


class DirectedScaleFreeGraph(BaseGraph):
	def __init__(self, N):
		self.name = 'Default directed Scale-Free random graph DSF_%i' % N
		self.N = N
		self.G = nx.scale_free_graph(N)

		self.params = dict()
		self.state = None


class KarateGraph(BaseGraph):
	"""Karate social club graph."""
	def __init__(self):
		self.name = "Zacharys Karate club"
		self.G = nx.karate_club_graph()
		self.N = nx.number_of_nodes(self.G)

		self.params = dict()
		self.state = None


class Experiment:
	"""Performs M MonteCarlo simulations on graph varying the hyperparameters."""
	def __init__(self, M, graph):
		self.M = M
		self.G = graph
		self.rho = dict()

	def __str__(self):
		s = 'Hyperparameters = {\n'
		s += '\tNet = ' + self.G.name + '\n'
		s += '\tN = ' + self.G.N + '\n'
		for key, value in self.G.params.items():
			if key == 'mu' or (key == 'beta' and value != None):
				s += '\t' + str(key) + ' = ' + str(np.mean(value)) + '\n'
			else:
				s += '\t' + str(key) + ' = ' + str(value) + '\n'
		s += '}'
		return s

	def prepare_initial(self):
		self.G.state = np.array([False] * self.G.N)
		infected_size = int(self.G.params['p0'] * self.G.N)
		infected_idx = np.random.choice(self.G.N, infected_size, replace=False)
		self.G.state[infected_idx] = True

	def _single_montecarlo(self):
		single_rho = [0] * self.G.params['Tmax']
	
		for it in range(self.G.params['Tmax']):
			recovering = list()
			new_infected = list()

			single_rho[it] = len([i for i in self.G.G if self.G.state[i] == True]) / self.G.N

			for node in self.G.G:
				if self.G.state[node] == True:
					recovering.append((node, self.G._is_recovering(node)))
				else:
					new_infected.append((node, self.G._gets_infected(node)))

			self.G._update_state(recovering, new_infected)

		return single_rho

	def run_montecarlo(self):
		avg_rho = np.zeros((self.G.params['Nrep'], self.G.params['Tmax']), 
			dtype=float)

		for EXP in range(self.G.params['Nrep']):
			avg_rho[EXP, :] = self._single_montecarlo()

		return np.mean(avg_rho[:, self.G.params['Ttrans']:])

	def execute(self, mu, beta):
		num_beta_values = len(beta)
		rho = [0] * num_beta_values
		total_exp = num_beta_values * self.G.params['Nrep']
		current_exp = 0

		print('Preparing execution of %i (EXP/beta) x %i (beta) = '
			'%i experiments with mu = %.2f for graph %s' % 
			(self.G.params['Nrep'], num_beta_values, 
				total_exp, np.mean(mu), self.G.name))

		for k in tqdm(range(num_beta_values)):
			self.G.params['mu'] = [mu] * self.G.N if np.isscalar(mu) else mu
			self.G.params['beta'] = [beta[k]] * self.G.N
			self.prepare_initial()

			rho[k] = self.run_montecarlo()
			current_exp += self.G.params['Nrep']
			tqdm.write('(%i/%i) Completed with beta = %f, average rho = %f' % 
				(current_exp, total_exp, beta[k], rho[k]))

		mu = np.mean(self.G.params['mu'])
		self.rho[str(mu)] = rho
		print('Finished.')
	
	def run_mu_comparison(self, mu, beta):
		for val in mu:
			self.execute(val, beta)

	def plot_time_vs_rho(self, it=None, display=False):
		if it is not None:
			plt.plot(range(self.G.params['Tmax']), self.avg_rho[it-1, :])
		else:
			raise IOError('No iteration specified.'
				' Please specify one experiment to plot.')
		if display:
			plt.show()

	def plot_single_beta_vs_rho(self, mu, beta, display=False):
		if self.rho:
			plt.plot(beta, self.rho[mu], label='mu = %.1f' % float(mu) )
		if display:
			plt.show()

	def plot_beta_vs_mu(self, dst_dir, beta, display=False):
		for mu in self.rho:
			self.plot_single_beta_vs_rho(mu, beta)
			plt.title(self.G.name, fontsize=16)
			plt.xlabel('beta', fontsize=14)
			plt.ylabel('rho', fontsize=14)
			plt.legend(bbox_to_anchor=(.02, 0.98), loc=2, borderaxespad=0., 
				prop={'size': 9})
			plt.grid()
			plt.axis([0.0, 1.0, 0.0, 1.0])
		if display:
			plt.show()
		else:
			plt.savefig(os.path.join(dst_dir, self.G.name + 'result'))
			plt.clf()


