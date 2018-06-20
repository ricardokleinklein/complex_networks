# encoding: utf-8
"""
Main program for the study of disease spreading in a
complex network. Different complex networks are
implemented by default and the experiments are 
performed in all of them.

Usage: 
	main.py <dst_dir>

Options:
	-h, --help:		Display this message
"""

import os

from utils import *
from docopt import docopt

if __name__ == '__main__':
	args = docopt(__doc__)
	dst_dir = args['<dst_dir>']

	N = 500
	mu = [0.1, 0.5, 0.9]
	beta = np.linspace(1e-4, 1, 51)
	p0 = 0.2
	Nrep = 20
	Tmax = 1000
	Ttrans = int(0.9 * Tmax)
	p = 0.05
	m = 6

	Graphs = [
		# PathGraph(N),
		# StarGraph(N),
		# WheelGraph(N),
		# HypercubeGraph(N),
		# ErdosRenyiGraph(N, p),
		PowerLawGraph(N, m, p),
		# BarabasiAlbertGraph(N, m),
		# DirectedScaleFreeGraph(N),
		# KarateGraph()
		]

	for G in Graphs:
		G.draw(dst_dir)
		G.set_params(mu=None, beta=None, Nrep=Nrep, 
			p0=p0, Tmax=Tmax, Ttrans=Ttrans)

		EXP = Experiment(len(beta), G)
		EXP.run_mu_comparison(mu, beta)
		EXP.plot_beta_vs_mu(dst_dir, beta)
