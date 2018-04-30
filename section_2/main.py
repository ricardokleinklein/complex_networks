"""
Main program of the section 2: generation of complex
networks.

Usage:
	main.py [options] <dst_dir>

options:
	-h, --help 		Display help message
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from docopt import docopt
from utils import *


def exp_ER(N, dst_dir):
	max_K = N * (N-1) // 2
	K = [max_K // 4, max_K // 2]
	P = [0.05, 0.25]
	for k in K:
		G = new_ER(N, K=k)
		name = 'ER(N,K) with N = %i and K = %i.png' % (N, k)
		draw(G, name, dst_dir)
	for p in P:
		G = new_ER(N, p=p)
		p = str(p)
		name = 'ER(N, p) with N = %i and p = %s.png' % (N, p)
		draw(G, name, dst_dir)

def exp_WS(N, dst_dir):
	K = [2, 4, 6]
	# P = [0.0, 0.5, 1.0]
	p = 0.5
	for k in K:
		G = new_WS(N, k, p)
		name = 'WS(N,p) with N = %i, K = %i and p = %s.png' % (N, k, str(p))
		draw(G, name, dst_dir)

def exp_BA(N, dst_dir):
	M = [1, 2, 10]
	for m in M:
		G = new_BA(N, m)
		name = 'BA(N,m) with N = %i and m = %i.png' % (N, m)
		draw(G, name, dst_dir)


def exp_CM(N, dst_dir):
	# Power Law
	is_powerLaw = True
	# gamma = [2.0, 2.3, 2.6, 3.0]
	gamma = [2.5]
	for g in gamma:
		G = new_CM(N, gamma=g, is_powerLaw=is_powerLaw)
		name = 'CM(N, gamma) with N = %i, gamma = %s and power-law.png' % (N, str(g))
		draw(G, name, dst_dir)

	# Poisson
	is_powerLaw = False
	# gamma = [2,4,16,32]
	gamma = [6]
	for g in gamma:
		G = new_CM(N, gamma=g, is_powerLaw=is_powerLaw)
		name = 'CM(N, gamma) with N = %i, gamma = %s and poisson.png' % (N, str(g))
		draw(G, name, dst_dir)


if __name__ == '__main__':
	args = docopt(__doc__)
	dst_dir = args["<dst_dir>"]

	# # Experiments on ER
	# N = [50, 1000, 10000]
	# [exp_ER(n, dst_dir) for n in N]

	# Experiments on WS
	# N = [50, 1000, 10000]
	# N = [100]
	# [exp_WS(n, dst_dir) for n in N]

	# Experiments on BA
	# N = [100]
	# [exp_BA(n, dst_dir) for n in N]

	# Experiments on CM
	N = [100, 1000, 10000]
	# N = [100]
	[exp_CM(n, dst_dir) for n in N]
	