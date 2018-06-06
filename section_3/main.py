"""
Main program of the section 3: community detection
in complex networks.

Usage:
	main.py [options] <in_dir> <dst_dir> 

options:
	-h, --help		Display help message
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from docopt import docopt

from utils import *

import community
import networkx as nx


if __name__ == "__main__":
	args = docopt(__doc__)
	in_dir = args["<in_dir>"]
	dst_dir = args["<dst_dir>"]

	graphs_dir = get_graphs(in_dir)
	for section in graphs_dir:
		graphs = [g for g in graphs_dir[section] if g.endswith('.net')]
		for g in graphs:
			# Load graph
			g_nx, g_ig = load_graph(os.path.join(in_dir, section, g))

			# Get partitions: Girvan-Newman algorithm
			# nx_gn, ig_gn = girvan_newman(g_nx, g_ig)
			# Get partitions: label propagation
			# nx_lp, ig_lp = label_propagation(g_nx, g_ig)
			# Get partitions: Louvain method
			nx_lo, ig_lo = louvain(g_nx, g_ig)
			
			# Estimate modularity
			# print(g, compute_modularity(g_nx, nx_lp, g_ig, ig_lp))

			g_ig, ig_lo = from_ig_to_nx(g_ig, ig_lo)
			draw_nx_graph(g_nx, g_ig, nx_lo, ig_lo)
