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


if __name__ == "__main__":
	args = docopt(__doc__)
	in_dir = args["<in_dir>"]
	dst_dir = args["<dst_dir>"]

	graphs_dir = get_paths(in_dir)
	data = get_graph_and_reference(graphs_dir)
	
	for i in range(len(data)):
		g_nx, g_ig, reference, name = data[i]
		label = convert_nx_to_pred(label_propagation(g_nx), g_nx)
		gir = girvan(g_nx)
		if reference is not None:
			pred = convert_nx_to_pred([i for i in gir][max(reference)-1], g_nx)
		else:
			pred = convert_nx_to_pred([i for i in gir][-1], g_nx)
		level = convert_ig_to_pred(multilevel(g_ig), g_ig)
		
		print(name)
		if reference is not None:
			print(jaccard(reference, label),
				jaccard(reference, pred),
				jaccard(reference, level))

			print(mutual_info(reference, label),
				mutual_info(reference, pred),
				mutual_info(reference, level))

		draw(g_nx, (label, pred, level, reference), dst_dir, name)