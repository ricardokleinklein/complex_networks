"""
Main program of the section 1: structural descriptors
in complex networks.

Usage:
	main.py <in_dir> <dst_dir>

options:
	-h, --help		Display help message
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from docopt import docopt
from tqdm import tqdm

from utils import *


if __name__ == '__main__':
	args = docopt(__doc__)
	in_dir = args["<in_dir>"]
	dst_dir = args["<dst_dir>"]

	graphs = compile_graph_libs(in_dir)
	table = list()

	for key in graphs.keys():
		for G_name in tqdm(graphs[key]):
			G = load_graph(os.path.join(in_dir, key, G_name))
			table.append((key,
				G_name, 
				get_num_vertices(G),
				get_num_edges(G),
				get_degree_info(G), 
				get_ACC(G),
				get_assortativity(G),
				get_APL(G),
				get_diameter(G)))
	
	write_file(table, dst_dir)