"""
Main program of the section 1: structural descriptors
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


if __name__ == '__main__':
	args = docopt(__doc__)
	in_dir = args["<in_dir>"]
	dst_dir = args["<dst_dir>"]

	graphs = compile_graph_libs(in_dir)

	names = ['model/ER1000k8.net',
		'model/SF_1000_g2.7.net',
		'model/ws1000.net',
		'real/airports_UW.net']

	graphs = compile_selected_graphs(in_dir, names)

	PDF = get_pdfs(graphs)

