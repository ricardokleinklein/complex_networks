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

import networkx as nx

if __name__ == '__main__':
	args = docopt(__doc__)
	dst_dir = args["<dst_dir>"]

	ER = new_ER(15, 10)

	draw(ER)


