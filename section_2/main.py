"""
Main program of the section 2: generation of complex
networks.

Usage:
	main.py [options] <dst_dir>

options:
	--nodes				Number of nodes
	-h, --help 		Display help message
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from docopt import docopt
from utils import *


if __name__ == '__main__':
	args = docopt(__doc__)
	dst_dir = args["<dst_dir>"]

	N = 1500
	K, p = N * (N-1) / 2, 0.01
	m = 10

	ER = new_ER(N, p=p)
	BA = new_BA(N, m)