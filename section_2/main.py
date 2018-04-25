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


if __name__ == '__main__':
	args = docopt(__doc__)
	dst_dir = args["<dst_dir>"]

	N = 1000
	k, K, p = 8, int(N * (N-1) / 2), 0.2
	m = 1

	# ER = new_ER(N, p=p)
	# WS = new_WS(N, k, p)
	# BA = new_BA(N, m)
	CM = new_CM(N, is_powerLaw=False)	
	# draw(CM)