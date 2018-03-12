from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import networkx as nx

def _load_graph(path):
	return nx.read_pajek(path)

def draw_graph(G, name):
	nx.draw_networkx(G, with_labels=True, font_weight='bold',
		label=name)