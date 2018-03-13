# complex_networks repository

This is a repository with different assignments and problems that deal with complex networks and graphs' properties. Most of the problems, as well as the notation and references are extracted and/or derived from [M.E.J. Newman's *Networks: An Introduction*](https://global.oup.com/academic/product/networks-9780199206650?cc=jp&lang=en&).

## Requirements
- python >= 2.7
- numpy >= 1.14
- matplotlib >= 2.2.0
- networkx >= 2.1
- docopt >= 0.6.0
- tqdm >= 4.19.0

## Data

In `~data/A1-networks.zip` there are several complex networks in *Pajek* (*.net*) format, splitted in three categories:
1. `toy`- Example nets.
2. `model` - Generated from network model's samples.
3. `real` - Actual networks.

You can import any graph available in those directories by calling:
```
import networkx as nx
G = nx.read_pajek('path_to_the_graph/graph.net')
```

Print any of them to have a preliminar view:
```
import matplotlib.pyplot as plt
nx.draw(G, with_labels=True, font_weights='bold')
plt.show()
```

## Section 1

**Goal:** Structural descriptors in complex networks.

The descriptors that are to compute are:
- Number of vertices
- Number of edges
- Minimum, mean and maximum degree of each network
- Average Clustering Coefficient
- Assortativity (preference for a network's nodes to attach to others that are similar in some way)
- Average Path Length
- Diameter (greatest length between two vertices in a network)

Furthermore, plot the figures representing the probability distribution function (*pdf*) and complementary cumulative distribution function (*ccdf*) for some networks.

To run the experiments proposed in this section:
```
python section1/main.py data/A1-networks/ /exp/section_1/table.txt
```
The graphs must be present in `data/A1-networks`, as it'll be generated when extracting the files included in this repository. When the processing is done, the results can be found as a table in `/exp/section_1/table.txt`. You can freely change where the folder containing the graphs `<in_dir>` and the directory in which save the results`<dst_dir>` point to.



