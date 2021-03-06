# complex_networks repository

This is a repository with different assignments and problems that deal with complex networks and graphs' properties. Most of the problems, as well as the notation and references are extracted and/or derived from [M.E.J. Newman's *Networks: An Introduction*](https://global.oup.com/academic/product/networks-9780199206650?cc=jp&lang=en&).

## Requirements
- python >= 2.7
- numpy >= 1.14
- matplotlib >= 2.2.0
- networkx >= 2.1
- igraph >= 0.6.0
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

## Section 2

**Goal:** Generation of some models of complex networks.

The networks to generate are the following:
- Erdös-Rényi (ER), either based on distribution G(N,K) or G(N,p).
- Watts-Strogatz (WS) model.
- Barabási & Albert (BA) model.
- Configuration Model.

For each of the models above, run a series of experiments to study the outcome when changing the hyper parameters of the net, such as its order, size or the probability function that determines the final configuration of the net's edges.

For every case, compare the degree distribution (*pdf*) of the generated network with the theoretical distribution.

To run the experiments proposed in this section:
```
python section2/main.py /exp/section_2
```

No input data is required, for this task is to generate the graphs. However, the different results, as well as the plots comparing the experimental and the theoretical *pdf*, are saved in `/exp/section_2`.

## Section 3

**Goal:** Community detection in complex networks.

In this section we apply three different community detection algorithms to a series of complex networks available in `/data/A3-networks.zip`. The algorithms are implementations taken from the network-oriented libraries *iGraph* and *networkx*: 

1. Label propagation (*networkx*) - Based on label propagation
2. Girvan-Newman algorithm (*networkx*) - Based on the removal of edges with high betweenness centrality score
3. Community-multilevel (*iGraph*) - Based on modularity score

After communities have been found, a comparison with some reference partitions are performed tby means of the *Jaccard index* and the *mutual information* score. Last, communities (partitions) found are visually compared to the reference partitions.

To execute this part,
```
python section3/main.py /data/A3-networks.zip /exp/section_3
```

Should do the job and afterwards results should be saved in `/exp/section_3`.

## Section 4

**Goal:** Epidemics and disease spread in complex networks - SIS model.

Using a Montecarlo simulation, we have to study the dynamics of epidemics and disease transmission in complex networks when considering a SIS (Susceptible - Infected -Susceptible) model. In this scenario, nodes in the network are all originally healthy, but depending on whether any given node's neighbors are infected or not at some point, the node can fall sick as well and get infected, being able to spread the disease to other neighbors until it recovers (in case it do).

The magnitudes of interest are the variation induced in the dynamics by slight changes in size of the network, its topology, the infection probability and the recovery index.

To run this section's experiments, type in in the command line:
```
python section4/main.py /exp/section_4
```

No input data is required, generating the required graphs within the script. Results and plots should be available in `/exp/section_4`.

***

# License

The complex_networks is licensed under the MIT "Expat" License:

> Copyright (c) 2018: Ricardo Faundez-Carrasco.
>
> Permission is hereby granted, free of charge, to any person obtaining
> a copy of this software and associated documentation files (the
> "Software"), to deal in the Software without restriction, including
> without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject to
> the following conditions:
>
> The above copyright notice and this permission notice shall be
> included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
> EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
> MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
> IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
> CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
> TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
> SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


