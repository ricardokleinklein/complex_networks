# complex_networks repository

This is a repository with different assignments and problems that deal with complex networks and graphs' properties. Most of the problems, as well as the notation and references are extracted and/or derived from [M.E.J. Newman's *Networks: An Introduction*](https://global.oup.com/academic/product/networks-9780199206650?cc=jp&lang=en&).

## Requirements
- numpy >= 1.14
- matplotlib
- networkx >= 2.1

## Section 1

**Goal:** Structural descriptors in complex networks.

In `data/A1-networks.zip` there are several complex networks in *Pajek* (*.net*) format, splitted in three categories:
1. `toy`- Example nets.
2. `model` - Generated from network model's samples.
3. `real` - Actual networks.

The descriptors that are to compute are:
- Number of vertices
- Number of edges
- Minimum, mean and maximum degree of each network
- Average Clustering Coefficient
- Assortativity (preference for a network's nodes to attach to others that are similar in some way)
- Average Path Length
- Diameter (greatest length between two vertices in a network)

Furthermore, plot the figures representing the probability distribution function (*pdf*) and complementary cumulative distribution function (*ccdf*) for some networks.

