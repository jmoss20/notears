# notears
**Python package implementing "DAGs with NO TEARS: Smooth Optimization for Structure Learning", Xun Zheng, Bryon Aragam, Pradeem Ravikumar and Eric P. Xing (March 2018, [arXiv:1803.01422](http://https://arxiv.org/pdf/1803.01422.pdf))**

This package implements the NOTEARS learning algorithm, and supplies a few useful utilities (e.g. for generating random graphs, simulating data from linear Gaussian models, measuring performance, and thresholding edge matrices returned by NOTEARS to ensure acyclicity).

Optimization is ultimately performed by the SciPy implementation of L-BFGS-B.

## Dependencies
```
numpy
scipy
networkx
```

## Usage
See `example_usage.ipynb` for a simple Jupyter notebook demonstrating usage.

In general, using this package looks like:
```python 
import notears

output_dict = notears.run(notears.notears_standard, data, notears.loss.least_squares_loss, notears.loss.least_squares_loss_grad)
thresholded_output = notears.utils.threshold_output(output_dict['W'])
```

## Parameters
### notears.run
```python
notears.run(variant, data, loss, loss_grad, c=0.25, r=10.0, e=1e-8, rnd_W_init=False, output_all_progress=False, verbose=False)
```
* `variant`: Which implementation of NOTEARS to use (at this time, only `notears.notears_standard` is implemented)
* `data`: An `n x d` numpy array, containing `n` samples (rows) of data from `d` variables (columns)
* `loss`: Function to use as loss function, can either pass `notears.loss.least_squares_loss` for least squares loss, `notears.loss.least_squares_loss_cov` for expectation of least squares loss (recommended when using a large number of samples), or define a custom loss function (see `notears/loss.py` for reference implementations).
* `loss_grad`: Function returning the gradient of the function specified as `loss`, with respect to the adjacency matrix `W`.  (e.g. `notears.loss.least_squares_loss_grad` or `notears.loss.least_squares_loss_cov_grad`)
* `c`: minimum rate of progress `c \in (0,1)` (see paper)
* `r`: penalty growth rate `r > 1` (see paper)
* `e`: acyclicity loss stopping criteria `\epsilon > 0` (see paper)
* `rnd_W_init`: boolean, denoting whether or not to initialize $W$ to standard normal random matrix (default is a zero matrix)
* `output_all_progress`: whether to return only the final value of the optimization process, or all intermediate values

Calling this function returns a dictionary of the form `{'h': h(W), 'loss': loss(W, data), 'W': W}`, unless `output_all_progress` is true, in which case it returns an array of such dictionaries.

## Utilities
Some useful utilities are provided in `notears/utils.py`, and can be accessed from `notears.utils`.
* `threshold_output(W, desired_edges=None, verbose=False)`: takes in a (possibly cyclic) adjacency matrix, returns an acyclic adjacency matrix either by removing as few edges (by weight) as possible, or by finding the acyclic graph with total number of edges closest to `desired_edges`.  
* `generate_random_dag(num_nodes, num_edges, probabilistic=False, edge_coefficient_range=[0.5, 2.0])`: returns tuple `(adj_ mat, g)` containing a weighted random DAG represented as numpy adjacency matrix and `networkx` DiGraph, respectively.  The `probabilistic` flag determines whether the graph will contain strictly `num_edges` number of edges, or some random number of edges with expectation `num_edges`.
* `simulate_from_dag_lg(adj_mat, n_sample, mean=0, variance=1)`: simulates `n_samples` samples from the linear Gaussian model specified by weighted DAG adjacency matrix `adj_mat`, with error terms drawn from `N(mean, variance)`.
* `compare_graphs_undirected(true_graph, estimated_graph)`: takes two adjacency matrices, and returns list `[true_positives, false_positives, true_negatives, false_negatives]` in terms of **adjacencies** (i.e. undirected edges).
* `compare_graphs_precision`, `compare_graphs_recall`, `compare_graphs_specificity`: take in the output of `compare_graphs_undirected`, returns either precision, recall, or specificity, again in terms of **adjacencies**.



