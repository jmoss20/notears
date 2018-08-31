import unittest

import sys
import os
import time

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import notears

def simple_test(num_nodes, num_edges, verbose=False, cov=False):
	tam, tg = notears.utils.generate_random_dag(num_nodes, num_edges, probabilistic=False, edge_coefficient_range=[0.5, 2.0])
	data = notears.utils.simulate_from_dag_lg(tam, 1000)

	if not cov:
		nt_weight = notears.run(notears.notears_standard, data, notears.loss.least_squares_loss, notears.loss.least_squares_loss_grad, e=1e-8, verbose=verbose)['W']
	else:
		nt_weight = notears.run(notears.notears_standard, data, notears.loss.least_squares_loss_cov, notears.loss.least_squares_loss_cov_grad, e=1e-8, verbose=verbose)['W']
	nt_eam = notears.utils.threshold_output(nt_weight, nx.number_of_edges(tg), verbose=verbose)

	cs = notears.utils.compare_graphs_undirected(tam, nt_eam)
	return notears.utils.compare_graphs_precision(cs), notears.utils.compare_graphs_recall(cs)

class SmallGraphTestCase(unittest.TestCase):
    def test_sparse(self):
        """ Test on n=10 e=10 sparse graph """
        prec, recall = simple_test(10,10)
        print("Sparse test, n=10 e=10 -- precision: {}, recall: {}".format(prec, recall))
        self.assertTrue(prec >= 0.9 and recall >= 0.9)

    def test_dense(self):
        """ Test on n=10 e=30 dense graph """
        prec, recall = simple_test(10, 30)
        print("Dense test, n=10 e=30 -- precision: {}, recall: {}".format(prec, recall))
        self.assertTrue(prec >= 0.9 and recall >= 0.9)

    def test_sparse_cov(self):
    	""" Test on n=10 e=10 sparse graph w cov loss """
    	prec, recall = simple_test(10, 10, cov=True)
    	print("Sparse test (cov loss), n=10, e=10 -- precision: {}, recall: {}".format(prec, recall))
    	self.assertTrue(prec >= 0.9 and recall >= 0.9)

    def test_dense(self):
        """ Test on n=10 e=30 dense graph w cov loss """
        prec, recall = simple_test(10, 30, cov=True)
        print("Dense test (cov loss), n=10 e=30 -- precision: {}, recall: {}".format(prec, recall))
        self.assertTrue(prec >= 0.9 and recall >= 0.9)	

if __name__ == '__main__':
    unittest.main()

