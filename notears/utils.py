import numpy as np
import networkx as nx

def threshold_output(W, desired_edges=None, verbose=False):
    if desired_edges != None:
        # Implements binary search for acyclic graph with the desired number of edges
        ws = sorted([abs(i) for i in np.unique(W)])
        best = W
        done = False
        
        mid = int(len(ws)/2.0)
        floor = 0
        ceil = len(ws)-1
        
        while not done:
            cut = np.array([[0.0 if abs(i) <= ws[mid] else 1.0 for i in j] for j in W])
            g = nx.from_numpy_array(cut, create_using=nx.DiGraph())
            try:
                nx.find_cycle(g)
                floor = mid
                mid = int((ceil-mid)/2.0) + mid
                if mid == ceil or mid == floor:
                    done = True
            except:
                if nx.number_of_edges(g) == desired_edges:
                    best = cut
                    done = True
                elif nx.number_of_edges(g) >= desired_edges:
                    best = cut
                    floor = mid
                    mid = int((ceil-mid)/2.0) + mid
                    if mid == ceil or mid == floor:
                        done = True
                else:
                    best = cut
                    ceil = mid
                    mid = int((floor-mid)/2.0) + floor
                    if mid == ceil or mid == floor:
                        done = True
    else:
        ws = sorted([abs(i) for i in np.unique(W)])
        best = None

        for w in ws:
            cut = np.array([[0.0 if abs(i) <= w else 1.0 for i in j] for j in W])
            g = nx.from_numpy_array(cut, create_using=nx.DiGraph())
            try:
                nx.find_cycle(g)
            except:
                return cut
    return best


def generate_random_dag(num_nodes, num_edges, probabilistic=False, edge_coefficient_range=[0.5, 2.0]):
    adj_mat = np.zeros([num_nodes, num_nodes])

    ranking = np.array([i for i in range(num_nodes)])
    np.random.shuffle(ranking)

    if not probabilistic:
        for e_i in range(num_edges):
            open = np.argwhere(adj_mat == 0)
            open = list(filter(lambda x: ranking[x[0]] < ranking[x[1]], open))
            choice = np.random.choice([i for i in range(len(open))])
            adj_mat[open[choice][0]][open[choice][1]] = np.random.uniform(edge_coefficient_range[0], edge_coefficient_range[1]) * np.random.choice([-1.0, 1.0])


    else:
        p = num_edges/((num_nodes**2)/2.0)
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if ranking[i] < ranking[j]:
                    if np.random.rand() < p:
                           adj_mat[i][j] = np.random.uniform(edge_coefficient_range[0], edge_coefficient_range[1]) * np.random.choice([-1.0, 1.0])
                            
    G_true = nx.from_numpy_matrix(adj_mat, create_using=nx.DiGraph())
    
    return adj_mat, G_true

def simulate_from_dag_lg(tam, n_sample, mean=0, variance=1):
    num_nodes = len(tam)

    def get_value(i, e):
        if values[i] == None:
            val = e[i]
            for j in range(num_nodes):
                if tam[j][i] != 0.0:
                    val += get_value(j, e) * tam[j][i]
            values[i] = val
            return val
        else:
            return values[i]
    
    simulation_data = []
    for i in range(n_sample):
        errors = np.random.normal(mean, variance, num_nodes)
        values = [None for _ in range(num_nodes)]
        for i in range(num_nodes):
            values[i] = get_value(i, errors)
            
        simulation_data.append(values)
        
    return simulation_data

def compare_graphs_undirected(true_graph, estimated_graph):
    num_edges = len(true_graph[np.where(true_graph != 0.0)])

    tam = np.array([[1 if x != 0.0 else 0.0 for x in y] for y in true_graph])
    tam_undir = tam + tam.T
    tam_undir = np.array([[1 if x != 0.0 else 0.0 for x in y] for y in tam_undir])
    tam_undir = np.triu(tam_undir)
    
    eam = np.array([[1 if x != 0.0 else 0.0 for x in y] for y in estimated_graph])
    eam_undir = eam + eam.T
    eam_undir = [[1 if x > 0 else 0 for x in y] for y in eam_undir]
    eam_undir = np.triu(eam_undir)

    tp = len(np.argwhere((tam_undir + eam_undir) == 2))
    fp = len(np.argwhere((tam_undir - eam_undir) < 0))
    tn = len(np.argwhere((tam_undir + eam_undir) == 0))
    fn = num_edges - tp
    
    return [tp, fp, tn, fn]

def compare_graphs_precision(x):
    if x[0] + x[1] == 0: return 0
    return float(x[0]) / float(x[0] + x[1])

def compare_graphs_recall(x):
    if x[0] + x[3] == 0: return 0
    return float(x[0]) / float(x[0] + x[3])

def compare_graphs_specificity(x):
    if x[2] + x[1] == 0: return 0
    return float(x[2]) / float(x[2] + x[1])
