# ******************************************************************************
# main.py
#
# Date      Name       Description
# ========  =========  ========================================================
# 2/24/21   Paudel     Initial version,
# ******************************************************************************

from utils import DataUtils, GraphUtils
from anomaly_detection import AnomalyDetection
from pikachu import PIKACHU
import argparse
from tqdm import tqdm
import pickle


import numpy as np

import warnings
warnings.filterwarnings("ignore")

import resource

# Define o limite máximo de memória em bytes (por exemplo, 2 GB)
# 2 GB = 2 * 1024 * 1024 * 1024 bytes
#memory_limit_bytes = 64 * 1024 * 1024 * 1024

# Define o limite máximo de memória para o programa
#resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    '''
    Usual pythonic way of parsing command line arguments
    :return: all command line arguments read
    '''
    #'/mnt/hdd/jfmvs/OpTC/testes/0+324k_30mins.csv'
    args = argparse.ArgumentParser("walk")
    args.add_argument('-k', "--dimensions", default = 64, type=int, help="Number of Dimension")
    args.add_argument('-ip', '--input', default = '/mnt/hdd/jfmvs/LANL/LANL_snapshots_4horas.csv', help='Dataset filename')
    args.add_argument('-l', '--walklen', default = 100, type=int, help='Walk length')
    args.add_argument('-n', '--numwalk', default = 1, type=int, help='Walk length')
    args.add_argument('-w', '--trainwin', type=int, default = 4,
                        help='Context size for optimization. Default is 2.')
    args.add_argument('-e', '--epoch', default= 3, type=int,
                        help='Number of epochs (for GRU)')
    args.add_argument('-i', '--iter', default=10, type=int,
                      help='Number of iteration (for edge probability estimation)')
    args.add_argument('-r', '--alpha', default=0.001, type=float,
                      help='Learning rate for edge probability estimation')
    args.add_argument('-s', '--support', default=10, type=int,
                      help='Support Set (# of neighbor for edge probability estimation)')
    args.add_argument('-t', '--train', nargs='?', const=True, default = True, type=str2bool,
                      help='Is training?')
    # args.add_argument('-o', '--output', type=str, default='results/anomalous_edges.csv',
                        # help='Output file for anomalous edges')
    args.add_argument('-d', '--dataset', type=str, default='LANL_4horas',
                      help='Name of the dataset')
    return args.parse_args()


if __name__=="__main__":
    args = parse_args()

    file = "LANL_4horas"
    data_file = '_' + args.dataset + '_' + file + '.pickle'
    
    if args.train:
        print("... Parsing Data ... \n")
        dp = DataUtils(data_folder=args.input)
        # data_df = dp.get_data()
        data_df, node_map = dp.get_data()
        with open('weights/node_map' + data_file, 'wb') as f:
           pickle.dump(node_map, f)
        print("... Generating Graphs ... \n")
        g_util = GraphUtils(node_map)
        graphs = []
        for t in tqdm(data_df.snapshot.unique()):
            graphs.append(g_util.create_graph(data_df[data_df['snapshot'] == t]))
        with open('weights/graphs' + data_file, 'wb') as f:
           pickle.dump(graphs, f)

    # '''
    with open('weights/node_map' + data_file, 'rb') as f:
        node_map = pickle.load(f)

    with open('weights/graphs' + data_file, 'rb') as f:
        graphs = pickle.load(f)

    print("\nTotal Graphs: ", len(graphs))
    print(len(graphs[0].edges()))
    node_list = [str(v) for k, v in node_map.items()]
    print("\nTotal Nodes: ", len(node_map))

    print("... Starting Graph Embedding ... \n")

    weight_file = '_' + args.dataset + '_' + file + '_d' + str(args.dimensions) + '.pickle'
    print(" \n********** PARAM **********\n", args)
    print("Weight Files: ", weight_file)
    print("********************\n")
    if args.train:
        pikachu = PIKACHU(args=args, node_list=node_list, node_map=node_map, graphs=graphs)
        pikachu.learn_embedding(weight_file)

    print("\n\n =====   Anomaly Detection ===== ")
    with open('weights/long_term' + weight_file, 'rb') as f:
        long_term_embs = pickle.load(f)
    long_term_embs = np.transpose(long_term_embs, (1, 0, 2))
    ad_long_term = AnomalyDetection(args, node_list, node_map, long_term_embs, idx = 0)
    param_file_name = '_' + args.dataset + '_' + file + '_d' + str(args.dimensions)
    ad_long_term.anomaly_detection(graphs, param_file='weights/param' + param_file_name)
    del ad_long_term