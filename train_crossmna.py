# encoding: utf8
from utils import *
from model import *
from AUC import *
import time
import torch
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import torch.nn as nn
import torch.utils.data as Data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0,1'

parser = ArgumentParser("network alignment",
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          conflict_handler='resolve')
parser.add_argument('--task', default='LinkPrediction', type=str)
parser.add_argument('--node-dim', default=200, type=int)
parser.add_argument('--layer-dim', default=200, type=int)
parser.add_argument('--batch-size', default=512, type=int)
parser.add_argument('--neg-samples', default=5, type=int)
parser.add_argument('--output', default='node2vec.pk', type=str)
args = parser.parse_args()

"""step 1. load data"""
def train_crossmna(dataset,undirected=True):
    layers, num_nodes, id2node = readfile(dataset,undirected=undirected)
    num_layers = len(layers.keys())
    print("num_laers", num_layers)

    """step 2. initial negative sampling table"""
    for layerid in layers:
        g = layers[layerid]
        g.init_neg()



    """step 3. create model"""
    model = CrossMNA(num_of_nodes=num_nodes, 
                            batch_size=args.batch_size,
                            node_dim=args.node_dim,
                            num_layer=num_layers, 
                            layer_dim=args.layer_dim
            ) 


    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    data_i, data_j, data_labels, data_layer = gen_batches(layers, K=args.neg_samples)
    data_i = torch.Tensor(data_i)
    data_j = torch.Tensor(data_j)
    data_labels = torch.Tensor(data_labels)
    data_layer = torch.Tensor(data_layer)
    torch_dataset = Data.TensorDataset(data_i ,data_j, data_labels, data_layer)
    """step 4. start training session"""

    for epoch in range(100):
        t = time.clock()
        loader = Data.DataLoader(dataset = torch_dataset, batch_size = args.batch_size *(args.neg_samples+1), shuffle = True, num_workers = 2)
        print("epoch {0}: time for generate batches={1}s".format(epoch, time.clock()-t))

        total_loss = 0.0
        t = time.clock()
        for batch_i, batch_j, batch_label, batch_layer in loader:
            u_i = Variable(batch_i)
            u_j = Variable(batch_j)
            labels = Variable(batch_label)
            this_layer = Variable(batch_layer)
            loss = model(u_i, u_j, this_layer,labels)
            optimizer.zero_grad()            
            total_loss += loss
            loss.backward()
            optimizer.step()
            # _, loss = sess.run([train_op, model.loss], feed_dict=feed_dict)

            
        print("epoch {0}: time for training={1},  total_loss={2} ".format(epoch, time.clock()-t, total_loss))
        if epoch  ==  99:
            
            if args.task == 'LinkPrediction':
                inter_vectors = model.n_embedding.weight.detach().numpy()
                W = model.w.detach().numpy()
                layers_embedding = model.l_embedding.weight.detach().numpy()

                node2vec = get_intra_emb(inter_vectors, W, layers_embedding, layers, id2node)

                return node2vec

