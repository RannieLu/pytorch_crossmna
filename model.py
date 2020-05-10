import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import LogSigmoid

class CrossMNA(nn.Module):
    
    def __init__(self, num_of_nodes, node_dim, layer_dim,
    batch_size, num_layer):
        super(CrossMNA, self).__init__()
        initrange = 0.5 / node_dim
        self.n_embedding = nn.Embedding(num_of_nodes, node_dim)
        self.n_embedding.weight.data.uniform_(-initrange, initrange)
        self.l_embedding = nn.Embedding(num_layer, layer_dim)
        self.l_embedding.weight.data.uniform_(-1,1)
        print("l embedding", self.l_embedding.weight.shape)
        self.w = nn.Parameter(torch.Tensor(node_dim, layer_dim))
        nn.init.normal_(self.w)
    
        
    def forward(self, i, j, l,label):
        n_i = self.n_embedding(i.long())
        n_j = self.n_embedding(j.long())
       # print(n_i.weight.shape," ",self.w.shape)
        l_tmp = self.l_embedding(l.long())
        l_i = l_tmp + torch.matmul(n_i,self.w)
        l_j = l_tmp + torch.matmul( n_j,self.w)
        loss = torch.sum(l_i * l_j)
        log_loss = LogSigmoid()
        loss = -torch.sum(log_loss(label * loss))
        return loss

        
        
    
    # def backward(self):
    #     pass



