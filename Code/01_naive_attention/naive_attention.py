import torch
import torch.nn as nn
import math
from torchgen.operator_versions.gen_mobile_upgraders_constant import MOBILE_UPGRADERS_HEADER_DESCRIPTION


class NaiveAttention(nn.Module):
    def __init__(self, d_model, head_num , head_dim):
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        self.head_num = head_num

        assert d_model % head_num == 0

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)


    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)



        print("Q shape: ", Q.shape)
        print("K shape: ", K.shape)
        #Rehspae in to (B,H,N,D)
        Q = Q.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)


        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores/math.sqrt(self.head_dim)

        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)

        return attention, output



# #Test
# Batch_size = 2
# Sequence_length = 64
# Hidden_dim = 512
# Attention_Head = 8
# Head_dimension = 64
#
#
#
# x = torch.randn(Batch_size, Sequence_length, Hidden_dim)
#
# model = NaiveAttention(d_model=Hidden_dim, head_num = Attention_Head, head_dim=Head_dimension)
#
# attention, output = model(x)
# print("Attention Shape:-",attention.shape)
# print("output Shape:-",output.shape)