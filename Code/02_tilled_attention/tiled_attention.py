import torch
import math

def tiled_attention(Q,K,V, block_size):
    """
    Q,K,V:
          [B,H,N,D]

    Return:
        o: [B,H,N,D]
    """
    B,H,N,D = Q.shape

    O = torch.zeros_like(Q)

    scale =  1.0 / math.sqrt(D)

    for q_start in range(0, N, block_size):

        q_end = min(q_start + block_size, N)

        Q_block = Q[:,:,q_start:q_end,:]


        m = torch.full((B,H,q_end-q_start), -float("inf"), device=Q.device,dtype=torch.float32)

        l = torch.zeros((B,H,q_end-q_start),device=Q.device,dtype=torch.float32)

        #Output accumulator
        O_block = torch.zeros((B,H,q_end-q_start,D), device=Q.device,dtype=torch.float32)


        for k_start in range(0,N,block_size):
            k_end = min(k_start + block_size, N)

            K_block = K[:,:,k_start:k_end,:]
            V_block = V[:,:,k_start:k_end,:]

            #Local Score Block
            S = torch.matmul(Q_block,K_block.transpose(-2,-1)) * scale

            #Online softmax update
            m_block = torch.max(S, dim=-1).values

            m_new = torch.maximum(m,m_block)

            exp_old = torch.exp(m-m_new)

            exp_new = torch.exp(S - m_new.unsqueeze(-1))

            l_new = (l* exp_old + exp_new.sum(dim=-1))

            #update output

            O_block = ( O_block * (l * exp_old / l_new).unsqueeze(-1) + torch.matmul(exp_new,V_block.float()) / l_new.unsqueeze(-1))

            #Update Statistics

            m = m_new
            l = l_new

            O[:,:,q_start:q_end,:] = O_block.to(O.dtype)

        return O


#Test
B = 2
H = 8
N = 4096
D = 64
block_size = 128
Q = torch.randn(B,H,N,D)
K = torch.randn(B,H,N,D)
V = torch.randn(B,H,N,D)

O = tiled_attention(Q,K,V,block_size)

print(O.shape)







