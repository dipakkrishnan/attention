# Reference: https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention

import torch
from torch.nn import Parameter, Embedding, Module
import torch.nn.functional as F

# Set torch seed
torch.manual_seed(123)


class SelfAttention(Module):

    def __init__(self, d: int, d_q: int, d_k: int, d_v: int):
        """
        Implements self-attention given user-specified dimensions of query, key, value matrices.

        :param d: dimensionality of the embedding vector that self-attention is being performed on
        :param d_q: query matrix dimensionality (d_q == d_k in practice)
        :param d_k: key matrix dimensionality (d_q == d_k in practice)
        :param d_v: value matrix dimensionality
        """
        super().__init__()
        # Save key dimension for score norm
        self.d_k = d_k

        # Initialize weight matrices
        self.wq = Parameter(torch.rand(d, d_q))
        self.wk = Parameter(torch.rand(d, d_k))
        self.wv = Parameter(torch.rand(d, d_v))

    def forward(self, X: torch.tensor):
        """ 
        Implements the forward pass for updating attention weights. 
        
        - Calculates attention scores
        - Normalizes attention scores
        - Obtain context vector by multiplying attention weights with values
        """
        try:
            # Calculate Q, K, V matrices
            Q = X @ self.wq
            K = X @ self.wk
            V = X @ self.wv

            print(Q)

            print(K)

            print(V)

            # Calculate unnormalized attention weights
            omega = Q @ K.T

            # Normalize attention weights
            norm_omega = F.softmax(
                omega / (self.d_k ** 0.5),
                dim=-1
            )

            # Obtain context vector
            return norm_omega @ V

        except Exception as e:
            # Catch general exceptions 
            print("Failed to execute forward pass.")
            raise e            

vocab_size = 50000

sentence = "A quick brown fox ran in the woods."

dc = {s:i for i,s 
      in enumerate(sorted(sentence.replace(',', '').split()))}

sentence_int = torch.tensor(
    [dc[s] for s in sentence.replace(',', '').split()]
)

embedding = Embedding(vocab_size, 3)

X = embedding(sentence_int)

attn = SelfAttention(
    d=X.shape[1],
    d_k=2,
    d_q=2,
    d_v=4
)

print(attn.forward(X))