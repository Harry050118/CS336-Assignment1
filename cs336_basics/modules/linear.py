import torch
import torch.nn as nn

class Linear(nn.Module):
    '''
    A simple linear layer that performs a linear transformation on the input data. 
    The layer should have learnable parameters (weights and biases) that are updated during training. 
    The forward method should take an input tensor and return the output tensor after applying the linear transformation.
    '''
    def __init__(self, in_features, out_features, device=None, dtype=None):
        '''
        Initializes the linear layer with the given input and output features. 
        The weights and biases are initialized randomly.
        '''
        super().__init__()  

        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        # 偏置项初始化为0
        self.biases = nn.Parameter(
            torch.zeros(out_features, device=device, dtype=dtype)
        ) 

        # nn.init.kaiming_uniform_(self.weights)
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(
        self, 
        x: torch.Tensor,
    ) -> torch.Tensor:
        
        '''
        Performs the linear transformation on the input tensor x. 
        The output is computed as the matrix multiplication of the input and weights, plus the biases.
        '''

        output = x @ self.W.T
        return output
