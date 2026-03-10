import torch
import numpy as np
import numpy.typing as npt

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    randomly sample a batch of sequences from the dataset.

    for example, if context_length=3 and we sample starting at index 2:
        input:  [x2, x3, x4]
        target: [x3, x4, x5]

    args:
        dataset:        1D numpy array of token IDs
        batch_size:     number of sequences to sample
        context_length: length of each sequence
        device:         pytorch device string, e.g. 'cpu' or 'cuda:0'
    returns:
        inputs:  (batch_size, context_length)  [xi,   xi+1, xi+2]
        targets: (batch_size, context_length)  [xi+1, xi+2, xi+3]
    '''
    max_start = len(dataset) - context_length 
    start_indices = np.random.randint(0, max_start, size=batch_size)   # shape (batch_size,)

    inputs  = np.stack([dataset[i : i + context_length] for i in start_indices])             # shape (batch_size, context_length)
    targets = np.stack([dataset[i + 1 : i + context_length + 1] for i in start_indices])

    inputs  = torch.tensor(inputs,  dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)
    return inputs, targets

