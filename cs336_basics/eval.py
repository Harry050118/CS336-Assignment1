import torch
import numpy as np
from cs336_basics.data import get_batch
from cs336_basics.loss import cross_entropy

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_data: np.ndarray,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: str,
    num_batches: int = 20
) -> float:
    
    '''
    evaluate model on validation set.
    returns average cross entropy loss over num_batches random batches.
    '''

    model.eval()
    losses = []
    for i in range(num_batches):
        inputs, targets = get_batch(val_data, batch_size, context_length, device)
        logits = model(inputs)
        val_loss = cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        losses.append(val_loss.item())   # .item() to get the scalar value from the tensor
        
    return sum(losses) / len(losses)
