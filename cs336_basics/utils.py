import os
from typing import IO, BinaryIO
import torch

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes]
) -> None:
    '''
    serialize model, optimizer state and iteration number to disk.
    
    saves everything needed to resume training:
        - model weights
        - optimizer state (moment estimates for AdamW, etc.)
        - iteration number (to resume lr schedule)

    args:
        model:     model to save
        optimizer: optimizer to save
        iteration: current training iteration
        out:       file path or file-like object to save to
    '''
    checkpoint = {
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration':            iteration
    }
    print(f'saving checkpoint to {out}...')

    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    '''
    restore model and optimizer state from a checkpoint.
    
    args:
        src:       file path or file-like object to load from
        model:     model to restore weights into
        optimizer: optimizer to restore state into
    returns:
        iteration: the iteration number saved in the checkpoint
    '''
    checkpoint = torch.load(src, weights_only=True)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f'loaded checkpoint from {src} (iteration {checkpoint["iteration"]})')
    
    return checkpoint['iteration']