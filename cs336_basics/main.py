import random

import numpy as np
import torch

from cs336_basics.train import train


def set_seed(seed: int) -> None:
    '''Set random seed for reproducibility.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    # Random seed
    seed = 42
    set_seed(seed)

    # Data paths (produced by tokenizer/preprocess_to_npy.py)
    train_path = '/mnt/dataset0/gjt/CS336/dataset/TinyStoriesV2-GPT4-tokenizer/train.npy'
    val_path = '/mnt/dataset0/gjt/CS336/dataset/TinyStoriesV2-GPT4-tokenizer/val.npy'

    # Model hyperparameters
    vocab_size = 10000
    context_length = 256
    d_model = 512
    num_heads = 16
    d_ff = 1344
    num_layers = 4
    rope_theta = 10000.0

    # Optimizer hyperparameters
    max_lr = 1e-3
    min_lr = 1e-4
    beta1 = 0.9
    beta2 = 0.95
    eps = 1e-8
    weight_decay = 0.1
    grad_clip = 1.0

    # Training setup
    batch_size = 32
    total_iters = 10000
    warmup_iters = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Logging and checkpointing
    log_interval = 100
    eval_interval = 500
    save_interval = 1000
    checkpoint_path = 'checkpoints/ckpt.pt'

    train(
        train_path=train_path,
        val_path=val_path,
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        rope_theta=rope_theta,
        max_lr=max_lr,
        min_lr=min_lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        batch_size=batch_size,
        total_iters=total_iters,
        warmup_iters=warmup_iters,
        device=device,
        log_interval=log_interval,
        eval_interval=eval_interval,
        save_interval=save_interval,
        checkpoint_path=checkpoint_path,
    )
