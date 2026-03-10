import math
import os

import numpy as np

from cs336_basics.data import get_batch
from cs336_basics.eval import evaluate
from cs336_basics.loss import cross_entropy
from cs336_basics.model import TransformerLM
from cs336_basics.optim import AdamW, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.utils import load_checkpoint, save_checkpoint


def train(
    # data
    train_path: str,
    val_path: str,
    # model hyperparameters
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    num_layers: int,
    rope_theta: float,
    # training hyperparameters
    batch_size: int,
    total_iters: int,
    warmup_iters: int,
    device: str,
    # optimizer hyperparameters
    max_lr: float,
    min_lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    grad_clip: float,
    # checkpointing / logging
    log_interval: int,
    eval_interval: int,
    save_interval: int,
    checkpoint_path: str | None = None,
) -> None:
    train_data = np.load(train_path, mmap_mode='r')
    val_data = np.load(val_path, mmap_mode='r')

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        theta=rope_theta,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=max_lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
    )

    # Restore from checkpoint if provided.
    start_iter = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_iter = load_checkpoint(checkpoint_path, model, optimizer)
        print(f'Resuming training from iteration {start_iter}...')

    model.train()
    for iter_idx in range(start_iter, total_iters):
        lr = get_lr_cosine_schedule(
            it=iter_idx,
            max_learning_rate=max_lr,
            min_learning_rate=min_lr,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=total_iters,
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        inputs, targets = get_batch(train_data, batch_size, context_length, device)
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), grad_clip)
        optimizer.step()

        if iter_idx % log_interval == 0:
            print(f'Iter {iter_idx}: train loss = {loss.item():.4f}')

        if iter_idx % eval_interval == 0:
            val_loss = evaluate(
                model=model,
                val_data=val_data,
                batch_size=batch_size,
                context_length=context_length,
                vocab_size=vocab_size,
                device=device,
            )
            val_ppl = math.exp(val_loss)
            print(
                f'Iter {iter_idx}: val loss = {val_loss:.4f}, '
                f'val_ppl = {val_ppl:.4f}'
            )
            model.train()

        if iter_idx % save_interval == 0 and checkpoint_path:
            save_checkpoint(model, optimizer, iter_idx, checkpoint_path)
            print(f'Checkpoint saved at iteration {iter_idx}')

    if checkpoint_path:
        save_checkpoint(model, optimizer, total_iters, checkpoint_path)
        print(f'Final checkpoint saved at iteration {total_iters}')
