import torch
import math
from typing import Callable
from typing import Iterable

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,   # alpha_max
    min_learning_rate: float,   # alpha_min
    warmup_iters: int,          # T_w
    cosine_cycle_iters: int,    # T_c
) -> float:
    '''
    warmup  (t < T_w):          alpha_t = t/T_w * alpha_max
    cosine  (T_w <= t <= T_c):  alpha_t = alpha_min + 0.5*(1 + cos((t-T_w)/(T_c-T_w)*pi)) * (alpha_max - alpha_min)
    post    (t > T_c):          alpha_t = alpha_min
    '''
    if warmup_iters > 0 and it < warmup_iters:
        # linear warmup from 0 to alpha_max
        return (it / warmup_iters) * max_learning_rate

    if it <= cosine_cycle_iters:
        # cosine annealing from alpha_max to alpha_min
        if warmup_iters == cosine_cycle_iters:
            return max_learning_rate  
        
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(progress * math.pi)) * (max_learning_rate - min_learning_rate)

    return min_learning_rate


class AdamW(torch.optim.Optimizer):
    '''
    AdamW optimizer (Loshchilov & Hutter, 2019)
    decouples weight decay from the gradient update:

    m = beta1 * m + (1 - beta1) * g          # first moment estimate
    v = beta2 * v + (1 - beta2) * g^2        # second moment estimate
    alpha_t = alpha * sqrt(1 - beta2^t) / (1 - beta1^t)   # bias correction
    theta = theta - alpha_t * m / (sqrt(v) + eps)          # gradient update
    theta = theta - alpha * lambda * theta                  # weight decay
    '''
    def __init__(
        self,
        params,
        lr: float = 1e-3,           # alpha: learning rate
        betas: tuple = (0.9, 0.999),# (beta1, beta2): moment decay rates
        eps: float = 1e-8,          # numerical stability constant
        weight_decay: float = 0.01  # lambda: weight decay coefficient
    ):
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr           = group['lr']
            beta1, beta2 = group['betas']
            eps          = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]

                # initialize state on first step
                if len(state) == 0:
                    state['t'] = 0                              # iteration counter
                    state['m'] = torch.zeros_like(p.data)      # first moment
                    state['v'] = torch.zeros_like(p.data)      # second moment

                state['t'] += 1
                t = state['t']
                m = state['m']
                v = state['v']

                # update first and second moment estimates
                m.mul_(beta1).add_(grad, alpha=1 - beta1)          # m = beta1*m + (1-beta1)*g
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)# v = beta2*v + (1-beta2)*g^2

                # bias-corrected learning rate
                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                # gradient update: theta = theta - alpha_t * m / (sqrt(v) + eps)
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)

                # weight decay: theta = theta - alpha * lambda * theta
                p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss

@torch.no_grad()
def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6   
) -> None:
    '''
    clip combined gradient of all parameters to have l2 norm at most max_l2_norm:
    if ||g||_2 <= max_l2_norm: do nothing
    else: g = g * max_l2_norm / (||g||_2 + eps)

    modifies parameter gradients in-place.
    '''
    # collect all gradients that exist
    grads = [p.grad for p in parameters if p.grad is not None]

    if len(grads) == 0:
        return

    # compute global l2 norm across all parameters
    total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)   # in-place scaling
