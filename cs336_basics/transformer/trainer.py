import math
import os
from typing import IO, BinaryIO, Callable, Iterable, Iterator, Optional
import torch
from torch import nn
import numpy.typing as npt
import numpy as np
from einops import einsum, rearrange


class AdamWOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 1
                    state['exp_avg'] = torch.zeros_like(param.data)
                    state['exp_avg_sq'] = torch.zeros_like(param.data)
                step = state['step']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = betas
                state['exp_avg'] = beta1 * exp_avg + (1 - beta1) * grad
                state['exp_avg_sq'] = beta2 * exp_avg_sq + (1 - beta2) * grad.pow(2)
                lr_t = lr * math.sqrt(1 - beta2 ** step) / (1 - beta1 ** step)

                param.data -= lr_t * state["exp_avg"] / (torch.sqrt(state["exp_avg_sq"]) + eps)
                param.data -= lr * weight_decay * param.data
                state['step'] += 1
        return loss


def learning_rate_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it < cosine_cycle_iters:
        cos_percent = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
            1 + math.cos(math.pi * cos_percent)
        )
    else:
        return min_learning_rate
