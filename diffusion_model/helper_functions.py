import torch
from prettytable import PrettyTable


def sech(x):
    return 1 / torch.cosh(x)


def generate_diffusion_time(size, epsilon=0, return_batch=False, weighted_time=False, device=None):
    """
    Generates equally spaced diffusion time values in [epsilon,1].
    The time is generated uniformly in [epsilon, 1] if return_batch is True.

    Epsilon is used for sampling later, in training we usually define a smaller value than epsilon.
    """
    if not return_batch and not weighted_time:
        time = torch.linspace(epsilon, 1, steps=size, dtype=torch.float32, device=device)
        return time
    if weighted_time:
        raise NotImplementedError("Weighted time points not implemented.")

    #time = torch.rand(size, dtype=torch.float32, device=device) * (1 - epsilon) + epsilon
    #beta_dist = torch.distributions.Beta(1, 3)
    #samples = beta_dist.sample((size,))
    #time = epsilon + (1 - epsilon) * samples
    # low discrepancy sequence
    # t_i = \mod (u_0 + i/k, 1)
    u0 = torch.rand(1, dtype=torch.float32, device=device)
    i = torch.arange(0, size, dtype=torch.float32, device=device)  # i as a tensor of indices
    time = ((u0 + i / size) % 1) * (1 - epsilon) + epsilon
    #time, _ = time.sort()

    # Add a new dimension so that each tensor has shape (size, 1)
    return time.unsqueeze(1)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
