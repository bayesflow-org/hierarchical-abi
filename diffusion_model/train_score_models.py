import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from diffusion_model.diffusion_sde_model import weighting_function
from diffusion_model.helper_functions import generate_diffusion_time


############ HIERARCHICAL COMPOSITIONAL SCORE MODEL TRAINING ############

def compute_hierarchical_score_loss(
        theta_global_batch, theta_local_batch, x_batch, model,
        epsilon_global_batch=None, epsilon_local_batch=None
):
    # Generate diffusion time and step size
    diffusion_time = generate_diffusion_time(size=theta_global_batch.shape[0],
                                             return_batch=True, device=theta_global_batch.device)

    # sample from the Gaussian kernel, just learn the noise
    if epsilon_global_batch is None:
        epsilon_global_batch = torch.randn_like(theta_global_batch, dtype=theta_global_batch.dtype,
                                                device=theta_global_batch.device)
    if epsilon_local_batch is None:
        epsilon_local_batch = torch.randn_like(theta_local_batch, dtype=theta_local_batch.dtype,
                                               device=theta_local_batch.device)

    # perturb the theta batch
    snr = model.sde.get_snr(t=diffusion_time)
    alpha, sigma = model.sde.kernel(log_snr=snr)
    z_global = alpha * theta_global_batch + sigma * epsilon_global_batch
    if model.max_number_of_obs > 1:
        # global params are not factorized to the same level as local params
        z_local = alpha.unsqueeze(1) * theta_local_batch + sigma.unsqueeze(1) * epsilon_local_batch
    else:
        z_local = alpha * theta_local_batch + sigma * epsilon_local_batch
    # predict from perturbed theta
    pred_global, pred_local = model(theta_global=z_global, theta_local=z_local,
                                    time=diffusion_time, x=x_batch, pred_score=False)

    effective_weight = weighting_function(diffusion_time, sde=model.sde, weighting_type=model.weighting_type,
                                          prediction_type=model.prediction_type)
    # calculate the loss (sum over the last dimension, mean over the batch)
    loss_global = torch.mean(effective_weight * torch.sum(torch.square(pred_global - epsilon_global_batch), dim=-1))
    loss_local = torch.mean(effective_weight * torch.sum(torch.square(pred_local - epsilon_local_batch), dim=-1))
    return loss_global + loss_local


# Training loop for Score Model
def train_hierarchical_score_model(model, dataloader, dataloader_valid=None,
                                   epochs=100, lr=1e-3, cosine_annealing=True,
                                   rectified_flow=False, device=None):
    print(f"Training {model.prediction_type}-model for {epochs} epochs with learning rate {lr} "
          f"and {model.weighting_type} weighting.")
    if model.sde.noise_schedule == 'flow_matching':
        rectified_flow = True
    if rectified_flow:
        print(f'Using rectified flow.')
    print(f"Model has {model.n_params_global} global parameters and {model.n_params_local} local parameters and is"
          f" uses compositional conditioning with {model.max_number_of_obs} observations.")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Add Cosine Annealing Scheduler
    scheduler = None
    if cosine_annealing:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Training loop
    loss_history = np.zeros((epochs, 2))
    for epoch in range(epochs):
        model.train()
        total_loss = []
        # for each sample in the batch, calculate the loss for a random diffusion time
        for theta_global_batch, epsilon_global_batch, theta_local_batch, epsilon_local_batch, x_batch in dataloader:
            # initialize the gradients
            optimizer.zero_grad()
            theta_global_batch = theta_global_batch.to(device)
            theta_local_batch = theta_local_batch.to(device)
            x_batch = x_batch.to(device)
            if rectified_flow:
                epsilon_global_batch = epsilon_global_batch.to(device)
                epsilon_local_batch = epsilon_local_batch.to(device)
                # calculate the loss
                loss = compute_hierarchical_score_loss(theta_global_batch=theta_global_batch,
                                                       theta_local_batch=theta_local_batch,
                                                       epsilon_global_batch=epsilon_global_batch,
                                                       epsilon_local_batch=epsilon_local_batch,
                                                       x_batch=x_batch, model=model)
            else:
                # calculate the loss
                loss = compute_hierarchical_score_loss(theta_global_batch=theta_global_batch,
                                                       theta_local_batch=theta_local_batch,
                                                       x_batch=x_batch, model=model)
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            total_loss.append(loss.item())
            # update batch if necessary
            dataloader.dataset.on_batch_end()
        if scheduler is not None:
            scheduler.step()
        # update dataset if necessary
        dataloader.dataset.on_epoch_end()

        # validate the model
        valid_loss = []
        if dataloader_valid is not None:
            model.eval()
            with torch.no_grad():
                for theta_global_batch, epsilon_global_batch, theta_local_batch, epsilon_local_batch, x_batch in dataloader_valid:
                    theta_global_batch = theta_global_batch.to(device)
                    theta_local_batch = theta_local_batch.to(device)
                    x_batch = x_batch.to(device)
                    if rectified_flow:
                        epsilon_global_batch = epsilon_global_batch.to(device)
                        epsilon_local_batch = epsilon_local_batch.to(device)
                        # calculate the loss
                        loss = compute_hierarchical_score_loss(theta_global_batch=theta_global_batch,
                                                               theta_local_batch=theta_local_batch,
                                                               epsilon_global_batch=epsilon_global_batch,
                                                               epsilon_local_batch=epsilon_local_batch,
                                                               x_batch=x_batch, model=model)
                    else:
                        # calculate the loss
                        loss = compute_hierarchical_score_loss(theta_global_batch=theta_global_batch,
                                                               theta_local_batch=theta_local_batch,
                                                               x_batch=x_batch, model=model)
                    valid_loss.append(loss.item())

        loss_history[epoch] = [np.mean(total_loss), np.mean(valid_loss)]
        print_str = f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(total_loss):.4f}, " \
                    f"Valid Loss: {np.mean(valid_loss):.4f}"
        print(print_str, end='\r')
    return loss_history


def compute_score_loss(theta_global_batch, x_batch, model, epsilon_global_batch=None):
    # Generate diffusion time and step size
    diffusion_time = generate_diffusion_time(size=theta_global_batch.shape[0],
                                             return_batch=True, device=theta_global_batch.device)

    # sample from the Gaussian kernel, just learn the noise
    if epsilon_global_batch is None:
        epsilon_global_batch = torch.randn_like(theta_global_batch, dtype=theta_global_batch.dtype,
                                                device=theta_global_batch.device)

    # perturb the theta batch
    snr = model.sde.get_snr(t=diffusion_time)
    alpha, sigma = model.sde.kernel(log_snr=snr)
    z_global = alpha * theta_global_batch + sigma * epsilon_global_batch
    # predict from perturbed theta
    pred_global = model(theta_global=z_global, time=diffusion_time, x=x_batch, pred_score=False)

    effective_weight = weighting_function(diffusion_time, sde=model.sde, weighting_type=model.weighting_type,
                                          prediction_type=model.prediction_type)
    # calculate the loss (sum over the last dimension, mean over the batch)
    loss_global = torch.mean(effective_weight * torch.sum(torch.square(pred_global - epsilon_global_batch), dim=-1))
    return loss_global



############ COMPOSITIONAL SCORE MODEL TRAINING ############

# Training loop for Score Model
def train_score_model(model, dataloader, dataloader_valid=None,
                      epochs=100, lr=1e-3, cosine_annealing=True,
                      rectified_flow=False, device=None):
    print(f"Training {model.prediction_type}-model for {epochs} epochs with learning rate {lr} "
          f"and {model.weighting_type} weighting.")
    if model.sde.noise_schedule == 'flow_matching':
        rectified_flow = True
    if rectified_flow:
        print(f'Using rectified flow.')
    print(f"Model has {model.n_params_global} parameters and is uses compositional conditioning "
          f"with {model.max_number_of_obs} observations.")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Add Cosine Annealing Scheduler
    scheduler = None
    if cosine_annealing:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Training loop
    loss_history = np.zeros((epochs, 2))
    for epoch in range(epochs):
        model.train()
        total_loss = []
        # for each sample in the batch, calculate the loss for a random diffusion time
        for theta_global_batch, epsilon_global_batch, x_batch in dataloader:
            # initialize the gradients
            optimizer.zero_grad()
            theta_global_batch = theta_global_batch.to(device)
            x_batch = x_batch.to(device)
            if rectified_flow:
                epsilon_global_batch = epsilon_global_batch.to(device)
                # calculate the loss
                loss = compute_score_loss(theta_global_batch=theta_global_batch,
                                          epsilon_global_batch=epsilon_global_batch,
                                          x_batch=x_batch, model=model)
            else:
                # calculate the loss
                loss = compute_score_loss(theta_global_batch=theta_global_batch, x_batch=x_batch, model=model)
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            total_loss.append(loss.item())
            # update batch if necessary
            dataloader.dataset.on_batch_end()
        if scheduler is not None:
            scheduler.step()
        # update dataset if necessary
        dataloader.dataset.on_epoch_end()

        # validate the model
        valid_loss = []
        if dataloader_valid is not None:
            model.eval()
            with torch.no_grad():
                for theta_global_batch, epsilon_global_batch, x_batch in dataloader_valid:
                    theta_global_batch = theta_global_batch.to(device)
                    x_batch = x_batch.to(device)
                    if rectified_flow:
                        epsilon_global_batch = epsilon_global_batch.to(device)
                        # calculate the loss
                        loss = compute_score_loss(theta_global_batch=theta_global_batch,
                                                  epsilon_global_batch=epsilon_global_batch,
                                                  x_batch=x_batch, model=model)
                    else:
                        # calculate the loss
                        loss = compute_score_loss(theta_global_batch=theta_global_batch, x_batch=x_batch, model=model)
                    valid_loss.append(loss.item())

        loss_history[epoch] = [np.mean(total_loss), np.mean(valid_loss)]
        print_str = f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(total_loss):.4f}, "\
                    f"Valid Loss: {np.mean(valid_loss):.4f}"
        print(print_str, end='\r')
    return loss_history
