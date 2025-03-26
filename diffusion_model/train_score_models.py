import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from diffusion_model.diffusion_sde_model import weighting_function


def clip_grad_norm_per_tensor(model, max_norm=1.5):
    for param in model.parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2)
            if grad_norm > max_norm:
                param.grad.data.mul_(max_norm / grad_norm)


def weighted_mse_loss(prediction, target, weight):
    return torch.mean(weight * torch.mean(torch.square(prediction - target), dim=-1))


############ HIERARCHICAL COMPOSITIONAL SCORE MODEL TRAINING ############

def compute_hierarchical_score_loss(
        theta_global_noisy, target_global, theta_local_noisy, target_local, x_batch,
        diffusion_time, model
):

    # perturb the theta batch
    # snr = model.sde.get_snr(t=diffusion_time)
    # alpha, sigma = model.sde.kernel(log_snr=snr)
    # z_global = alpha * theta_global_batch + sigma * epsilon_global_batch
    # if model.max_number_of_obs > 1:
    #     # global params are not factorized to the same level as local params
    #     z_local = alpha.unsqueeze(1) * theta_local_batch + sigma.unsqueeze(1) * epsilon_local_batch
    # else:
    #     z_local = alpha * theta_local_batch + sigma * epsilon_local_batch
    # predict from perturbed theta
    pred_global, pred_local = model(theta_global=theta_global_noisy, theta_local=theta_local_noisy,
                                    time=diffusion_time, x=x_batch, pred_score=False)

    weight = weighting_function(diffusion_time, sde=model.sde, weighting_type=model.weighting_type,
                                          prediction_type=model.prediction_type)
    # calculate the loss
    loss_global = weighted_mse_loss(pred_global, target_global, weight)
    loss_local = weighted_mse_loss(pred_local, target_local, weight)
    return loss_global + loss_local


############ COMPOSITIONAL SCORE MODEL TRAINING ############

def compute_score_loss(theta_noisy, target, x_batch, diffusion_time, model):
    # predict from perturbed theta
    pred = model(theta=theta_noisy, time=diffusion_time, x=x_batch, pred_score=False)

    weight = weighting_function(diffusion_time, sde=model.sde, weighting_type=model.weighting_type,
                                          prediction_type=model.prediction_type).flatten()
    # calculate the loss
    loss_global = weighted_mse_loss(pred, target, weight)

    #if add_summary_loss and not isinstance(model.summary_net, nn.Identity):
    #    # add extra loss for the summary net
    #    dim_theta = pred.shape[-1]
    #    pred_summary = model.summary_net(x_batch)[..., :dim_theta]
    #    loss_summary = weighted_mse_loss(pred_summary, theta, weight)
    #    loss_global += loss_summary
    return loss_global


########### Training #################


# Training loop for Score Model
def train_score_model(model, dataloader, hierarchical=False, dataloader_valid=None,
                      epochs=1000, lr=5e-4, cosine_annealing=True, add_summary_loss=False, device=None):
    print(f"Training {model.prediction_type}-model for {epochs} epochs with learning rate {lr} "
          f"and {model.weighting_type} weighting.")
    model.to(device)

    if dataloader.dataset.online_learning:  # taken from bayesflow
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    # Add Cosine Annealing Scheduler
    scheduler = None
    if cosine_annealing:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader),
                                                         eta_min=lr**3)  # taken from bayesflow

    # Training loop
    loss_history = np.zeros((epochs, 2))
    for epoch in range(epochs):
        model.train()
        total_loss = []
        # for each sample in the batch, calculate the loss for a random diffusion time
        # param_target, data, time
        with torch.enable_grad():
            for batch in dataloader:
                # initialize the gradients
                optimizer.zero_grad()

                if hierarchical:
                    theta_global_noisy, target_global, theta_local_noisy, target_local, x_batch, diffusion_time = batch
                    theta_global_noisy = theta_global_noisy.to(device)
                    target_global = target_global.to(device)
                    theta_local_noisy = theta_local_noisy.to(device)
                    target_local = target_local.to(device)
                    x_batch = x_batch.to(device)
                    diffusion_time = diffusion_time.to(device)
                    # calculate the loss
                    loss = compute_hierarchical_score_loss(theta_global_noisy=theta_global_noisy,
                                                           target_global=target_global,
                                                           theta_local_noisy=theta_local_noisy,
                                                           target_local=target_local,
                                                           x_batch=x_batch,
                                                           diffusion_time=diffusion_time, model=model)
                else:
                    theta_noisy, target, x_batch, diffusion_time = batch
                    theta_noisy = theta_noisy.to(device)
                    target = target.to(device)
                    x_batch = x_batch.to(device)
                    diffusion_time = diffusion_time.to(device)
                    # calculate the loss
                    loss = compute_score_loss(theta_noisy=theta_noisy, target=target, x_batch=x_batch,
                                              diffusion_time=diffusion_time, model=model)
                loss.backward()
                # gradient clipping
                clip_grad_norm_per_tensor(model, 1.5)  # same as in keras / bayesflow
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                total_loss.append(loss.item())
                # update batch if necessary
                dataloader.dataset.on_batch_end()
        # update dataset if necessary
        dataloader.dataset.on_epoch_end()

        # validate the model
        valid_loss = []
        if dataloader_valid is not None:
            model.eval()
            with torch.no_grad():
                for val_batch in dataloader_valid:
                    if hierarchical:
                        theta_global_noisy, target_global, theta_local_noisy, target_local, x_batch, diffusion_time = val_batch
                        theta_global_noisy = theta_global_noisy.to(device)
                        target_global = target_global.to(device)
                        theta_local_noisy = theta_local_noisy.to(device)
                        target_local = target_local.to(device)
                        x_batch = x_batch.to(device)
                        diffusion_time = diffusion_time.to(device)
                        # calculate the loss
                        v_loss = compute_hierarchical_score_loss(theta_global_noisy=theta_global_noisy,
                                                                 target_global=target_global,
                                                                 theta_local_noisy=theta_local_noisy,
                                                                 target_local=target_local,
                                                                 x_batch=x_batch,
                                                                 diffusion_time=diffusion_time, model=model)
                    else:
                        theta_noisy, target, x_batch, diffusion_time = val_batch
                        theta_noisy = theta_noisy.to(device)
                        target = target.to(device)
                        x_batch = x_batch.to(device)
                        diffusion_time = diffusion_time.to(device)
                        # calculate the loss
                        v_loss = compute_score_loss(theta_noisy=theta_noisy, target=target, x_batch=x_batch,
                                                    diffusion_time=diffusion_time, model=model)
                    valid_loss.append(v_loss.item())
                # update dataset if necessary
                dataloader_valid.dataset.on_epoch_end()

            loss_history[epoch] = [np.mean(total_loss), np.mean(valid_loss)]
            print_str = f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(total_loss):.4f}, "\
                            f"Valid Loss: {np.mean(valid_loss):.4f}"
            print(print_str, end='\r')
        else:
            loss_history[epoch] = np.mean(total_loss)
            print_str = f"Epoch {epoch + 1}/{epochs}, Loss: {np.mean(total_loss):.4f}"
            print(print_str, end='\r')
    return loss_history
