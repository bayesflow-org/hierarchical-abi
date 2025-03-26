import keras

from tqdm import tqdm

from bayesflow.types import Shape, Tensor
from bayesflow.utils import (
    expand_right_as,
    expand_right_to,
    find_network,
    keras_kwargs,
    serialize_value_or_type,
    deserialize_value_or_type,
)
from bayesflow.networks.inference_network import InferenceNetwork


# noinspection SpellCheckingInspection
@keras.saving.register_keras_serializable(package="bayesflow.networks")
class SimpleDiffusion(InferenceNetwork):
    """Implements Denoising Diffusion simplified from SimpleDiffusion [1].

    [1] Paper 1: arxiv:2301.11093
    """

    MLP_DEFAULT_CONFIG = {
        "widths": (256, 256, 256, 256, 256),
        "activation": "mish",
        "kernel_initializer": "he_normal",
        "residual": True,
        "dropout": 0.05,
        "spectral_normalization": False,
    }

    def __init__(
            self,
            subnet: str | type = "mlp",
            base_distribution: str = "normal",
            timesteps: int = 1000,
            train_time: str = "discrete",
            pred_param: str = "v",
            loss_param: str = "v",
            noise_schedule_name: str = "cosine",
            param_d: int = 512,
            shift_low_d: int = 64,
            shift_high_d: int = 256,
            logsnr_min: float = -15,
            logsnr_max: float = 15,
            return_mean: bool = True,
            **kwargs,
    ):
        super().__init__(base_distribution=base_distribution, **keras_kwargs(kwargs))
        self.seed_generator = keras.random.SeedGenerator()
        self.subnet = subnet
        if subnet == "mlp":
            subnet_kwargs = SimpleDiffusion.MLP_DEFAULT_CONFIG.copy()
            subnet_kwargs.update(kwargs.get("subnet_kwargs", {}))
        else:
            subnet_kwargs = kwargs.get("subnet_kwargs", {})
        self.diffusion_backbone = find_network(subnet, **subnet_kwargs)
        self.requires_projection = False
        if subnet == "mlp":
            self.projector = keras.layers.Dense(
                units=None,
                bias_initializer="zeros",
                name="projector"
            )
            self.requires_projection = True
        self.timesteps = timesteps
        self.train_time = train_time
        self.pred_param = pred_param
        self.loss_param = loss_param
        self.return_mean = return_mean
        self.min_snr_weighting = kwargs.get("min_snr_weighting", False)
        self.schedule = {
            'name': noise_schedule_name,
            'image_d': param_d,
            'shift_low_d': shift_low_d,
            'shift_high_d': shift_high_d,
            'logsnr_min': logsnr_min,
            'logsnr_max': logsnr_max,
        }

        self.loss = keras.losses.MeanSquaredError(reduction=None)

        # serialization: store all parameters necessary to call __init__
        self.config = {
            "base_distribution": base_distribution,
            **kwargs,
        }
        self.config = serialize_value_or_type(self.config, "subnet", subnet)

    def get_config(self):
        base_config = super().get_config()
        return base_config | self.config

    @classmethod
    def from_config(cls, config):
        config = deserialize_value_or_type(config, "subnet")
        return cls(**config)

    def build(self, xz_shape: Shape, conditions_shape: Shape = None) -> None:
        super().build(xz_shape)
        input_shape = list(xz_shape)
        input_shape[-1] += 1
        if conditions_shape is not None:
            input_shape[-1] += conditions_shape[-1]
        input_shape = tuple(input_shape)
        self.diffusion_backbone.build(input_shape)
        input_shape = self.diffusion_backbone.compute_output_shape(input_shape)
        if self.subnet == 'mlp':
            self.projector.units = xz_shape[-1]
            self.projector.build(input_shape)

    def _forward(self, x: Tensor, conditions: Tensor = None, training: bool = False, **kwargs) -> Tensor:
        raise NotImplementedError("DenoisingDiffusion._forward not implemented")

    def _inverse(self, z: Tensor, conditions: Tensor = None, training: bool = False, **kwargs) -> Tensor:
        """
        Performs the inverse diffusion process to generate samples from the prior distribution.
        :param z: (Tensor) The input noise tensor sampled from the base distribution.
        :param conditions: (Tensor) The condition tensor.
        :param training: (bool) Whether the model is in training mode.
        :param kwargs:
        :return: x_pred (Tensor): The generated sample.
        """
        z_t = z
        ds = keras.ops.shape(z)[0]
        bs = keras.ops.shape(z)[1]
        time = keras.ops.linspace(1.0, 0.0, self.timesteps+1)
        for t_curr, t_next in tqdm(zip(time[:-1], time[1:]), desc='Diffusion sampling', total=self.timesteps):
            t_curr = keras.ops.full([ds, bs], t_curr, dtype=keras.ops.dtype(z_t))
            t_next = keras.ops.full([ds, bs], t_next, dtype=keras.ops.dtype(z_t))
            logsnr_curr = expand_right_as(self.get_logsnr(t_curr), z_t)
            logsnr_next = expand_right_as(self.get_logsnr(t_next), z_t)
            if conditions is not None:
                z_t_c = keras.ops.concatenate([z_t, conditions, logsnr_curr], axis=-1)
                #t = expand_right_as(t_curr, z_t)
                #z_t_c = keras.ops.concatenate([z_t, conditions, t], axis=-1)
            else:
                z_t_c = keras.ops.concatenate([z_t, logsnr_curr], axis=-1)
                #t = expand_right_as(t_curr, z_t)
                #z_t_c = keras.ops.concatenate([z_t, t], axis=-1)
            pred = self.diffusion_backbone(z_t_c, training=False)
            if self.requires_projection:
                pred = self.projector(pred, training=False)
            mu, variance = self.ddpm_sampler_step(z_t, pred, logsnr_curr, logsnr_next)
            z_t = mu + keras.random.normal(keras.ops.shape(mu)) * keras.ops.sqrt(variance)
        x_pred = self.clip(mu) if self.return_mean else z_t
        return x_pred

    def compute_metrics(self, x: Tensor, conditions: Tensor = None, stage: str = "training", sample_weight=None) -> dict[str, Tensor]:
        base_metrics = super().compute_metrics(x, conditions=conditions, stage=stage)
        loss = self.sd_loss(x, conditions, training=stage == "training")
        return base_metrics | {'loss': loss}

    def logsnr_schedule(self, t: Tensor, image_d: int, noise_d: int) -> Tensor:
        """
        cosine: standard cosine schedule. Not recommended for high resolution image generation but serves is used in the
        other two schedules for computing unshifted logSNR values. t_min and t_max will also be shifted.
        """
        logsnr_max = self.schedule['logsnr_max'] + keras.ops.log(noise_d/image_d)
        logsnr_min = self.schedule['logsnr_min'] + keras.ops.log(noise_d/image_d)
        t_min = keras.ops.arctan(keras.ops.exp(-0.5 * logsnr_max))
        t_max = keras.ops.arctan(keras.ops.exp(-0.5 * logsnr_min))
        logsnr_t = -2.0 * keras.ops.log(keras.ops.tan(t_min + (t_max - t_min) * t))
        return logsnr_t

    def logsnr_schedule_cosine_shifted(self, t: Tensor, image_d: int, noise_d: int) -> Tensor:
        """
        cosine_shifted: cosine schedule with shifted logSNR. Recommended with 64x64 reference noise due to better
        performance in early iterations although 32x32 reference noise is better overall for image sizes 128x128 and
        256x256 (Table 2).
        """
        logsnr_t = self.logsnr_schedule(t, image_d, noise_d)
        logsnr_t_shifted = logsnr_t + 2 * keras.ops.log(noise_d/image_d)
        return logsnr_t_shifted

    def logsnr_schedule_interpolated(self, t: Tensor) -> Tensor:
        """
        cosine_interpolated: cosine schedule with interpolated logSNR between two reference noise levels.
        Recommended when sampling guidance is desired. E.g. low noise = 32x32, high noise = 256x256 for image resolution
        512x512.
        """
        logsnr_low_shifted = self.logsnr_schedule_cosine_shifted(t, self.schedule['image_d'], self.schedule['shift_low_d'])
        logsnr_high_shifted = self.logsnr_schedule_cosine_shifted(t, self.schedule['image_d'], self.schedule['shift_high_d'])
        logsnr_interpolated = t * logsnr_low_shifted + (keras.ops.ones_like(t) - t)*logsnr_high_shifted
        return logsnr_interpolated

    def logsnr_schedule_ddpm(self, t: Tensor) -> Tensor:
        logsnr_max = self.schedule['logsnr_max'] + keras.ops.log(self.schedule['shift_low_d']/self.schedule['image_d'])
        logsnr_min = self.schedule['logsnr_min'] + keras.ops.log(self.schedule['shift_low_d']/self.schedule['image_d'])
        t_min = keras.ops.sqrt(keras.ops.log(1 + keras.ops.exp(-logsnr_max)))
        t_max = keras.ops.sqrt(keras.ops.log(1 + keras.ops.exp(-logsnr_min)))
        t_trunc = t_min + (t_max - t_min) * t
        logsnr_ddpm = -keras.ops.log(keras.ops.exp(keras.ops.square(t_trunc)) - 1)
        return logsnr_ddpm

    def get_logsnr(self, t: Tensor) -> Tensor:
        """
        Function to compute the logSNR value at timestep t. The logSNR value is computed based on the schedule selected.

        Args:
        t (tf.Tensor): The timestep in [0, 1].

        Returns:
        logsnr (tf.Tensor): The logSNR value at timestep t.
        """
        if self.schedule['name'] == 'cosine_shifted':
            return self.logsnr_schedule_cosine_shifted(t, self.schedule['image_d'], self.schedule['shift_low_d'])
        elif self.schedule['name'] == 'cosine_interpolated':
            return self.logsnr_schedule_interpolated(t)
        elif self.schedule['name'] == 'cosine':
            return self.logsnr_schedule(t, self.schedule['image_d'], self.schedule['image_d'])
        elif self.schedule['name'] == 'ddpm':
            return self.logsnr_schedule_ddpm(t)
        else:
            raise NotImplementedError(f"Schedule {self.schedule['name']} not implemented")

    def clip(self, x: Tensor) -> Tensor:
        clipped_x = keras.ops.clip(x, -1, 1)
        return clipped_x

    def diffuse(self, x: Tensor, alpha_t: Tensor, sigma_t: Tensor) -> tuple[Tensor, Tensor]:
        """
        Diffuses the input tensor x with the given alpha_t and sigma_t values.
        Args:
            x (Tensor): The input tensor.
            alpha_t (Tensor): The alpha_t value.
            sigma_t (Tensor): The sigma_t value.

        Returns:
            z_t (Tensor): The diffused tensor.
            eps_t (Tensor): The noise tensor.
        """
        eps_t = keras.random.normal(keras.ops.shape(x))
        z_t = alpha_t * x + sigma_t * eps_t
        return z_t, eps_t

    def ddpm_sampler_step(self, z_t: Tensor, pred: Tensor, logsnr_t: Tensor, logsnr_s: Tensor) -> tuple[Tensor, Tensor]:
        """
        Performs a single DDPM sampling step as described in the appendix of [1].

        Args:
            z_t (Tensor): The diffused sample at time t.
            pred (Tensor): The model's prediction ('v' or 'eps').
            logsnr_t (Tensor): The logSNR values at time t.
            logsnr_s (Tensor): The logSNR values at the sampling time s.

        Returns:
            mu (Tensor): The sampled tensor at time s.
            variance (Tensor): The variance of the sample at time s.
        """
        c = expand_right_as(-keras.ops.expm1(logsnr_t - logsnr_s), z_t)
        alpha_t = expand_right_as(keras.ops.sqrt(keras.ops.sigmoid(logsnr_t)), z_t)
        alpha_s = expand_right_as(keras.ops.sqrt(keras.ops.sigmoid(logsnr_s)), z_t)
        sigma_t = expand_right_as(keras.ops.sqrt(keras.ops.sigmoid(-logsnr_t)), z_t)
        sigma_s = expand_right_as(keras.ops.sqrt(keras.ops.sigmoid(-logsnr_s)), z_t)

        if self.pred_param == 'v':
            x_pred = alpha_t * z_t - sigma_t * pred
        elif self.pred_param == 'eps':
            x_pred = (z_t - sigma_t * pred) / alpha_t
        else:
            raise NotImplementedError(f'Prediction parameterization {self.pred_param} not implemented')

        x_pred = self.clip(x_pred)

        mu = alpha_s * (z_t * (keras.ops.ones_like(c) - c) / alpha_t + c * x_pred)
        variance = keras.ops.square(sigma_s) * c

        return mu, variance

    def sd_loss(self, x: Tensor, c: Tensor = None, training=False) -> Tensor:
        """
        A function to compute the loss of the model. The loss is computed as the mean squared error between the
        predicted loss target tensor and the true loss target tensor. Target loss tensors can either be "v" or "eps"
        and are computed from the network predicted tensors which can also be "v" or "eps".

        Args:
            x (Tensor): The input tensor.
            c (Tensor): The condition tensor.
            training (bool): Whether the model is in training mode.

        Returns:
            loss (Tensor): The loss value.
        """
        if self.train_time == 'continuous':
            t = keras.random.uniform((keras.ops.shape(x)[0],))
        elif self.train_time == 'discrete':
            i = keras.random.randint((keras.ops.shape(x)[0],), minval=0, maxval=self.timesteps)
            t = keras.ops.cast(i, keras.ops.dtype(x)) / keras.ops.cast(self.timesteps, keras.ops.dtype(x))
        else:
            raise NotImplementedError(f"Training time {self.train_time} not implemented")

        logsnr_t = expand_right_as(self.get_logsnr(t), x)
        alpha_t = keras.ops.sqrt(keras.ops.sigmoid(logsnr_t))
        sigma_t = keras.ops.sqrt(keras.ops.sigmoid(-logsnr_t))

        z_t, eps_t = self.diffuse(x, alpha_t, sigma_t)
        v_t = alpha_t * eps_t - sigma_t * x
        gt_loss_param = v_t if self.loss_param == 'v' else eps_t
        if c is not None:
            z_t_c = keras.ops.concatenate([z_t, c, logsnr_t], axis=-1)
            #t = expand_right_as(t, z_t)
            #z_t_c = keras.ops.concatenate([z_t, c, t], axis=-1)
        else:
            z_t_c = keras.ops.concatenate([z_t, logsnr_t], axis=-1)
            #t = expand_right_as(t, z_t)
            #z_t_c = keras.ops.concatenate([z_t, t], axis=-1)
        pred = self.diffusion_backbone(z_t_c, training=training)
        if self.requires_projection:
            pred = self.projector(pred, training=training)
        # Simple Diffusion setting for UViT training: v -> L(v)
        if (self.pred_param == 'v' and self.loss_param == 'v') or (
                self.pred_param == 'eps' and self.loss_param == 'eps'):
            pred_loss_param = pred
        # Simple Diffusion setting for UNet training: v -> L(eps)
        elif self.pred_param == 'v' and self.loss_param == 'eps':
            pred_loss_param = alpha_t * pred + sigma_t * z_t
        elif self.pred_param == 'eps' and self.loss_param == 'v':
            pred_loss_param = (pred - sigma_t * z_t) / alpha_t
        else:
            raise NotImplementedError(
                f'Prediction parameterization {self.pred_param} and loss parameterization {self.loss_param} not implemented')

        loss = self.loss(pred_loss_param, gt_loss_param)

        # Apply min-SNR weighting (https://arxiv.org/pdf/2303.09556)
        if self.min_snr_weighting:
            snr = keras.ops.exp(logsnr_t)
            weight_t = keras.ops.clip(snr, x_min=0.0, x_max=5.0)  # min(SNR, gamma=5)
            if self.loss_param == 'v':
                loss *= weight_t / (1 + snr)
            elif self.loss_param == 'eps':
                loss *= weight_t / snr
            else:
                raise NotImplementedError(f'Prediction parameterization {self.pred_param} not implemented')
        loss = keras.ops.mean(loss)
        return loss


@keras.saving.register_keras_serializable(package="bayesflow.networks")
class DenoisingDiffusion(InferenceNetwork):
    # https://keras.io/examples/generative/ddpm/#training
    MLP_DEFAULT_CONFIG = {
        "widths": (256, 256, 256, 256, 256),
        "activation": "mish",
        "kernel_initializer": "he_normal",
        "residual": True,
        "dropout": 0.05,
        "spectral_normalization": False,
    }

    def __init__(
            self,
            subnet: str | type = "mlp",
            base_distribution: str = "normal",
            beta_start=1e-4,
            beta_end=0.02,
            timesteps=1000,
            clip_min=-5.0,
            clip_max=5.0,
            **kwargs,
    ):
        super().__init__(base_distribution=base_distribution, **keras_kwargs(kwargs))
        self.seed_generator = keras.random.SeedGenerator()
        self.subnet = subnet
        if subnet == "mlp":
            subnet_kwargs = SimpleDiffusion.MLP_DEFAULT_CONFIG.copy()
            subnet_kwargs.update(kwargs.get("subnet_kwargs", {}))
        else:
            subnet_kwargs = kwargs.get("subnet_kwargs", {})
        self.diffusion_backbone = find_network(subnet, **subnet_kwargs)
        self.requires_projection = False
        if subnet == "mlp":
            self.projector = keras.layers.Dense(
                units=None,
                bias_initializer="zeros",
                name="projector"
            )
            self.requires_projection = True
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Define the linear variance schedule
        self.betas = betas = keras.ops.linspace(
            beta_start,
            beta_end,
            timesteps,
            dtype='float64',  # Using float64 for better precision
        )
        self.num_timesteps = int(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = keras.ops.cumprod(alphas, axis=0)
        alphas_cumprod_prev = keras.ops.append(1.0, alphas_cumprod[:-1])

        self.betas = keras.ops.cast(betas, dtype='float32')
        self.alphas_cumprod = keras.ops.cast(alphas_cumprod, dtype='float32')
        self.alphas_cumprod_prev = keras.ops.cast(alphas_cumprod_prev, dtype='float32')

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = keras.ops.cast(
            keras.ops.sqrt(alphas_cumprod), dtype='float32'
        )

        self.sqrt_one_minus_alphas_cumprod = keras.ops.cast(
            keras.ops.sqrt(1.0 - alphas_cumprod), dtype='float32'
        )

        self.log_one_minus_alphas_cumprod = keras.ops.cast(
            keras.ops.log(1.0 - alphas_cumprod), dtype='float32'
        )

        self.sqrt_recip_alphas_cumprod = keras.ops.cast(
            keras.ops.sqrt(1.0 / alphas_cumprod), dtype='float32'
        )
        self.sqrt_recipm1_alphas_cumprod = keras.ops.cast(
            keras.ops.sqrt(1.0 / alphas_cumprod - 1), dtype='float32'
        )

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = keras.ops.cast(posterior_variance, dtype='float32')

        # Log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = keras.ops.cast(
            keras.ops.log(keras.ops.maximum(posterior_variance, 1e-20)), dtype='float32'
        )

        self.posterior_mean_coef1 = keras.ops.cast(
            betas * keras.ops.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            dtype='float32',
        )

        self.posterior_mean_coef2 = keras.ops.cast(
            (1.0 - alphas_cumprod_prev) * keras.ops.sqrt(alphas) / (1.0 - alphas_cumprod),
            dtype='float32',
        )

        self.loss = keras.losses.get('mse')
        self.loss = keras.losses.MeanSquaredError("mean")
        # serialization: store all parameters necessary to call __init__
        self.config = {
            "base_distribution": base_distribution,
            **kwargs,
        }
        self.config = serialize_value_or_type(self.config, "subnet", subnet)

    def get_config(self):
        base_config = super().get_config()
        return base_config | self.config

    @classmethod
    def from_config(cls, config):
        config = deserialize_value_or_type(config, "subnet")
        return cls(**config)

    def build(self, xz_shape: Shape, conditions_shape: Shape = None) -> None:
        super().build(xz_shape)
        input_shape = list(xz_shape)
        input_shape[-1] += 1
        if conditions_shape is not None:
            input_shape[-1] += conditions_shape[-1]
        input_shape = tuple(input_shape)
        self.diffusion_backbone.build(input_shape)
        input_shape = self.diffusion_backbone.compute_output_shape(input_shape)
        if self.subnet == 'mlp':
            self.projector.units = xz_shape[-1]
            self.projector.build(input_shape)

    def _forward(self, x: Tensor, conditions: Tensor = None, training: bool = False, **kwargs) -> Tensor:
        raise NotImplementedError("DenoisingDiffusion._forward not implemented")

    def _inverse(self, z: Tensor, conditions: Tensor = None, training: bool = False, **kwargs) -> Tensor:
        # 1. Randomly sample noise (starting point for reverse process)
        samples = z
        batch_size = keras.ops.shape(z)[0]
        bs = keras.ops.shape(z)[1]
        for t in tqdm(reversed(range(0, self.timesteps)), desc='Diffusion sampling', total=self.timesteps):
            tt = keras.ops.cast(keras.ops.full([batch_size, bs], t), dtype=keras.ops.dtype(z))
            tt = expand_right_as(tt, samples) / self.timesteps
            if conditions is not None:
                z_t_c = keras.ops.concatenate([samples, conditions, tt], axis=-1)
            else:
                z_t_c = keras.ops.concatenate([samples, tt], axis=-1)
            pred_noise = self.diffusion_backbone(z_t_c, training=False)
            if self.requires_projection:
                pred_noise = self.projector(pred_noise, training=False)
            samples = self.p_sample(pred_noise, samples, t, clip_denoised=True)
        return samples

    def compute_metrics(self, x: Tensor, conditions: Tensor = None, stage: str = "training", sample_weight=None) -> dict[str, Tensor]:
        base_metrics = super().compute_metrics(x, conditions=conditions, stage=stage)
        loss = self.sd_loss(x, conditions, training=stage == "training")
        return base_metrics | {'loss': loss}

    def sd_loss(self, x: Tensor, c: Tensor = None, training=False) -> Tensor:
        i = keras.random.randint((keras.ops.shape(x)[0],), minval=0, maxval=self.timesteps)
        noise = keras.random.normal(shape=keras.ops.shape(x), dtype=keras.ops.dtype(x))
        z_t = self.q_sample(x, i, noise)
        t = expand_right_as(i, z_t) / self.timesteps
        if c is not None:
            z_t_c = keras.ops.concatenate([z_t, c, t], axis=-1)
        else:
            z_t_c = keras.ops.concatenate([z_t, t], axis=-1)
        pred_noise = self.diffusion_backbone(z_t_c, training=True)
        if self.requires_projection:
            pred_noise = self.projector(pred_noise, training=training)
        loss = self.loss(noise, pred_noise)
        return loss

    def q_sample(self, x_start, t, noise):
        """Diffuse the data.

        Args:
            x_start: Initial sample (before the first diffusion step)
            t: Current timestep
            noise: Gaussian noise to be added at the current timestep
        Returns:
            Diffused samples at timestep `t`
        """
        x_start_shape = keras.ops.shape(x_start)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, keras.ops.shape(x_start))
        sqrt_1m_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape)
        z_t = sqrt_alpha * x_start + sqrt_1m_alpha * noise
        return z_t

    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        """Sample from the diffusion model.

        Args:
            pred_noise: Noise predicted by the diffusion model
            x: Samples at a given timestep for which the noise was predicted
            t: Current timestep
            clip_denoised (bool): Whether to clip the predicted noise
                within the specified range or not.
        """
        model_mean, _, model_log_variance = self.p_mean_variance(pred_noise, x=x, t=t, clip_denoised=clip_denoised)
        noise = keras.random.normal(shape=x.shape, dtype=x.dtype)
        # No noise when t == 0
        nonzero_mask = 1 - keras.ops.cast(keras.ops.equal(t, 0), 'float32')
        nonzero_mask = expand_right_as(nonzero_mask, x)
        return model_mean + nonzero_mask * keras.ops.exp(0.5 * model_log_variance) * noise

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = keras.ops.clip(x_recon, self.clip_min, self.clip_max)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        x_t_shape = keras.ops.shape(x_t)
        sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape)
        sqrt_recipm1_alpha = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape)
        x_recon = sqrt_recip_alpha * x_t - sqrt_recipm1_alpha * noise
        return x_recon

    def q_posterior(self, x_start, x_t, t):
        """Compute the mean and variance of the diffusion
        posterior q(x_{t-1} | x_t, x_0).

        Args:
            x_start: Stating point(sample) for the posterior computation
            x_t: Sample at timestep `t`
            t: Current timestep
        Returns:
            Posterior mean and variance at current timestep
        """

        x_t_shape = keras.ops.shape(x_t)
        mean_coef_1 = self._extract(self.posterior_mean_coef1, t, x_t_shape)
        mean_coef_2 = self._extract(self.posterior_mean_coef2, t, x_t_shape)
        posterior_mean = mean_coef_1 * x_start + mean_coef_2 * x_t
        posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t_shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _extract(self, a, t, x_shape):
        """Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.

        Args:
            a: Tensor to extract from
            t: Timestep for which the coefficients are to be extracted
            x_shape: Shape of the current batched samples
        """
        batch_size = x_shape[0]
        out = keras.ops.take(a, t, axis=0)
        return expand_right_to(out, len(x_shape))
