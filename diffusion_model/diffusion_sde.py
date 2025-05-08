import torch

from diffusion_model.helper_functions import sech


class SDE:
    def __init__(self, kernel_type='variance_preserving', noise_schedule='cosine',
                 log_snr_min=-15, log_snr_max=15, s_shift_cosine=0.):
        if kernel_type not in ['variance_preserving', 'sub_variance_preserving']:
            raise ValueError("Invalid kernel type.")
        if noise_schedule not in ['linear', 'cosine', 'flow_matching', 'edm-training', 'edm-sampling']:
            raise ValueError("Invalid noise schedule.")

        if noise_schedule == 'flow_matching' and kernel_type == 'variance_preserving':
            raise ValueError("Variance-preserving kernel is not compatible with 'flow_matching' noise schedule.")

        if noise_schedule == 'edm-training':
            print('Remember to switch the noise schedule for sampling!')

        self.kernel_type = kernel_type
        self.noise_schedule = noise_schedule
        self.s_shift_cosine = s_shift_cosine

        # needed for schedule
        self._log_snr_min = log_snr_min
        self._log_snr_max = log_snr_max

        # for edm: lambda \in [− log σ^2_max, − log σ^2_min]
        self.sigma_max = torch.tensor(80)
        self.sigma_min = torch.tensor(0.002)
        self.p_mean = -1.2
        self.p_std = 1.2
        self.rho = 7
        if noise_schedule in ['edm-training', 'edm-sampling']:
            self._log_snr_min = -2 * torch.log(self.sigma_max)
            self._log_snr_max = -2 * torch.log(self.sigma_min)
            self.t_min = self.get_t_from_snr(self._log_snr_max)
            self.t_max = self.get_t_from_snr(self._log_snr_min)
            print(f"Using EDM noise schedule with log_snr_min: {self._log_snr_min}, log_snr_max: {self._log_snr_max}")
        else:
            self.t_min = self.get_t_from_snr(torch.tensor(self._log_snr_max))
            self.t_max = self.get_t_from_snr(torch.tensor(self._log_snr_min))

        print(f"Kernel type: {self.kernel_type}, noise schedule: {self.noise_schedule}")
        print(f"t_min: {self.t_min}, t_max: {self.t_max}")
        print('alpha, sigma:',
              self.kernel(log_snr=self.get_snr(t=0)),
              self.kernel(log_snr=self.get_snr(t=1)))

    @property
    def log_snr_min(self):
        return self._log_snr_min

    @log_snr_min.setter
    def log_snr_min(self, value):
        self._log_snr_min = value
        # Update t_max because it depends on log_snr_min.
        self.t_max = self.get_t_from_snr(torch.tensor(self._log_snr_min))

    @property
    def log_snr_max(self):
        return self._log_snr_max

    @log_snr_max.setter
    def log_snr_max(self, value):
        self._log_snr_max = value
        # Update t_min because it depends on log_snr_max.
        self.t_min = self.get_t_from_snr(torch.tensor(self._log_snr_max))

    def kernel(self, log_snr):
        if self.kernel_type == 'variance_preserving':
            return self._variance_preserving_kernel(log_snr=log_snr)
        if self.kernel_type == 'sub_variance_preserving':
            return self._sub_variance_preserving_kernel(log_snr=log_snr)
        raise ValueError("Invalid kernel type.")

    def grad_log_kernel(self, x, x0, t):  # t is not truncated
        if self.kernel_type == 'variance_preserving':
            return self._grad_log_variance_preserving_kernel(x, x0, t)
        if self.kernel_type == 'sub_variance_preserving':
            return self._grad_log_sub_variance_preserving_kernel(x, x0, t)
        raise ValueError("Invalid kernel type.")

    def get_snr(self, t):  # t is not truncated
        t_trunc = self.t_min + (self.t_max - self.t_min) * t
        if self.noise_schedule == 'linear':
            return -torch.log(torch.exp(torch.square(t_trunc)) - 1)
        if self.noise_schedule == 'cosine':  # this is usually used with variance_preserving
            return -2 * torch.log(torch.tan(torch.pi * t_trunc / 2)) + 2 * self.s_shift_cosine
        if self.noise_schedule == 'flow_matching':  # this usually used with sub_variance_preserving
            return 2 * torch.log((1 - t_trunc) / t_trunc)
        if self.noise_schedule == 'edm-training':
            # training
            dist = torch.distributions.Normal(loc=-2*self.p_mean, scale=2*self.p_std)
            snr = dist.icdf(t_trunc)
            snr = snr.clamp(min=self._log_snr_min.to(snr.device), max=self._log_snr_max.to(snr.device))
            return snr
        if self.noise_schedule == 'edm-sampling':
            # sampling
            snr = -2 * self.rho * torch.log(self.sigma_max ** (1/self.rho) + (1 - t_trunc) * (self.sigma_min ** (1/self.rho) - self.sigma_max ** (1/self.rho)))
            return snr
        raise ValueError("Invalid 'noise_schedule'.")

    def get_t_from_snr(self, snr):  # not truncated
        # Invert the noise scheduling to recover t
        if self.noise_schedule == 'linear':
            # SNR = -log(exp(t^2) - 1)
            # => t = sqrt(log(1 + exp(-snr)))
            return torch.sqrt(torch.log(1 + torch.exp(-snr)))
        if self.noise_schedule == 'cosine':
            # SNR = -2 * log(tan(pi*t/2))
            # => t = 2/pi * arctan(exp(-snr/2))
            return 2 / torch.pi * torch.atan(torch.exp((2 * self.s_shift_cosine - snr) / 2))
        if self.noise_schedule == 'flow_matching':
            # SNR = 2 * log((1-t)/t)
            # => t = 1 / (1 + exp(snr/2))
            return 1 / (1 + torch.exp(snr / 2))
        if self.noise_schedule == 'edm-training':
            # SNR = dist.icdf(t_trunc)
            # => t = dist.cdf(snr)
            dist = torch.distributions.Normal(loc=-2*self.p_mean, scale=2*self.p_std)
            return dist.cdf(snr)
        if self.noise_schedule == 'edm-sampling':
            # SNR = -2 * rho * log(sigma_max ** (1/rho) + (1 - t) * (sigma_min ** (1/rho) - sigma_max ** (1/rho)))
            # => t = 1 - ((torch.exp(-snr/(2*rho)) - sigma_max ** (1/rho)) / (sigma_min ** (1/rho) - sigma_max ** (1/rho)))
            return 1 - ((torch.exp(-snr/(2*self.rho)) - self.sigma_max ** (1/self.rho)) / (self.sigma_min ** (1/self.rho) - self.sigma_max ** (1/self.rho)))
        raise ValueError("Invalid 'noise_schedule'.")

    def _get_snr_derivative(self, t):  # t is not truncated
        """
        Compute d/dt log(1 + e^(-snr(t))) for the truncated schedules.
        """
        # Compute the truncated time t_trunc
        t_trunc = self.t_min + (self.t_max - self.t_min) * t
        # Compute snr(t) using the existing function
        snr = self.get_snr(t=t)

        # Compute d/dx snr(x) based on the noise schedule
        if self.noise_schedule == 'linear':
            # d/dx snr(x) = - 2*x*exp(x^2) / (exp(x^2) - 1)
            dsnr_dx = - (2 * t_trunc * torch.exp(t_trunc**2)) / (torch.exp(t_trunc**2) - 1)
        elif self.noise_schedule == 'cosine':
            # d/dx snr(x) = -2*pi/sin(pi*x)
            dsnr_dx = - (2 * torch.pi) / torch.sin(torch.pi * t_trunc)
        elif self.noise_schedule == 'flow_matching':
            # d/dx snr(x) = -2/(x*(1-x))
            dsnr_dx = - 2 / (t_trunc * (1 - t_trunc))
        elif self.noise_schedule == 'edm-training':
            raise ValueError("EDM-training noise schedule should be used for training only.")
        elif self.noise_schedule == 'edm-sampling':
            # SNR = -2*rho*log(s_max + (1 - x)*(s_min - s_max))
            s_max = self.sigma_max ** (1 / self.rho)
            s_min = self.sigma_min ** (1 / self.rho)
            u = s_max + (1 - t_trunc) * (s_min - s_max)
            # d/dx snr = 2*rho*(s_min - s_max) / u
            dsnr_dx = 2 * self.rho * (s_min - s_max) / u
        else:
            raise ValueError("Invalid 'noise_schedule'.")

        # Chain rule: d/dt snr(t) = d/dx snr(x) * (t_max - t_min)
        dsnr_dt = dsnr_dx * (self.t_max - self.t_min)

        # f(t) = log(1 + e^{-snr(t)})  =>  f'(t) = - (e^{-snr}/(1+e^{-snr})) * dsnr_dt
        factor = torch.exp(-snr) / (1 + torch.exp(-snr))
        return -factor * dsnr_dt

    def _beta_integral(self, t):  # t is not truncated
        """Song et al. (2021) defined this as integral over the beta, here we express is equivalently via the snr"""
        return torch.log(1 + torch.exp(-self.get_snr(t)))

    @staticmethod
    def _variance_preserving_kernel(log_snr):
        """
        Computes the variance-preserving kernel p(x_t | x_0) for the diffusion process.

        Args:
            log_snr (torch.Tensor): The log of the signal-to-noise ratio.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The mean and standard deviation of the kernel at time t.
        """
        alpha_t = torch.sqrt(torch.sigmoid(log_snr))
        sigma_t = torch.sqrt(torch.sigmoid(-log_snr))
        return alpha_t, sigma_t

    def _grad_log_variance_preserving_kernel(self, x, x0, t):  # t is not truncated
        """
        Computes the gradient of the log probability density of the variance-preserving kernel.

        Given the kernel p(x_t|x_0) = N(x_t; mean = x0 * exp(-0.5 * I), variance = 1 - exp(-I))
        where I = ∫₀ᵗ beta(s) ds, the gradient is:

            ∇ₓ log p(x_t|x_0) = -(x_t - x0 * exp(-0.5 * I)) / (1 - exp(-I))

        Args:
            x (torch.Tensor): The current state x_t.
            x0 (torch.Tensor): The initial state.
            t (torch.Tensor): The time parameter.

        Returns:
            torch.Tensor: The gradient ∇ₓ log p(x_t|x0).
        """
        if self.noise_schedule == 'flow_matching':
            raise ValueError("Variance-preserving kernel is not compatible with 'flow_matching' noise schedule.")
        integral = self._beta_integral(t)
        mean_factor = torch.exp(-0.5 * integral)
        variance = 1 - torch.exp(-integral)         # variance for the VP kernel
        grad = -(x - mean_factor * x0) / variance
        return grad

    def _sub_variance_preserving_kernel(self, log_snr):
        """
        Computes the sub-variance-preserving kernel p(x_t | x_0) for the diffusion process.
        Args:
            t (torch.Tensor): The time at which to evaluate the kernel in [0,1]. Should be not too close to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The mean and standard deviation of the kernel at time t.
        """
        if self.noise_schedule == 'flow_matching':
            t = self.get_t_from_snr(snr=log_snr)
            x = self.t_min + (self.t_max - self.t_min) * t
            alpha_t = 1 - x
            sigma_t = x
        else:
            alpha_t = 1 / (1 + torch.exp(-log_snr))
            sigma_t = 1  - torch.square(alpha_t)
        return alpha_t, sigma_t

    def _grad_log_sub_variance_preserving_kernel(self, x, x0, t):  # t is not truncated
        """
        Computes the gradient of the log probability density of the sub-variance-preserving kernel.

        For this kernel, we use the convention that:
            mean = x0 * exp(-0.5 * I)
            "std" = 1 - exp(-I)  -> variance = (1 - exp(-I))^2
        so that:

            ∇ₓ log p(x_t|x_0) = -(x_t - x0 * exp(-0.5 * I)) / ((1 - exp(-I))^2)

        Args:
            x (torch.Tensor): The current state x_t.
            x0 (torch.Tensor): The initial state.
            t (torch.Tensor): The time parameter.

        Returns:
            torch.Tensor: The gradient ∇ₓ log p(x_t|x0).
        """
        integral = self._beta_integral(t)
        mean_factor = torch.exp(-0.5 * integral)
        variance = (1 - torch.exp(-integral))**2  # variance for the sub-VP kernel
        grad = -(x - mean_factor * x0) / variance
        return grad

    def get_f_g(self, t, x=None):  # t is not truncated
        if self.kernel_type == 'variance_preserving':
            # Compute beta(t)
            beta_t = self._get_snr_derivative(t)

            # Compute drift f(x, t) = -0.5 * beta(t) * x
            if x is not None:
                f = -0.5 * beta_t * x

            # Compute diffusion coefficient g(t) = sqrt(beta(t))
            g = torch.sqrt(beta_t)
        elif self.kernel_type == 'sub_variance_preserving':
            # Compute beta(t)
            beta_t = self._get_snr_derivative(t)

            # Compute drift f(x, t) = -0.5 * beta(t) * x
            if x is not None:
                f = -0.5 * beta_t * x

            # Compute diffusion coefficient g(t) = sqrt(beta(t) * (1 - exp(-2 \int_0^t beta(s) ds)))
            g = torch.sqrt(beta_t * (1 - torch.exp(-2 * self._beta_integral(t))))
        else:
            raise ValueError("Invalid kernel type.")
        if x is None:
            return g
        return f, g


def weighting_function(t, sde, weighting_type, prediction_type='error'):

    if prediction_type == 'score':
        # likelihood weighting, since beta(t) = g(t)^2
        g_t = sde.get_f_g(t=t)
        return torch.square(g_t)

    if weighting_type is None:
        return torch.ones_like(t)

    if weighting_type == 'likelihood_weighting':
        g_t = sde.get_f_g(t=t)
        log_snr = sde.get_snr(t=t)
        sigma_t = sde.kernel(log_snr=log_snr)[1]  # divide by sigma^2, since the loss is on the score
        return torch.square(g_t / sigma_t)

    log_snr = sde.get_snr(t=t)
    if weighting_type == 'flow_matching':
        alpha = sde.kernel(log_snr=log_snr)[0]
        if sde.noise_schedule == 'flow_matching':
            return torch.exp(-log_snr / 2)
        elif sde.noise_schedule == 'cosine':
            return torch.square(alpha * (torch.exp(-log_snr) +1)) / (2 * torch.pi * torch.cosh(-log_snr/2))  # flow matching equivalent
        else:
            raise NotImplementedError("Flow matching not implemented for this noise schedule.")

    if weighting_type == 'sigmoid':
        return torch.sigmoid(-log_snr + 2)

    if weighting_type == 'min-snr':
        gamma = 5
        return sech(log_snr / 2) * torch.minimum(torch.ones_like(log_snr), gamma * torch.exp(-log_snr))

    if weighting_type == 'edm':
        return torch.exp(-log_snr) + 1
        #sigma_data = 1.
        #return torch.exp(-log_snr) / torch.square(sigma_data) + 1

    raise ValueError("Invalid weighting...")
