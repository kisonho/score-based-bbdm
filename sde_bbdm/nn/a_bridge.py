import math, torch
from diffusion import DiffusionData
from diffusion.nn import FastSamplingDiffusionModule, LatentDiffusionModule
from typing import Optional, TypeVar, Union

Module = TypeVar('Module', bound=torch.nn.Module)
E = TypeVar('E', bound=Optional[torch.nn.Module])
D = TypeVar('D', bound=Optional[torch.nn.Module])


class ABridgeModule(LatentDiffusionModule[Module, E, D], FastSamplingDiffusionModule[Module]):
    """
    The A-Bridge BBDM with algorithm in sec. 1 and 2 offered by Prof. Wang.

    - Properties:
        - c_lambda: The lambda coefficient for the A-Bridge BBDM in `float`.
    """
    c_lambda: float
    """The lambda coefficient for the A-Bridge BBDM in `float`."""

    def __init__(self, diff_model: Module, time_steps: int, *, c_lambda: float = 2, encoder: E = None, decoder: D = None) -> None:
        """
        Initialize the A-Bridge BBDM module.

        - Parameters:
            - diff_model: The diffusion model in `torch.nn.Module`.
            - time_steps: The number of time steps in `int`.
            - c_lambda: The lambda coefficient for the A-Bridge BBDM in `float`.
            - encoder: The encoder model in `torch.nn.Module`.
            - decoder: The decoder model in `torch.nn.Module`.
        """
        super().__init__(diff_model, time_steps, encoder=encoder, decoder=decoder)
        self.c_lambda = c_lambda

    def forward(self, data: DiffusionData) -> torch.Tensor:
        data = DiffusionData(data.x, data.t)
        return super().forward(data)

    def forward_diffusion(self, data: torch.Tensor, t: Optional[torch.Tensor] = None, /, condition: Optional[torch.Tensor] = None) -> tuple[
        DiffusionData, torch.Tensor]:
        # enter latent space
        x_start = self.encode(data)
        assert condition is not None, 'Condition is required for forward diffusion.'
        condition = self.encode(condition)
        t = torch.randint(1, self.time_steps + 1, (x_start.shape[0],), device=data.device).long() if t is None else t.to(data.device)
        assert x_start.shape == condition.shape, f'X_start and condition must have the same shape, got x_start={data.shape} and condition{condition.shape}.'

        # step1 create t
        T = self.time_steps
        t_reshaped = t.view([x_start.shape[0]] + [1 for _ in range(len(x_start.shape[1:]))]) # expand shape of t to match x_start dimensions
        m_t = t_reshaped / self.time_steps
        # print(t_reshaped)
                # step2 create noise
        noise = torch.randn_like(x_start, device=x_start.device)
        B_t = self.c_lambda * (1 - m_t) * (torch.log(1 / (1 - m_t))) ** 0.5
        B_t = torch.where(torch.eq(t_reshaped, T), torch.zeros_like(B_t), B_t)
        # print(B_t)
        xt = (1 - m_t) * x_start + m_t * condition +  B_t * noise
        objective = m_t * (condition - x_start) + B_t * noise
        return DiffusionData(xt, t), objective

    def sampling_step(self, data: DiffusionData, i: int, /, *, return_noise: bool = False, predicted_obj: Optional[torch.Tensor] = None) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # check if fast sampling
        if self.fast_sampling_steps is not None:
            # get time steps
            tau = self.fast_sampling_steps[i]
            tau_minus_one = self.fast_sampling_steps[i + 1] if i < len(self.fast_sampling_steps) - 1 else 0

            # fast sampling step
            t = torch.full(tuple([data.x.shape[0]] + [1 for _ in range(len(data.x.shape[1:]))]), tau, device=data.x.device)
            data = DiffusionData(data.x, t, condition=data.condition)
            return self.fast_sampling_step(data, tau, tau_minus_one, return_noise=return_noise, predicted_obj=predicted_obj)

        # check condition
        assert data.condition is not None, "Condition must be given for A-Bridge."

        # m_t = t/T
        t = i
        T = self.time_steps
        m_t = t / T
        m_t = torch.full(tuple([data.x.shape[0]] + [1 for _ in range(len(data.x.shape[1:]))]), m_t, device=data.x.device)

        # replace random noise into condition for the first sampling step
        if t == T:
            x_t = data.condition
            data = DiffusionData(x_t, data.t, condition=data.condition)

        # predict noise
        if predicted_obj is None: 
            predicted_obj = self.forward(data)
            assert predicted_obj is not None, "Predicted noise must be given."

        # initialize new noise
        new_noise = torch.randn_like(data.x, device=data.x.device) if t > 2 else 0

        # sampling equation
        assert data.condition is not None, "Condition must be given."
        if i == 1:
            x_t_minus_one: torch.Tensor = data.x - predicted_obj
        elif i == self.time_steps:
            noise = torch.randn_like(data.x, device=data.x.device, dtype=data.x.dtype)
            x_t_minus_one: torch.Tensor = 0.9998552 * data.x + 0.0001447648 * (data.x - predicted_obj) - 0.0014142 * noise
        else:
            beta_t = T - t + 1
            gamma_t = math.log(T / beta_t)
            C = 1 - 1 / beta_t + 1 / (beta_t * gamma_t)
            c_xt = 1 / C * (1 + 1 / (T * gamma_t))
            c_yt = 1 / C * (1 / beta_t - (t - 1) / (T * beta_t * gamma_t))
            c_epst = 1 / C * (1 / (T * gamma_t))
            c_zt = self.c_lambda / C * (1 - m_t + 1 / T) ** 0.5 * (1 / T) ** 0.5
            x_t_minus_one: torch.Tensor = c_xt * data.x - c_yt * data.condition - c_epst * predicted_obj - c_zt * new_noise
        return (x_t_minus_one, predicted_obj) if return_noise else x_t_minus_one

    def fast_sampling_step(self, data: DiffusionData, tau: int, tau_minus_one: int, /, *, return_noise: bool = False, predicted_obj: Optional[torch.Tensor] = None) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # check condition
        assert data.condition is not None, "Condition must be given for A-Bridge."
        T = self.time_steps

        # replace random noise into condition for the first sampling step
        if tau == self.time_steps:
            x_t = data.condition
            data = DiffusionData(x_t, data.t, condition=data.condition)

        # predict noise
        if predicted_obj is None: 
            predicted_obj, _ = self.forward(data)
            assert predicted_obj is not None, "Predicted noise must be given."

        # unpack data
        assert data.condition is not None, "Condition must be given."
        x_tau = data.x
        y = data.condition

        # if this is the last step
        if tau_minus_one == 0:
            x_tau_minus_one: torch.Tensor = x_tau - predicted_obj
            return (x_tau_minus_one, predicted_obj) if return_noise else x_tau_minus_one

        # m_t = t/T
        m_tau = tau / T
        m_tau = torch.full([x_tau.shape[0],] + [1 for _ in range(len(data.x.shape[1:]))], m_tau, device=x_tau.device)

        # m_{t-1} = (t-1)/T
        m_tau_minus_one = tau_minus_one / T
        m_tau_minus_one = torch.full([x_tau.shape[0],] + [1 for _ in range(len(data.x.shape[1:]))], m_tau_minus_one, device=data.x.device)

        # y_{t-1} and y_t
        y_tau_minus_one = (1 - m_tau_minus_one) * (x_tau - predicted_obj) + m_tau_minus_one * y
        y_tau = (1 - m_tau) * (x_tau - predicted_obj) + m_tau * y

        # initialize new noise
        new_noise = torch.randn_like(x_tau, device=x_tau.device)

        # sampling equation
        B_tau_minus_one = self.c_lambda * (1 - m_tau_minus_one) * (torch.log(1 / (1 - m_tau_minus_one))) ** 0.5
        B_tau = self.c_lambda * (1 - m_tau) * (torch.log(1 / (1 - m_tau))) ** 0.5 if tau != T else 1
        sigma_tau = 0.5 * B_tau_minus_one
        x_tau_minus_one = y_tau_minus_one + (torch.sqrt(B_tau_minus_one ** 2 - sigma_tau ** 2) / B_tau) * (x_tau - y_tau) + sigma_tau * new_noise
        return (x_tau_minus_one, predicted_obj) if return_noise else x_tau_minus_one
