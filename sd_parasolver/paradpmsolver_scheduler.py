# Standard library imports
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import torch
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_utils import SchedulerOutput
from diffusers.utils import deprecate
from diffusers.utils.torch_utils import randn_tensor

# Type checking imports
if TYPE_CHECKING:
    import torch.jit
    import torch.jit._state

class ParaDPMSolverMultistepScheduler(DPMSolverMultistepScheduler):
    # careful when overriding __init__ function, can break things due to expected_keys parameter in configuration_utils
    # if necessary copy the whole init statement from parent class
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        solver_order: int = 2,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        algorithm_type: str = "dpmsolver++",
        solver_type: str = "midpoint",
        lower_order_final: bool = True,
        euler_at_final: bool = False,
        use_karras_sigmas: Optional[bool] = False,
        use_lu_lambdas: Optional[bool] = False,
        final_sigmas_type: Optional[str] = "zero",  # "zero", "sigma_min"
        lambda_min_clipped: float = -float("inf"),
        variance_type: Optional[str] = None,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        rescale_betas_zero_snr: bool = False,
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            trained_betas=trained_betas,
            solver_order=solver_order,
            prediction_type=prediction_type,
            thresholding=thresholding,
            dynamic_thresholding_ratio=dynamic_thresholding_ratio,
            sample_max_value=sample_max_value,
            algorithm_type=algorithm_type,
            solver_type=solver_type,
            lower_order_final=lower_order_final,
            euler_at_final=euler_at_final,
            use_karras_sigmas=use_karras_sigmas,
            use_lu_lambdas=use_lu_lambdas,
            final_sigmas_type=final_sigmas_type,
            lambda_min_clipped=lambda_min_clipped,
            variance_type=variance_type,
            timestep_spacing=timestep_spacing,
            steps_offset=steps_offset,
            rescale_betas_zero_snr=rescale_betas_zero_snr,
        )
        if algorithm_type in ["dpmsolver", "sde-dpmsolver"]:
            deprecation_message = f"algorithm_type {algorithm_type} is deprecated and will be removed in a future version. Choose from `dpmsolver++` or `sde-dpmsolver++` instead"
            deprecate("algorithm_types dpmsolver and sde-dpmsolver", "1.0.0", deprecation_message)

        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        if rescale_betas_zero_snr:
            # Close to 0 without being 0 so first sigma is not inf
            # FP16 smallest positive subnormal works well here
            self.alphas_cumprod[-1] = 2**-24

        # Currently we only support VP-type noise schedule
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # settings for DPM-Solver
        if algorithm_type not in ["dpmsolver", "dpmsolver++", "sde-dpmsolver", "sde-dpmsolver++"]:
            if algorithm_type == "deis":
                self.register_to_config(algorithm_type="dpmsolver++")
            else:
                raise NotImplementedError(f"{algorithm_type} does is not implemented for {self.__class__}")

        if solver_type not in ["midpoint", "heun"]:
            if solver_type in ["logrho", "bh1", "bh2"]:
                self.register_to_config(solver_type="midpoint")
            else:
                raise NotImplementedError(f"{solver_type} does is not implemented for {self.__class__}")

        if algorithm_type not in ["dpmsolver++", "sde-dpmsolver++"] and final_sigmas_type == "zero":
            raise ValueError(
                f"`final_sigmas_type` {final_sigmas_type} is not supported for `algorithm_type` {algorithm_type}. Please choose `sigma_min` instead."
            )

        # setable values
        self.num_inference_steps = None
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)
        self.model_outputs = [None] * solver_order
        self.lower_order_nums = 0
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

        #added
        self.base_timesteps =  None
        self.base_sigmas = None

        self.coarse_timesteps = None
        self.coarse_timesteps_sigmas = None

        self.initial_timesteps = None
        self.initial_timesteps_sigmas = None

        self.fine_timesteps_matrix = None
        self.max_fine_timestep_num = None
        self.interval_len = None
        self.coarse_timestep_num = None
        self.fine_timesteps_sigmas_matrix = None

    def dpm_solver_first_order_update(
        self,
        model_output: torch.FloatTensor,
        *args,
        sample: torch.FloatTensor = None,
        noise: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        One step for the first-order DPMSolver (equivalent to DDIM).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from the learned diffusion model.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.FloatTensor`:
                The sample tensor at the previous timestep.
        """
        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
        prev_timestep = args[1] if len(args) > 1 else kwargs.pop("prev_timestep", None)
        if sample is None:
            if len(args) > 2:
                sample = args[2]
            else:
                raise ValueError(" missing `sample` as a required keyward argument")
        if timestep is not None:
            deprecate(
                "timesteps",
                "1.0.0",
                "Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        if prev_timestep is not None:
            deprecate(
                "prev_timestep",
                "1.0.0",
                "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)

        h = lambda_t - lambda_s
        if self.config.algorithm_type == "dpmsolver++":
            if alpha_t.numel() >1:
                a = (sigma_t / sigma_s).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                b = (alpha_t * (torch.exp(-h) - 1.0)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            else:
                a = (sigma_t / sigma_s)
                b = (alpha_t * (torch.exp(-h) - 1.0))
            x_t = a * sample -  b * model_output
        elif self.config.algorithm_type == "dpmsolver":
            if alpha_t.numel() < 2:
                alpha = alpha_t / alpha_s
                d = sigma_t * (torch.exp(h) - 1.0)
            else:
                alpha = (alpha_t / alpha_s).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                d = (sigma_t * (torch.exp(h) - 1.0)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            x_t = alpha * sample - d * model_output
        elif self.config.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            if alpha_t.numel() >1:
                a = (sigma_t / sigma_s * torch.exp(-h)).unsqueeze(2).unsqueeze(3)
                b = (alpha_t * (1 - torch.exp(-2.0 * h))).unsqueeze(2).unsqueeze(3)
                c = sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            else:
                a = (sigma_t / sigma_s * torch.exp(-h))
                b = (alpha_t * (1 - torch.exp(-2.0 * h)))
                c = sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h))
            x_t = (
                a * sample
                + b * model_output
                + c * noise
            )
        elif self.config.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            if alpha_t.numel() < 2:
                alpha = alpha_t / alpha_s
                d = sigma_t * (torch.exp(h) - 1.0)
                n = sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0)
            else:
                alpha = (alpha_t / alpha_s).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                d = (sigma_t * (torch.exp(h) - 1.0)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                n = (sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            x_t = (
                alpha * sample - 2.0 * d * model_output + n * noise
            )
        return x_t
    def multistep_dpm_solver_second_order_update(
        self,
        model_output_list: List[torch.FloatTensor],
        *args,
        sample: torch.FloatTensor = None,
        noise: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        One step for the second-order multistep DPMSolver.

        Args:
            model_output_list (`List[torch.FloatTensor]`):
                The direct outputs from learned diffusion model at current and latter timesteps.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.FloatTensor`:
                The sample tensor at the previous timestep.
        """
        timestep_list = args[0] if len(args) > 0 else kwargs.pop("timestep_list", None)
        prev_timestep = args[1] if len(args) > 1 else kwargs.pop("prev_timestep", None)
        if sample is None:
            if len(args) > 2:
                sample = args[2]
            else:
                raise ValueError(" missing `sample` as a required keyward argument")
        if timestep_list is not None:
            deprecate(
                "timestep_list",
                "1.0.0",
                "Passing `timestep_list` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        if prev_timestep is not None:
            deprecate(
                "prev_timestep",
                "1.0.0",
                "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        sigma_t, sigma_s0, sigma_s1 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
        )

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

        m0, m1 = model_output_list[-1], model_output_list[-2]

        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1

        r0 = h_0 / h
        if h.numel() >1:
            a = (1.0 / r0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            #print(r0.size(),a.size(),m0.size(),m1.size())
        else:
            a = (1.0 / r0)
        D0, D1 = m0, a * (m0 - m1)

        if self.config.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2211.01095 for detailed derivations
            if self.config.solver_type == "midpoint":
                if h.numel() > 1:
                    a = (sigma_t / sigma_s0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    b = (alpha_t * (torch.exp(-h) - 1.0)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    c = 0.5 * (alpha_t * (torch.exp(-h) - 1.0)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    # d = h_0.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                else:
                    a = (sigma_t / sigma_s0)
                    b = (alpha_t * (torch.exp(-h) - 1.0))
                    c = 0.5 * (alpha_t * (torch.exp(-h) - 1.0))
                    # d = h_0.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                x_t = (
                    a * sample
                    - b * D0
                    - c * D1
                )
            elif self.config.solver_type == "heun":
                if h.numel() > 1:
                    a = (sigma_t / sigma_s0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    b = (alpha_t * (torch.exp(-h) - 1.0)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    c = (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    # d = h_0.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                else:
                    a = (sigma_t / sigma_s0)
                    b = (alpha_t * (torch.exp(-h) - 1.0))
                    c = (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0))
                x_t = (
                    a * sample
                    - b * D0
                    +  c * D1
                )
        elif self.config.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            if self.config.solver_type == "midpoint":
                if h.numel() > 1:
                    a = (sigma_t / sigma_s0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    b = (alpha_t * (torch.exp(-h) - 1.0)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    c = 0.5 * (alpha_t * (torch.exp(-h) - 1.0)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    # d = h_0.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                else:
                    a = (sigma_t / sigma_s0)
                    b = (alpha_t * (torch.exp(-h) - 1.0))
                    c = 0.5 * (alpha_t * (torch.exp(-h) - 1.0))
                    # d = h_0.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                x_t = (
                    a * sample
                    - b * D0
                    - c * D1
                )
            elif self.config.solver_type == "heun":
                if h.numel() > 1:
                    a = (alpha_t / alpha_s0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    b = (alpha_t * (torch.exp(-h) - 1.0)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    c = 0.5 * (alpha_t * (torch.exp(-h) - 1.0)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    # d = h_0.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                else:
                    a = (alpha_t / alpha_s0)
                    b = (alpha_t * (torch.exp(-h) - 1.0))
                    c = 0.5 * (alpha_t * (torch.exp(-h) - 1.0))
                    # d = h_0.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                x_t = (
                    a * sample
                    - b * D0
                    - c * D1
                )
        elif self.config.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            if self.config.solver_type == "midpoint":
                if h.numel() > 1:
                    a = (sigma_t / sigma_s0 * torch.exp(-h)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    b = (alpha_t * (1 - torch.exp(-2.0 * h))).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    c = 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    d = sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                else:
                    a = (sigma_t / sigma_s0 * torch.exp(-h))
                    b = (alpha_t * (1 - torch.exp(-2.0 * h)))
                    c = 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h)))
                    d = sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h))
                x_t = (
                    a * sample
                    + b * D0
                    + c * D1
                    + d * noise
                )
            elif self.config.solver_type == "heun":
                if h.numel() > 1:
                    a = (sigma_t / sigma_s0 * torch.exp(-h)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    b = (alpha_t * (1 - torch.exp(-2.0 * h))).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    c = (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    d = sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                else:
                    a = (sigma_t / sigma_s0 * torch.exp(-h))
                    b = (alpha_t * (1 - torch.exp(-2.0 * h)))
                    c = (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0))
                    d = sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h))
                x_t = (
                    a * sample
                    + b * D0
                    + c * D1
                    + d * noise
                )

        elif self.config.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            if self.config.solver_type == "midpoint":
                if h.numel() > 1:
                    a = (alpha_t / alpha_s0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    b = 2.0 * (sigma_t * (torch.exp(h) - 1.0)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    c = (sigma_t * (torch.exp(h) - 1.0)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    d = sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                else:
                    a = (alpha_t / alpha_s0)
                    b = 2.0 * (sigma_t * (torch.exp(h) - 1.0))
                    c = (sigma_t * (torch.exp(h) - 1.0))
                    d = sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0)
                x_t = (
                    a * sample
                    - b * D0
                    - c * D1
                    + d * noise
                )
            elif self.config.solver_type == "heun":
                if self.config.solver_type == "midpoint":
                    if h.numel() > 1:
                        a = (alpha_t / alpha_s0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                        b = 2.0 * (sigma_t * (torch.exp(h) - 1.0)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                        c = 2.0 * (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                        d = sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    else:
                        a = (alpha_t / alpha_s0)
                        b = 2.0 * (sigma_t * (torch.exp(h) - 1.0))
                        c = 2.0 * (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0))
                        d = sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0)
                    x_t = (
                            a * sample
                            - b * D0
                            - c * D1
                            + d * noise
                    )
        return x_t

    def convert_model_output_and_output_pred_x0(
        self,
        model_output: torch.FloatTensor,
        *args,
        sample: torch.FloatTensor = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Convert the model output to the corresponding type the DPMSolver/DPMSolver++ algorithm needs. DPM-Solver is
        designed to discretize an integral of the noise prediction model, and DPM-Solver++ is designed to discretize an
        integral of the data prediction model.

        <Tip>

        The algorithm and model type are decoupled. You can use either DPMSolver or DPMSolver++ for both noise
        prediction and data prediction models.

        </Tip>

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from the learned diffusion model.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `Tuple[torch.FloatTensor, torch.FloatTensor]`:
                A tuple containing (epsilon, x0_pred) where:
                - epsilon: the predicted noise
                - x0_pred: the predicted initial sample
        """
        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError("missing `sample` as a required keyward argument")
        if timestep is not None:
            deprecate(
                "timesteps",
                "1.0.0",
                "Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        # DPM-Solver++ needs to solve an integral of the data prediction model.
        if self.config.algorithm_type in ["dpmsolver++", "sde-dpmsolver++"]:
            if self.config.prediction_type == "epsilon":
                # DPM-Solver and DPM-Solver++ only need the "mean" output.
                if self.config.variance_type in ["learned", "learned_range"]:
                    model_output = model_output[:, :3]
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                if alpha_t.numel() > 1:
                    alpha_t = alpha_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    sigma_t = sigma_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                x0_pred = (sample - sigma_t * model_output) / alpha_t
                epsilon = model_output
            elif self.config.prediction_type == "sample":
                x0_pred = model_output
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                if alpha_t.numel() > 1:
                    alpha_t = alpha_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    sigma_t = sigma_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                epsilon = (sample - alpha_t * x0_pred) / sigma_t
            elif self.config.prediction_type == "v_prediction":
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                if alpha_t.numel() > 1:
                    alpha_t = alpha_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    sigma_t = sigma_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                x0_pred = alpha_t * sample - sigma_t * model_output
                epsilon = alpha_t * model_output + sigma_t * sample
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the DPMSolverMultistepScheduler."
                )

            if self.config.thresholding:
                x0_pred = self._threshold_sample(x0_pred)

            return epsilon, x0_pred

        # DPM-Solver needs to solve an integral of the noise prediction model.
        elif self.config.algorithm_type in ["dpmsolver", "sde-dpmsolver"]:
            if self.config.prediction_type == "epsilon":
                # DPM-Solver and DPM-Solver++ only need the "mean" output.
                if self.config.variance_type in ["learned", "learned_range"]:
                    epsilon = model_output[:, :3]
                else:
                    epsilon = model_output
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                if alpha_t.numel() > 1:
                    alpha_t = alpha_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    sigma_t = sigma_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.config.prediction_type == "sample":
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                if alpha_t.numel() > 1:
                    alpha_t = alpha_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    sigma_t = sigma_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                epsilon = (sample - alpha_t * model_output) / sigma_t
                x0_pred = model_output
            elif self.config.prediction_type == "v_prediction":
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                if alpha_t.numel() > 1:
                    alpha_t = alpha_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    sigma_t = sigma_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                epsilon = alpha_t * model_output + sigma_t * sample
                x0_pred = alpha_t * sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the DPMSolverMultistepScheduler."
                )

            if self.config.thresholding:
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                if alpha_t.numel() > 1:
                    alpha_t = alpha_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    sigma_t = sigma_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                x0_pred = (sample - sigma_t * epsilon) / alpha_t
                x0_pred = self._threshold_sample(x0_pred)
                epsilon = (sample - alpha_t * x0_pred) / sigma_t

            return epsilon, x0_pred
    def compute_epsilon_with_pred_x0(
        self,
        x0_pred: torch.FloatTensor,
        *args,
        sample: torch.FloatTensor = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Compute epsilon (noise) and processed x0 prediction from given predicted x0.
        
        This function handles different algorithm types (DPM-Solver/DPM-Solver++) and prediction
        types (epsilon/sample/v_prediction) to compute the noise estimate and optionally apply
        thresholding to the predicted x0.
        
        Args:
            x0_pred (torch.FloatTensor): The model's predicted x0 (denoised sample)
            *args: Variable length argument list that may contain:
                - timestep (deprecated): Current timestep (position 0)
                - sample (optional): Noisy sample if not provided as keyword (position 1)
            sample (torch.FloatTensor, optional): The noisy input sample at current timestep
            **kwargs: Additional keyword arguments that may contain:
                - timestep (deprecated): Current timestep
        
        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: A tuple containing:
                - epsilon: The computed noise estimate
                - x0_pred: The processed x0 prediction (with thresholding if enabled)
        """
        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError("missing `sample` as a required keyward argument")
        if timestep is not None:
            deprecate(
                "timesteps",
                "1.0.0",
                "Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        # DPM-Solver++ needs to solve an integral of the data prediction model.
        if self.config.algorithm_type in ["dpmsolver++", "sde-dpmsolver++"]:
            if self.config.prediction_type == "epsilon":
                # DPM-Solver and DPM-Solver++ only need the "mean" output.
                if self.config.variance_type in ["learned", "learned_range"]:
                    x0_pred = x0_pred[:, :3]
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                if alpha_t.numel() > 1:
                    alpha_t = alpha_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    sigma_t = sigma_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                epsilon = (sample - alpha_t * x0_pred) / sigma_t
            elif self.config.prediction_type == "sample":
                # x0_pred = model_output
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                if alpha_t.numel() > 1:
                    alpha_t = alpha_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    sigma_t = sigma_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                epsilon = (sample - alpha_t * x0_pred) / sigma_t
            elif self.config.prediction_type == "v_prediction":
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                if alpha_t.numel() > 1:
                    alpha_t = alpha_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    sigma_t = sigma_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                # x0_pred = alpha_t * sample - sigma_t * model_output
                epsilon = alpha_t * x0_pred + sigma_t * sample
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the DPMSolverMultistepScheduler."
                )

            if self.config.thresholding:
                x0_pred = self._threshold_sample(x0_pred)

            return epsilon,x0_pred

        # DPM-Solver needs to solve an integral of the noise prediction model.
        elif self.config.algorithm_type in ["dpmsolver", "sde-dpmsolver"]:

            if self.config.prediction_type == "epsilon":
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                if alpha_t.numel() > 1:
                    alpha_t = alpha_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    sigma_t = sigma_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                epsilon = (sample - alpha_t * x0_pred) / sigma_t
            elif self.config.prediction_type == "sample":
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                if alpha_t.numel() > 1:
                    alpha_t = alpha_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    sigma_t = sigma_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                epsilon = (sample - alpha_t * model_output) / sigma_t
                # x0_pred = model_output
            elif self.config.prediction_type == "v_prediction":
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                # #print(alpha_t.size(), sample.size(), sigma_t.size(), model_output.size())
                if alpha_t.numel() > 1:
                    alpha_t = alpha_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    sigma_t = sigma_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                epsilon = alpha_t * x0_pred + sigma_t * sample

                # x0_pred = alpha_t * sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the DPMSolverMultistepScheduler."
                )

            if self.config.thresholding:
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                if alpha_t.numel() > 1:
                    alpha_t = alpha_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    sigma_t = sigma_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                # x0_pred = (sample - sigma_t * epsilon) / alpha_t
                x0_pred = self._threshold_sample(x0_pred)
                epsilon = (sample - alpha_t * x0_pred) / sigma_t

            return epsilon,x0_pred
    def _get_variance_v1(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance
    def _get_variance(self, timestep, prev_timestep=None):
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
        the multistep DPMSolver.

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`LEdits++`].
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Improve numerical stability for small number of steps
        lower_order_final = (self.step_index == len(self.timesteps) - 1) and (
            self.config.euler_at_final
            or (self.config.lower_order_final and len(self.timesteps) < 15)
            or self.config.final_sigmas_type == "zero"
        )
        lower_order_second = (
            (self.step_index == len(self.timesteps) - 2) and self.config.lower_order_final and len(self.timesteps) < 15
        )

        epsilon,x0_pred = self.convert_model_output_and_output_pred_x0(model_output, sample=sample)
        if self.config.algorithm_type in ["dpmsolver++", "sde-dpmsolver++"]:
            model_output = x0_pred
        elif self.config.algorithm_type in ["dpmsolver", "sde-dpmsolver"]:
            model_output = epsilon
        else: raise NotImplementedError
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)




        if self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"] and variance_noise is None:
            noise = randn_tensor(
                model_output.shape, generator=generator, device=model_output.device, dtype=torch.float32
            )
        elif self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
            noise = variance_noise.to(device=model_output.device, dtype=torch.float32)
        else:
            noise = None

        if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.dpm_solver_first_order_update(model_output, sample=sample, noise=noise)
        elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
            prev_sample = self.multistep_dpm_solver_second_order_update(self.model_outputs, sample=sample, noise=noise)
        else:
            prev_sample = self.multistep_dpm_solver_third_order_update(self.model_outputs, sample=sample,noise=noise)

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        # Cast sample back to expected dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,x0_pred)

        return SchedulerOutput(prev_sample=prev_sample)
    def batch_step_with_noise(
        self,
        model_output: torch.FloatTensor,
        timesteps: List[int],
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Step function propagating the sample with the multistep DPM-Solver.

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`~scheduling_utils.SchedulerOutput`] or `tuple`: [`~scheduling_utils.SchedulerOutput`] if `return_dict` is
            True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
        # #print("self.sigmas",self.sigmas)
        self.sigmas = self.sigmas.to(model_output.device)

        # self.lambda_t = self.lambda_t.to(model_output.device)
        # self.alpha_t = self.alpha_t.to(model_output.device)
        # self.sigma_t = self.sigma_t.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)
        t = timesteps.to(model_output.device)

        matches = (self.timesteps[None, :] == t[:, None])
        edgecases = ~matches.any(dim=1)
        step_index = torch.argmax(matches.int(), dim=1)
        step_index[edgecases] = len(self.timesteps) - 1 # if no match, then set to len(self.timesteps) - 1
        edgecases = (step_index == len(self.timesteps) - 1)

        prev_t = self.timesteps[ torch.clip(step_index+1, max=len(self.timesteps) - 1) ]
        prev_t[edgecases] = 0
        self._step_index = step_index

        t = t.view(-1, *([1]*(model_output.ndim - 1)))
        prev_t = prev_t.view(-1, *([1]*(model_output.ndim - 1)))

        epsilon,x0_pred = self.convert_model_output_and_output_pred_x0(model_output, sample=sample)
        if self.config.algorithm_type in ["dpmsolver++", "sde-dpmsolver++"]:
            model_output = x0_pred
        elif self.config.algorithm_type in ["dpmsolver", "sde-dpmsolver"]:
            model_output = epsilon
        else: raise NotImplementedError

        #print("self.config.solver_order",self.config.solver_order)
        if self.config.solver_order == 1 or len(t) == 1:
            prev_sample = self.dpm_solver_first_order_update(model_output, t, prev_t, sample,noise=torch.zeros_like(sample))
        elif self.config.solver_order == 2 or len(t) == 2:
            # first element in batch must do first_order_update
            prev_sample1 = self.dpm_solver_first_order_update(model_output[:1], t[:1], prev_t[:1], sample[:1],noise=torch.zeros_like(sample))

            model_outputs_list = [model_output[:-1], model_output[1:]]
            timestep_list = [t[:-1], t[1:]]
            prev_sample2 = self.multistep_dpm_solver_second_order_update(
                model_outputs_list, timestep_list, prev_t[1:], sample[1:]
                ,noise=torch.zeros_like(sample)
            )

            prev_sample = torch.cat([prev_sample1, prev_sample2], dim=0)
        else:
            # first element in batch must do first_order_update
            prev_sample1 = self.dpm_solver_first_order_update(model_output[:1], t[:1], prev_t[:1], sample[:1],noise=torch.zeros_like(sample))

            # second element in batch must do second_order update
            model_outputs_list = [model_output[:1], model_output[1:2]]
            timestep_list = [t[:1], t[1:2]]
            prev_sample2 = self.multistep_dpm_solver_second_order_update(
                model_outputs_list, timestep_list, prev_t[1:2], sample[1:2]
                , noise=torch.zeros_like(sample)
            )

            model_outputs_list = [model_output[:-2], model_output[1:-1], model_output[2:]]
            timestep_list = [t[:-2], t[1:-1], t[2:]]
            prev_sample3 = self.multistep_dpm_solver_third_order_update(
                model_outputs_list, timestep_list, prev_t[2:], sample[2:]
            )

            prev_sample = torch.cat([prev_sample1, prev_sample2, prev_sample3], dim=0)

        # doing this otherwise set_timesteps throws an error
        # if worried about efficiency, can override the set_timesteps function
        self.lambda_t = self.lambda_t.to('cpu')
        initial_para_dur = 0
        return prev_sample,initial_para_dur,x0_pred
    def batch_step_no_noise(
        self,
        model_output: torch.FloatTensor,
        timesteps: List[int],
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Step function propagating the sample with the multistep DPM-Solver.

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`~scheduling_utils.SchedulerOutput`] or `tuple`: [`~scheduling_utils.SchedulerOutput`] if `return_dict` is
            True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
        # #print("self.sigmas",self.sigmas)
        self.sigmas = self.sigmas.to(model_output.device)

        # self.lambda_t = self.lambda_t.to(model_output.device)
        # self.alpha_t = self.alpha_t.to(model_output.device)
        # self.sigma_t = self.sigma_t.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)
        t = timesteps.to(model_output.device)
        #print("timesteps",timesteps)
        # #print("self.step_index",self.step_index)
        matches = (self.timesteps[None, :] == t[:, None])
        edgecases = ~matches.any(dim=1)
        step_index = torch.argmax(matches.int(), dim=1)
        step_index[edgecases] = len(self.timesteps) - 1 # if no match, then set to len(self.timesteps) - 1
        edgecases = (step_index == len(self.timesteps) - 1)
        # #print("step_index", step_index)
        prev_t = self.timesteps[ torch.clip(step_index+1, max=len(self.timesteps) - 1) ]
        prev_t[edgecases] = 0
        self._step_index = step_index
        #print("self.step_index", self.step_index)
        t = t.view(-1, *([1]*(model_output.ndim - 1)))
        prev_t = prev_t.view(-1, *([1]*(model_output.ndim - 1)))

        epsilon,x0_pred = self.convert_model_output_and_output_pred_x0(model_output, sample=sample)
        if self.config.algorithm_type in ["dpmsolver++", "sde-dpmsolver++"]:
            model_output = x0_pred
        elif self.config.algorithm_type in ["dpmsolver", "sde-dpmsolver"]:
            model_output = epsilon
        else: raise NotImplementedError

        #print("self.config.solver_order",self.config.solver_order)
        if self.config.solver_order == 1 or len(t) == 1:
            prev_sample = self.dpm_solver_first_order_update(model_output, t, prev_t, sample,noise=torch.zeros_like(sample))
        elif self.config.solver_order == 2 or len(t) == 2:
            # first element in batch must do first_order_update
            prev_sample1 = self.dpm_solver_first_order_update(model_output[:1], t[:1], prev_t[:1], sample[:1],noise=torch.zeros_like(sample))

            model_outputs_list = [model_output[:-1], model_output[1:]]
            timestep_list = [t[:-1], t[1:]]
            prev_sample2 = self.multistep_dpm_solver_second_order_update(
                model_outputs_list, timestep_list, prev_t[1:], sample[1:]
                ,noise=torch.zeros_like(sample)
            )

            prev_sample = torch.cat([prev_sample1, prev_sample2], dim=0)
        else:
            # first element in batch must do first_order_update
            prev_sample1 = self.dpm_solver_first_order_update(model_output[:1], t[:1], prev_t[:1], sample[:1],noise=torch.zeros_like(sample))

            # second element in batch must do second_order update
            model_outputs_list = [model_output[:1], model_output[1:2]]
            timestep_list = [t[:1], t[1:2]]
            prev_sample2 = self.multistep_dpm_solver_second_order_update(
                model_outputs_list, timestep_list, prev_t[1:2], sample[1:2]
                , noise=torch.zeros_like(sample)
            )

            model_outputs_list = [model_output[:-2], model_output[1:-1], model_output[2:]]
            timestep_list = [t[:-2], t[1:-1], t[2:]]
            prev_sample3 = self.multistep_dpm_solver_third_order_update(
                model_outputs_list, timestep_list, prev_t[2:], sample[2:]
            )

            prev_sample = torch.cat([prev_sample1, prev_sample2, prev_sample3], dim=0)

        # doing this otherwise set_timesteps throws an error
        # if worried about efficiency, can override the set_timesteps function
        self.lambda_t = self.lambda_t.to('cpu')
        return prev_sample
    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of equal spacing between timesteps is used. If `timesteps` is passed,
                `num_inference_steps` must be `None`.

        """
        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `custom_timesteps`.")

        if timesteps is not None:
            for i in range(1, len(timesteps)):
                if timesteps[i] >= timesteps[i - 1]:
                    raise ValueError("`custom_timesteps` must be in descending order.")

            if timesteps[0] >= self.config.num_train_timesteps:
                raise ValueError(
                    f"`timesteps` must start before `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps}."
                )

            timesteps = np.array(timesteps, dtype=np.int64)
            self.custom_timesteps = True
        else:
            if num_inference_steps > self.config.num_train_timesteps:
                raise ValueError(
                    f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                    f" maximal {self.config.num_train_timesteps} timesteps."
                )

            self.num_inference_steps = num_inference_steps
            self.custom_timesteps = False

            clipped_idx = torch.searchsorted(torch.flip(self.lambda_t, [0]), self.config.lambda_min_clipped)
            last_timestep = ((self.config.num_train_timesteps - clipped_idx).numpy()).item()

            # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
            if self.config.timestep_spacing == "linspace":
                timesteps = (
                    np.linspace(0, last_timestep - 1, num_inference_steps + 1).round()[::-1][:-1].copy().astype(np.int64)
                )
            elif self.config.timestep_spacing == "leading":
                step_ratio = last_timestep // (num_inference_steps + 1)
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = (np.arange(0, num_inference_steps + 1) * step_ratio).round()[::-1][:-1].copy().astype(np.int64)
                timesteps += self.config.steps_offset
            elif self.config.timestep_spacing == "trailing":
                step_ratio = self.config.num_train_timesteps / num_inference_steps
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = np.arange(last_timestep, 0, -step_ratio).round().copy().astype(np.int64)
                timesteps -= 1
            else:
                raise ValueError(
                    f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
                )

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        log_sigmas = np.log(sigmas)

        if self.config.use_karras_sigmas:
            # sigmas = np.flip(sigmas).copy()
            # sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            # timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round()
            raise ValueError(
                f"use_karras_sigmas is not supported. "
            )
        elif self.config.use_lu_lambdas:
            # lambdas = np.flip(log_sigmas.copy())
            # lambdas = self._convert_to_lu(in_lambdas=lambdas, num_inference_steps=num_inference_steps)
            # sigmas = np.exp(lambdas)
            # timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round()
            raise ValueError(
                f"use_lu_lambdas is not supported. "
            )
        else:
            sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)

        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(
                f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}"
            )

        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)

        self.num_inference_steps = len(timesteps)

        self.model_outputs = [
            None,
        ] * self.config.solver_order
        self.lower_order_nums = 0

        # add an index counter for schedulers that allow duplicated timesteps
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    def set_coarse_and_fine_timesteps(self, num_time_subintervals: int, num_preconditioning_steps: int):
        if num_time_subintervals > len(self.timesteps):
            raise ValueError(f"`time subintervals number` num_time_subintervals must be no more than {len(self.timesteps)}.")
        
        # Calculate coarse timesteps, i.e. the starting timesteps for each subintervals
        coarse_step_index = np.linspace(0.0, len(self.timesteps), num=num_time_subintervals,endpoint=False).round().copy().astype(np.int64)
        self.base_timesteps =  self.timesteps
        self.base_sigmas = self.sigmas
        self.coarse_timesteps = self.timesteps[coarse_step_index]
        self.coarse_timesteps_sigmas =  torch.cat([self.sigmas[coarse_step_index],self.sigmas[-1].unsqueeze(0)])

        # Calculate initial timesteps for preconditioning of initialization
        init_step_index = np.linspace(0.0, num_time_subintervals, num=num_preconditioning_steps, endpoint=False).round().copy().astype(
            np.int64)
        self.initial_timesteps = self.coarse_timesteps[init_step_index]
        self.initial_timesteps_sigmas = torch.cat([self.coarse_timesteps_sigmas[init_step_index], self.coarse_timesteps_sigmas[-1].unsqueeze(0)])

        # Prepare fine timesteps matrix, i.e., the timesteps within each subintervals
        steps_matrix = []
        interval_indexs = zip(coarse_step_index,np.append(coarse_step_index[1:], len(self.timesteps)))
        interval_len = np.append(coarse_step_index[1:], len(self.timesteps)) - coarse_step_index
        self.interval_len = torch.tensor(interval_len.tolist(), dtype=torch.int64)
        max_interval_len = np.max(interval_len)
        for i,(start_idx,end_idx) in enumerate(interval_indexs):
            times = self.timesteps[start_idx:end_idx]
            values_to_insert = torch.tensor([times[-1].item()]* (max_interval_len - interval_len[i]),device = self.timesteps.device,dtype = self.timesteps.dtype)
            steps_matrix.append(torch.cat([times,values_to_insert]))

        # Store results
        self.fine_timesteps_matrix = torch.stack(steps_matrix)
        self.max_fine_timestep_num = max_interval_len
        self.coarse_timestep_num = num_time_subintervals

    def dpmsolver_step(
        self,
        pred_original_sample: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        This function predicts the sample from the previous timestep by running the SDE in reverse. It's adapted from 
        the `step()` method in the Scheduler class, with one key modification: instead of using `model_out` as input, 
        it takes `pred_original_sample`. The function then advances the diffusion process starting from this predicted 
        original sample.

        Args:
            pred_original_sample (`torch.Tensor`):
                The predcited original sample.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`LEdits++`].
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Improve numerical stability for small number of steps
        lower_order_final = (self.step_index == len(self.timesteps) - 1) and (
            self.config.euler_at_final
            or (self.config.lower_order_final and len(self.timesteps) < 15)
            or self.config.final_sigmas_type == "zero"
        )
        lower_order_second = (
            (self.step_index == len(self.timesteps) - 2) and self.config.lower_order_final and len(self.timesteps) < 15
        )

        epsilon,x0_pred = self.compute_epsilon_with_pred_x0(pred_original_sample, sample=sample)
        if self.config.algorithm_type in ["dpmsolver++", "sde-dpmsolver++"]:
            model_output = x0_pred
        elif self.config.algorithm_type in ["dpmsolver", "sde-dpmsolver"]:
            model_output = epsilon
        else: raise NotImplementedError
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)




        if self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"] and variance_noise is None:
            noise = randn_tensor(
                model_output.shape, generator=generator, device=model_output.device, dtype=torch.float32
            )
        elif self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
            noise = variance_noise.to(device=model_output.device, dtype=torch.float32)
        else:
            noise = None

        if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.dpm_solver_first_order_update(model_output, sample=sample, noise=noise)
        elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
            prev_sample = self.multistep_dpm_solver_second_order_update(self.model_outputs, sample=sample, noise=noise)
        else:
            prev_sample = self.multistep_dpm_solver_third_order_update(self.model_outputs, sample=sample,noise=noise)

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        # Cast sample back to expected dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,x0_pred)

        return SchedulerOutput(prev_sample=prev_sample)