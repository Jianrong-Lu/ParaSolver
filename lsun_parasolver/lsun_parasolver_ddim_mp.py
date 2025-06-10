
import inspect
from typing import (
    TYPE_CHECKING, Dict, Iterator, List, Optional, 
    Sequence, Set, TypeVar, Union, cast, Any, Callable, Tuple
)
if TYPE_CHECKING:
    import torch.jit
    import torch.jit._state

import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines import DDIMPipeline
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput




def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        # accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # if not accepts_timesteps:
        #     raise ValueError(
        #         f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
        #         f" timestep schedules. Please check whether you are using the correct scheduler."
        #     )
        scheduler.set_timesteps(num_inference_steps=num_inference_steps,device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class ParaSolverDDIMDiffusionPipeline(DDIMPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__(unet, scheduler)


        # overwriting parent class
        scheduler = scheduler
        self.register_modules(unet=unet, scheduler=scheduler)

        #Added
        self.initial_latents = dict()
        self.batch_size = None
        self.predicted_original_sample = None


    def initialize_pipeline_variables(self):
        #pip initialize
        self.initial_latents = dict()
        self.batch_size = None
        self.predicted_original_sample = None


        #scheduler
        self.scheduler.base_timesteps =  None
        # self.scheduler.base_sigmas = None

        self.scheduler.coarse_timesteps = None
        # self.scheduler.coarse_timesteps_sigmas = None

        self.scheduler.initial_timesteps = None
        # self.scheduler.initial_timesteps_sigmas = None

        self.scheduler.fine_timesteps_matrix = None
        self.scheduler.max_fine_timestep_num = None
        self.scheduler.coarse_timestep_num = None
        self.scheduler.fine_timesteps_sigmas_matrix = None

        self.scheduler.interval_len =None



    @torch.no_grad()
    def paradigms_forward_worker(
            self,
            mp_queues: Optional[torch.FloatTensor] = None,
            device: Optional[str] = None,
    ):
        while True:
            ret = mp_queues[0].get()
            if ret is None:
                del ret
                return

            (latent_model_input, t_vec, chunk, idx, begin_idx) = ret
            model_output_chunk = self.unet(
                latent_model_input[chunk].flatten(0, 1).to(device),
                t_vec[chunk].flatten(0, 1).to(device),
            )[0]

            del ret

            mp_queues[1].put(
                (model_output_chunk, idx),
            )

    @torch.no_grad()
    def preconditioning_initialization(
        self,
        num_inference_steps:int,
        timesteps:torch.FloatTensor,
        image: torch.FloatTensor,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        use_clipped_model_output: Optional[bool] = None,
        num_preconditioning_steps:int=0,
        ):
        with torch.no_grad():
            initial_para_dur = 0
            stats_pass_count = 0
            stats_flop_count = 0
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if num_preconditioning_steps>0:
                        if i < num_preconditioning_steps:
                            model_output = self.unet(image, t).sample
                            generator = generator.manual_seed(t.item())
                            image, x_0 = self.scheduler.step(
                                model_output, t, image, eta=eta, use_clipped_model_output=True,
                                generator=generator, return_dict=True
                            ).values()
                            self.predicted_original_sample = x_0.clone()

                            stats_pass_count += 1
                            stats_flop_count += 1 * image.size()[1]

                            end.record()
                            torch.cuda.synchronize()
                            initial_para_dur = start.elapsed_time(end)

                        else:
                            generator = generator.manual_seed(t.item())
                            image, _ = self.scheduler.ddim_step(
                                self.predicted_original_sample, t, image, eta=eta, use_clipped_model_output=True,
                                generator=generator, return_dict=True
                            ).values()
                    else:
                        model_output = randn_tensor(
                            image.shape, generator=generator, device=image.device,
                            dtype=image.dtype) 
                        generator = generator.manual_seed(t.item())
                        image, _ = self.scheduler.step(
                            model_output, t, image, eta=eta, use_clipped_model_output=True,
                            generator=generator, return_dict=True
                        ).values()
                    progress_bar.update()
                    if i == num_inference_steps - 1:
                        self.initial_latents[-1] = image
                    else:
                        self.initial_latents[timesteps[i+1].item()] = image
        return  stats_pass_count,stats_flop_count,initial_para_dur
    @torch.no_grad()
    def initialize_points_outside_current_window(
        self,
        num_inference_steps:int,
        timesteps:torch.FloatTensor,
        image: torch.FloatTensor,
        X_0: torch.FloatTensor,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        use_clipped_model_output: Optional[bool] = None,
        start_point:int=0,
        end_point:int=0
        ):
        with torch.no_grad():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if i >= start_point and i < end_point:
                        generator = generator.manual_seed(t.item())
                        image, _ = self.scheduler.ddim_step(
                            X_0, t, image, eta=eta, use_clipped_model_output=True,
                            generator=generator, return_dict=True
                        ).values()
                        progress_bar.update()
                        if i == num_inference_steps - 1:
                            self.initial_latents[-1] = image
                        else:
                            self.initial_latents[timesteps[i+1].item()] = image
            end.record()
            torch.cuda.synchronize()
            initial_para_dur = start.elapsed_time(end)
        return initial_para_dur

    @torch.no_grad()
    def preparation_for_diffusion(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )


        # set step values
        device = self._execution_device
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device)
        #print("def preparation_for_lsun_diffusion: timesteps, num_inference_steps",timesteps, num_inference_steps)



        image = randn_tensor(image_shape, generator=generator, device=device, dtype=self.unet.dtype)

        # coarse_timesteps = None
        self.initial_latents[timesteps[0].item()] = image
        self.batch_size = batch_size


    @torch.no_grad()
    def parasolver_forward(
            self,
            num_inference_steps: int = 1000,
            num_time_subintervals: int = 50,
            num_preconditioning_steps: int = 0,
            N_F: int = 1,
            parallel: int = 10,
            tolerance: float = 0.1,
            picard_tlr: float = 0.01,
            batch_size: int = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            eta: float = 0.0,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            full_return: bool = True,
            mp_queues: Optional[torch.FloatTensor] = None,
            device: Optional[str] = None,
            num_consumers: Optional[int] = None,
            initialized_method: str = "seq"
    ):

        #print("lsun ddim parallel pipeline!", flush=True)


        #1.Reset all Settings to their initial state
        self.initialize_pipeline_variables()

        #2.Set base time steps and coarse time steps as well as related parameters
        num_time_subintervals = min(num_time_subintervals,num_inference_steps)
        num_preconditioning_steps = min(num_time_subintervals, num_preconditioning_steps)
        parallel = min(parallel, num_time_subintervals)

        self.scheduler.set_timesteps(num_inference_steps)  
        self.scheduler.set_coarse_and_fine_timesteps(num_time_subintervals=num_time_subintervals,num_preconditioning_steps=num_preconditioning_steps)

        #3. Using coarse scheduler setting
        self.scheduler.timesteps = self.scheduler.coarse_timesteps
        self.scheduler.num_inference_steps =len(self.scheduler.coarse_timesteps)

        self.preparation_for_diffusion(batch_size = batch_size,generator=generator,num_inference_steps=num_time_subintervals,timesteps=self.scheduler.coarse_timesteps)

        #5. Prepare_noise
        noise_array = torch.zeros_like(torch.stack([self.initial_latents[999]]*(num_time_subintervals)))
        for j in range(num_time_subintervals):
            cur_t = self.scheduler.coarse_timesteps[j].item()
            if j < num_time_subintervals - 1:
                prev_timestep = self.scheduler.coarse_timesteps[j+1].item()
            else:
                prev_timestep = 0
            var = self.scheduler._get_variance(cur_t, prev_timestep)
            std_dev_t = eta * var ** (0.5)
            if eta > 0:
                generator.manual_seed(cur_t)
                variance_noise = randn_tensor(
                    self.initial_latents[999].shape, generator=generator, device=self.initial_latents[999].device, dtype=self.initial_latents[999].dtype
                )
                variance = std_dev_t * variance_noise
                noise_array[j] = variance.clone()

        #4. Initialize initial  points of all subinterval
        self.scheduler.timesteps = self.scheduler.coarse_timesteps
        # # self.scheduler.sigmas = self.scheduler.coarse_timesteps_sigmas
        self.scheduler.num_inference_steps = num_time_subintervals

        #print("initial_latents keys", self.initial_latents.keys())
        stats_pass_count, stats_flop_count,initial_para_dur = self.preconditioning_initialization(num_inference_steps=self.scheduler.num_inference_steps,
                                        timesteps=self.scheduler.timesteps,
                                        generator=generator,
                                        eta=eta,
                                        image=list(self.initial_latents.values())[0],
                                        num_preconditioning_steps=num_preconditioning_steps)


        # 4. Parallel parameters initializatiom
        timesteps = self.scheduler.coarse_timesteps  #
        latents_time_evolution_buffer = torch.stack(list(self.initial_latents.values())) 
        time_evolution_buffer_matrix = self.scheduler.fine_timesteps_matrix  
        self.initial_latents.clear()





        #6.Prepare stop parameters
        inverse_variance_norm = 1. / torch.tensor(
            [self.scheduler._get_variance_v1(timesteps[j].item(),timesteps[j + 1].item() if j < len(timesteps) - 1 else 0)
             for j in list(range(len(timesteps))) + [len(timesteps) - 1]]).to(torch.float32).to(device)

        latent_dim = latents_time_evolution_buffer[0].numel()
        inverse_variance_norm = inverse_variance_norm[:, None] / latent_dim
        scaled_tolerance = (tolerance ** 2)


        #7.Set fine scheduler  for parallel iteration, such that  the scheduler can accurately find the previous timesteps or sigmas
        self.scheduler.timesteps = self.scheduler.base_timesteps
        self.scheduler.num_inference_steps = len(self.scheduler.base_timesteps)
        # self.scheduler.sigmas = self.scheduler.base_sigmas
        # self.scheduler._step_index = None


        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        end_i = parallel
        i = 0 #begin_idx = i
        k = 0
        noises_dur_time = 0
        update_dur_time = 0
        while i < num_time_subintervals:
            parallel_len_outer = end_i - i

            timestep_list = time_evolution_buffer_matrix[i:end_i].T
            block_latents_list = latents_time_evolution_buffer[i:end_i].clone()
            interval_len = self.scheduler.interval_len[i:end_i]
            max_fine_timestep_num, _ = torch.max(interval_len, dim=0)
            #Solving subproblems concurrently, with each one processed sequentially through iteration. 
            for begin_idx in range(max_fine_timestep_num):
                indices_started = torch.nonzero(interval_len > begin_idx).view(-1)
                parallel_len = len(indices_started)
                block_t = timestep_list[begin_idx][indices_started][:, None].repeat(1, self.batch_size ).to(
                    block_latents_list.device)
                t_vec = block_t
                # expand the latents if we are doing classifier free guidance
                block_latents = block_latents_list[indices_started]
                latent_model_input = block_latents


                if parallel_len <= 2 or num_consumers == 1:
                    model_output = self.unet(
                        latent_model_input.flatten(0, 1).to(device),
                        t_vec.flatten(0, 1).to(device),
                    )[0]
                else:
                    chunks = torch.arange(parallel_len).tensor_split(num_consumers)
                    num_chunks = min(parallel_len, num_consumers)

                    for idx1 in range(num_chunks):
                        mp_queues[0].put(
                            (latent_model_input, t_vec, chunks[idx1], idx1, begin_idx)
                        )

                    model_output = [None for _ in range(num_chunks)]

                    for _ in range(num_chunks):
                        ret = mp_queues[1].get()
                        model_output_chunk, idx = ret
                        model_output[idx] = model_output_chunk.to(device)
                        del ret

                    model_output = torch.cat(model_output)

                # end1.record()
                # torch.cuda.synchronize()
                # elapsed = start1.elapsed_time(end1)
                # elapsed_per_t = elapsed / parallel_len
                # #print("model parallel elapsed time:", elapsed, "elapsed_per_t:", elapsed_per_t)

                per_latent_shape = model_output.shape[1:]

                model_output = model_output.reshape(
                    parallel_len * self.batch_size ,
                    *per_latent_shape
                )

                block_latents_new,noises_dur,x_0 = self.scheduler.batch_step_with_noise(
                    model_output=model_output,
                    timesteps=block_t.flatten(0, 1),
                    sample=block_latents.flatten(0, 1),
                    eta=eta,
                )

                block_latents_new = block_latents_new.reshape(block_latents.shape).to(model_output.dtype)
                x_0 = x_0.reshape(block_latents.shape).to(model_output.dtype)
                if begin_idx == 0:
                    self.predicted_original_sample = x_0.clone()
                else:
                    self.predicted_original_sample[indices_started] = x_0.clone()
                noises_dur_time = noises_dur_time + noises_dur
                block_latents_list[indices_started] = block_latents_new
                stats_pass_count += 1
                stats_flop_count += parallel_len*block_latents_new.size()[1]

            delta = block_latents_list - latents_time_evolution_buffer[i:end_i]
            cumulative_delta = torch.cumsum(delta, dim=0)
            block_latents_new = latents_time_evolution_buffer[i][None,] + cumulative_delta


            cur_error_vec = (block_latents_new.float() - latents_time_evolution_buffer[i + 1: end_i + 1].float()).reshape(
                parallel_len_outer, self.batch_size , -1)

            #Parallel correction of initial points.
            latents_time_evolution_buffer[i + 1:end_i + 1] = block_latents_new
            # initialize the new sliding window latents with the end of the current window,
            # should be better than random initialization


            cur_error = torch.linalg.norm(cur_error_vec, dim=-1).pow(2)
            error_ratio = cur_error * inverse_variance_norm[i + 1:end_i + 1]
            error_ratio = torch.nn.functional.pad(
                error_ratio, (0, 0, 0, 1), value=1e9
            )  # handle the case when everything is below ratio, by padding the end of parallel_len dimension
            print(f"i={i}_error_ratio_aft_pading:", error_ratio[-3:])
            any_error_at_time = torch.max(error_ratio > scaled_tolerance, dim=1).values.int()
            ind = torch.argmax(any_error_at_time).item()
            # compute the new begin and end idxs for the window
            new_i = i + min(1 + ind, parallel)
            new_end_i = min(new_i + parallel, num_time_subintervals)

            if end_i < num_time_subintervals-1:
                X_0 = self.predicted_original_sample[-1]
                update_dur = self.initialize_points_outside_current_window(
                                                num_inference_steps=self.scheduler.coarse_timestep_num,
                                                timesteps=self.scheduler.coarse_timesteps,
                                                generator=generator,
                                                eta=eta,
                                                image=latents_time_evolution_buffer[end_i],
                                                X_0 = X_0,
                                                start_point=end_i,
                                                end_point = new_end_i)
                update_dur_time = update_dur_time + update_dur
                block_latents_new = torch.stack(list(self.initial_latents.values()))
                latents_time_evolution_buffer[end_i+1:new_end_i+1] = block_latents_new
                self.initial_latents.clear()


            i = new_i
            end_i = new_end_i
            k = k + 1

            print(f"i={i}, k={k}, ind={ind}, stride={min(1 + ind, end_i)}")

        print("pass count", stats_pass_count)
        print("flop count", stats_flop_count)
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()

        print("initial elapsed time:", initial_para_dur)
        print("noise elapsed time:", noises_dur_time)
        print("updata elapsed time:", update_dur_time)
        print("parallel elapsed time:",start.elapsed_time(end) + initial_para_dur - noises_dur_time- update_dur_time)
        stats = {
            'pass_count': stats_pass_count,
            'flops_count': stats_flop_count,
            'time': start.elapsed_time(end) + initial_para_dur - noises_dur_time- update_dur_time,
        }
        print("total elapsed time:",stats['time'])
        print("done", flush=True)


        image = latents_time_evolution_buffer[-1]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        def process_image(image):
            if output_type == "pil":
                image = self.numpy_to_pil(image)

            # Offload last model to CPU
            if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                #print("offload hook", flush=True)
                self.final_offload_hook.offload()

            return image

        image = process_image(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image), stats
