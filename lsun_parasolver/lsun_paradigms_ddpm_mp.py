
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch.jit
    import torch.jit._state

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines import DDPMPipeline
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
import inspect
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Sequence, Set, TypeVar, Union
import torch


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
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
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class ParaDiGMSDDPMDiffusionPipeline(DDPMPipeline):
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
    def ParaDiGMS_paradigms_forward(
            self,
            batch_size: int = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            num_inference_steps: int = 1000,
            eta=0.0,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            num_consumers: int = 1,
            mp_queues: Optional[torch.FloatTensor] = None,
            tolerance: float = 0.1,
            parallel: int = 10,
    ):


        print("parallel ParaDiGMS_DDPM pipeline!", flush=True)


        device = self._execution_device
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

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=device,dtype=self.unet.dtype)

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device)
        print("def preparation_for_lsun_diffusion: timesteps, num_inference_steps",timesteps, num_inference_steps)



        stats_pass_count = 0
        stats_flop_count = 0
        parallel = min(parallel, len(self.scheduler.timesteps))

        begin_idx = 0
        end_idx = parallel
        latents_time_evolution_buffer = torch.stack([image] * (len(self.scheduler.timesteps) + 1))

        noise_array = torch.zeros_like(latents_time_evolution_buffer)
        for j in range(len(self.scheduler.timesteps)):
            base_noise = torch.randn_like(image)
            noise = (self.scheduler._get_variance(self.scheduler.timesteps[j]) ** 0.5) * base_noise
            noise_array[j] = noise.clone()

        # We specify the error tolerance as a ratio of the scheduler's noise magnitude. We similarly compute the error tolerance
        # outside of the denoising loop to avoid recomputing it at every step.
        # We will be dividing the norm of the noise, so we store its inverse here to avoid a division at every step.
        inverse_variance_norm = 1. / torch.tensor(
            [self.scheduler._get_variance(self.scheduler.timesteps[j]) for j in range(len(self.scheduler.timesteps))] + [0]).to(
            noise_array.device)
        latent_dim = noise_array[0, 0].numel()
        inverse_variance_norm = inverse_variance_norm[:, None] / latent_dim

        scaled_tolerance = (tolerance ** 2)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        while begin_idx < num_inference_steps:
            # start2 = torch.cuda.Event(enable_timing=True)
            # end2 = torch.cuda.Event(enable_timing=True)
            # start2.record()
            # these have shape (parallel_dim, 2*batch_size, ...)
            # parallel_len is at most parallel, but could be less if we are at the end of the timesteps
            # we are processing batch window of timesteps spanning [begin_idx, end_idx)
            parallel_len = end_idx - begin_idx
            # print("parallel", parallel, "parallel_len", parallel_len)
            # block_prompt_embeds = torch.stack([self.prompt_embeds] * parallel_len)
            block_latents = latents_time_evolution_buffer[begin_idx:end_idx]
            block_t = self.scheduler.timesteps[begin_idx:end_idx, None].repeat(1, batch_size)
            t_vec = block_t
            print(block_latents.size(),block_t.size())
            # expand the latents if we are doing classifier free guidance
            latent_model_input = block_latents


            # start1 = torch.cuda.Event(enable_timing=True)
            # end1 = torch.cuda.Event(enable_timing=True)

            # start1.record()

            if parallel_len <= 2 or num_consumers == 1:
                # print("parallel_len", parallel_len, "num_consumers", num_consumers)
                model_output = self.unet(
                    latent_model_input.flatten(0, 1).to(device),
                    t_vec.flatten(0, 1).to(device),
                )[0]
            else:
                chunks = torch.arange(parallel_len).tensor_split(num_consumers)
                num_chunks = min(parallel_len, num_consumers)

                for idx1 in range(num_chunks):
                    mp_queues[0].put(
                        (
                        latent_model_input, t_vec,  chunks[idx1], idx1, begin_idx)
                    )

                model_output = [None for _ in range(num_chunks)]

                for _ in range(num_chunks):
                    ret = mp_queues[1].get()
                    model_output_chunk, idx = ret
                    model_output[idx] = model_output_chunk.to(device)
                    del ret

                model_output = torch.cat(model_output)

            # end1.record()

            # # Waits for everything to finish running
            # torch.cuda.synchronize()

            # print(begin_idx, end_idx, flush=True)
            # elapsed = start1.elapsed_time(end1)
            # elapsed_per_t = elapsed / parallel_len
            # print("model parallel elapsed time:", elapsed, "elapsed_per_t:", elapsed_per_t)

            per_latent_shape = model_output.shape[1:]
            print(model_output.shape, flush=True)
            model_output = model_output.reshape(
                parallel_len * batch_size , *per_latent_shape
            )
            print(model_output.shape, flush=True)
            block_latents_denoise = self.scheduler.batch_step_no_noise(
                model_output=model_output,
                timesteps=block_t.flatten(0, 1),
                sample=block_latents.flatten(0, 1),
            ).reshape(block_latents.shape)

            # back to shape (parallel_dim, batch_size, ...)
            # now we want to add the pre-sampled noise
            # parallel sampling algorithm requires computing the cumulative drift from the beginning
            # of the window, so we need to compute cumulative sum of the deltas and the pre-sampled noises.
            delta = block_latents_denoise - block_latents
            cumulative_delta = torch.cumsum(delta, dim=0)
            cumulative_noise = torch.cumsum(noise_array[begin_idx:end_idx], dim=0)

            # if we are using an ODE-like scheduler (like DDIM), we don't want to add noise
            if self.scheduler._is_ode_scheduler:
                cumulative_noise = 0

            block_latents_new = latents_time_evolution_buffer[begin_idx][None,] + cumulative_delta + cumulative_noise
            cur_error_vec = (block_latents_new.float() - latents_time_evolution_buffer[begin_idx + 1:end_idx + 1].float()).reshape(
                parallel_len, batch_size , -1)
            cur_error = torch.linalg.norm(cur_error_vec, dim=-1).pow(2)
            error_ratio = cur_error * inverse_variance_norm[begin_idx + 1:end_idx + 1]

            # find the first index of the vector error_ratio that is greater than error tolerance
            # we can shift the window for the next iteration up to this index
            error_ratio = torch.nn.functional.pad(
                error_ratio, (0, 0, 0, 1), value=1e9
            )  # handle the case when everything is below ratio, by padding the end of parallel_len dimension
            any_error_at_time = torch.max(error_ratio > scaled_tolerance, dim=1).values.int()
            ind = torch.argmax(any_error_at_time).item()

            # compute the new begin and end idxs for the window
            new_begin_idx = begin_idx + min(1 + ind, parallel)
            new_end_idx = min(new_begin_idx + parallel, len(self.scheduler.timesteps))

            # store the computed latents for the current window in the global buffer
            latents_time_evolution_buffer[begin_idx + 1:end_idx + 1] = block_latents_new
            # initialize the new sliding window latents with the end of the current window,
            # should be better than random initialization
            latents_time_evolution_buffer[end_idx:new_end_idx + 1] = latents_time_evolution_buffer[end_idx][None,]

            begin_idx = new_begin_idx
            end_idx = new_end_idx

            stats_pass_count += 1
            stats_flop_count += parallel_len * block_latents_new.size()[1]

            # end2.record()
            # # Waits for everything to finish running
            # torch.cuda.synchronize()

            print(f"begin_idx：{begin_idx}, end_idx:{end_idx}", flush=True)
            # elapsed1 = start2.elapsed_time(end2)
            # elapsed_per_t1 = (elapsed1 - elapsed) / parallel_len
            # print("batch computing elapsed time:", elapsed1, elapsed1 - elapsed, "elapsed_per_t:", elapsed_per_t1)

        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        print("parallel ParaDiGMS_DDPM pipeline end!")
        print("pass count", stats_pass_count)
        print("flop count", stats_flop_count)
        print('time', start.elapsed_time(end))




        stats = {
            'pass_count': stats_pass_count,
            'flops_count': stats_flop_count,
            'time': start.elapsed_time(end),
        }


        image = latents_time_evolution_buffer[-1]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        def process_image(image):
            if output_type == "pil":
                image = self.numpy_to_pil(image)

            # Offload last model to CPU
            if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                print("offload hook", flush=True)
                self.final_offload_hook.offload()

            return image

        image = process_image(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image), stats
