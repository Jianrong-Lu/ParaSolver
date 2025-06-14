
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch.jit._state

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
import inspect
from typing import TYPE_CHECKING, Dict, List, Optional,  Union,Any,Callable
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

class ParaStableDiffusionPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            image_encoder,
            requires_safety_checker
        )


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
            # import os

            # pid = os.getpid()
            # print("Process ID (PID):", pid,device)
            (latent_model_input, t_vec, block_prompt_embeds, cross_attention_kwargs, chunk, idx, begin_idx) = ret
            # print("begin_idx",begin_idx,"chunk_idx",idx,"chunk",chunk, "Process ID (PID):", pid,device)
            model_output_chunk = self.unet(
                latent_model_input[chunk].flatten(0, 1).to(device),
                t_vec[chunk].flatten(0, 1).to(device),
                encoder_hidden_states=block_prompt_embeds[chunk].flatten(0, 1).to(device),
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            del ret

            mp_queues[1].put(
                (model_output_chunk, idx),
            )

    @torch.no_grad()
    def paradigms_forward(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            parallel: int = 10,
            tolerance: float = 0.1,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            full_return: bool = False,
            mp_queues: Optional[torch.FloatTensor] = None,
            device: Optional[str] = None,
            num_consumers: Optional[int] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        print("parallel pipeline!", flush=True)

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        scheduler = self.scheduler

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        # print(scheduler.timesteps)
        stats_pass_count = 0
        stats_flop_count = 0
        parallel = min(parallel, len(scheduler.timesteps))

        begin_idx = 0
        end_idx = parallel
        latents_time_evolution_buffer = torch.stack([latents] * (len(scheduler.timesteps) + 1))

        # We specify the error tolerance as a ratio of the scheduler's noise magnitude. We similarly compute the error tolerance
        # outside of the denoising loop to avoid recomputing it at every step.
        # We will be dividing the norm of the noise, so we store its inverse here to avoid a division at every step.
        noise_array = torch.zeros_like(latents_time_evolution_buffer)
        for j in range(len(scheduler.timesteps)):
            base_noise = torch.randn_like(latents)
            noise = (self.scheduler._get_variance(scheduler.timesteps[j]) ** 0.5) * base_noise
            noise_array[j] = noise.clone()

        # We specify the error tolerance as a ratio of the scheduler's noise magnitude. We similarly compute the error tolerance
        # outside of the denoising loop to avoid recomputing it at every step.
        # We will be dividing the norm of the noise, so we store its inverse here to avoid a division at every step.
        inverse_variance_norm = 1. / torch.tensor(
            [scheduler._get_variance(scheduler.timesteps[j]) for j in range(len(scheduler.timesteps))] + [0]).to(
            noise_array.device)
        latent_dim = noise_array[0, 0].numel()
        inverse_variance_norm = inverse_variance_norm[:, None] / latent_dim

        scaled_tolerance = (tolerance ** 2)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        k = 1
        while begin_idx < len(scheduler.timesteps):
            # these have shape (parallel_dim, 2*batch_size, ...)
            # parallel_len is at most parallel, but could be less if we are at the end of the timesteps
            # we are processing batch window of timesteps spanning [begin_idx, end_idx)
            parallel_len = end_idx - begin_idx

            block_prompt_embeds = torch.stack([prompt_embeds] * parallel_len)
            block_latents = latents_time_evolution_buffer[begin_idx:end_idx]
            block_t = scheduler.timesteps[begin_idx:end_idx, None].repeat(1, batch_size * num_images_per_prompt)
            t_vec = block_t
            if do_classifier_free_guidance:
                t_vec = t_vec.repeat(1, 2)

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([block_latents] * 2, dim=1) if do_classifier_free_guidance else block_latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t_vec)

            start1 = torch.cuda.Event(enable_timing=True)
            end1 = torch.cuda.Event(enable_timing=True)

            start1.record()

            if parallel_len <= 2 or num_consumers == 1:
                print("parallel_len",parallel_len,"num_consumers",num_consumers)
                model_output = self.unet(
                    latent_model_input.flatten(0, 1).to(device),
                    t_vec.flatten(0, 1).to(device),
                    encoder_hidden_states=block_prompt_embeds.flatten(0, 1).to(device),
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
            else:
                chunks = torch.arange(parallel_len).tensor_split(num_consumers)
                num_chunks = min(parallel_len, num_consumers)

                for i in range(num_chunks):
                    mp_queues[0].put(
                        (
                        latent_model_input, t_vec, block_prompt_embeds, cross_attention_kwargs, chunks[i], i, begin_idx)
                    )

                model_output = [None for _ in range(num_chunks)]

                for _ in range(num_chunks):
                    ret = mp_queues[1].get()
                    model_output_chunk, idx = ret
                    model_output[idx] = model_output_chunk.to(device)
                    del ret

                model_output = torch.cat(model_output)

            end1.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()

            print(begin_idx, end_idx, flush=True)
            elapsed = start1.elapsed_time(end1)
            elapsed_per_t = elapsed / parallel_len
            # print(elapsed, elapsed_per_t)

            per_latent_shape = model_output.shape[1:]
            if do_classifier_free_guidance:
                model_output = model_output.reshape(
                    parallel_len, 2, batch_size * num_images_per_prompt, *per_latent_shape
                )
                noise_pred_uncond, noise_pred_text = model_output[:, 0], model_output[:, 1]
                model_output = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            model_output = model_output.reshape(
                parallel_len * batch_size * num_images_per_prompt, *per_latent_shape
            )

            block_latents_denoise = scheduler.batch_step_no_noise(
                model_output=model_output,
                timesteps=block_t.flatten(0, 1),
                sample=block_latents.flatten(0, 1),
                # **extra_step_kwargs,
            ).reshape(block_latents.shape)

            # back to shape (parallel_dim, batch_size, ...)
            # now we want to add the pre-sampled noise
            # parallel sampling algorithm requires computing the cumulative drift from the beginning
            # of the window, so we need to compute cumulative sum of the deltas and the pre-sampled noises.
            delta = block_latents_denoise - block_latents
            cumulative_delta = torch.cumsum(delta, dim=0)
            cumulative_noise = torch.cumsum(noise_array[begin_idx:end_idx], dim=0)

            # if we are using an ODE-like scheduler (like DDIM), we don't want to add noise
            if scheduler._is_ode_scheduler:
                cumulative_noise = 0

            block_latents_new = latents_time_evolution_buffer[begin_idx][None,] + cumulative_delta + cumulative_noise
            cur_error_vec = (block_latents_new - latents_time_evolution_buffer[begin_idx + 1:end_idx + 1]).reshape(
                parallel_len, batch_size * num_images_per_prompt, -1)
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
            new_end_idx = min(new_begin_idx + parallel, len(scheduler.timesteps))

            # store the computed latents for the current window in the global buffer
            latents_time_evolution_buffer[begin_idx + 1:end_idx + 1] = block_latents_new
            # self.save_process_image(1, 1, output_type, tolerance, num_inference_steps, device,
            #                                                 latents_time_evolution_buffer, k)

            # initialize the new sliding window latents with the end of the current window,
            # should be better than random initialization
            latents_time_evolution_buffer[end_idx:new_end_idx + 1] = latents_time_evolution_buffer[end_idx][None,]

            begin_idx = new_begin_idx
            end_idx = new_end_idx
            k = k + 1
            stats_pass_count += 1
            stats_flop_count += parallel_len*block_latents_new.size()[1]


        # for (k,latents) in enumerate(latents_time_evolution_buffer):
        #     if k % 5==0:
        #         self.save_process_image(1, 1, output_type, tolerance, num_inference_steps, device,
        #                         latents, k)
        latents = latents_time_evolution_buffer[-1]
        print("pass count", stats_pass_count)
        print("flop count", stats_flop_count)

        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        print(start.elapsed_time(end))
        print("done", flush=True)



        def process_image(latents):
            if output_type == "latent":
                image = latents
                has_nsfw_concept = None
            elif output_type == "pil":
                # 8. Post-processing
                # print("post-processing", flush=True)
                image = self.decode_latents(latents)

                # 9. Run safety checker
                # print("safety check", flush=True)
                # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
                image, has_nsfw_concept = image, False

                # 10. Convert to PIL
                # print("conver to PIL", flush=True)
                image = self.numpy_to_pil(image)
            else:
                # 8. Post-processing
                image = self.decode_latents(latents)

                # 9. Run safety checker
                image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # Offload last model to CPU
            if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                print("offload hook", flush=True)
                self.final_offload_hook.offload()

            return image, has_nsfw_concept

        stats = {
            'pass_count': stats_pass_count,
            'flops_count': stats_flop_count,
            'time': start.elapsed_time(end),
        }

        if full_return:
            output = [process_image(latents) for latents in latents_time_evolution_buffer]

            if not return_dict:
                return [(image, has_nsfw_concept) for (image, has_nsfw_concept) in output]

            return [StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept) for
                    (image, has_nsfw_concept) in output]
        else:
            (image, has_nsfw_concept) = process_image(latents)

            if not return_dict:
                return (image, has_nsfw_concept)

            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), stats
