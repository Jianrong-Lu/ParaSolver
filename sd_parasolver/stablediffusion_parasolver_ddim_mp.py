
import inspect
from typing import (TYPE_CHECKING, Dict, Iterator, List, Optional, Sequence, 
                    Set, TypeVar, Union, cast, Any, Callable, Tuple)

import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils import deprecate
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import (CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, 
                         CLIPVisionModelWithProjection)

if TYPE_CHECKING:
    import torch.jit._state

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

class ParaSolverDDIMStableDiffusionPipeline(StableDiffusionPipeline):
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
        self.initial_latents = dict()
        self.prompt_embeds = None
        self.timestep_cond = None
        self.added_cond_kwargs = None
        self.batch_size = None
        self.num_images_per_prompt = None
        self.extra_step_kwargs = None
        self.coarse_num_warmup_steps = None
        self.callback = None
        self.callback_steps = None
        self.predicted_original_sample = None

    def initialize_pipeline_variables(self):
        #pip initialize
        self.initial_latents = dict()
        self.prompt_embeds = None
        self.timestep_cond = None
        self.added_cond_kwargs = None
        self.batch_size = None
        self.num_images_per_prompt = None
        self.extra_step_kwargs = None
        self.coarse_num_warmup_steps = None
        self.callback = None
        self.callback_steps = None
        self.predicted_original_sample = None
        #scheduler
        self.scheduler.base_timesteps =  None
        self.scheduler.base_sigmas = None

        self.scheduler.coarse_timesteps = None
        self.scheduler.coarse_timesteps_sigmas = None

        self.scheduler.initial_timesteps = None
        self.scheduler.initial_timesteps_sigmas = None

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

            (latent_model_input, t_vec, block_prompt_embeds, cross_attention_kwargs, chunk, idx, begin_idx) = ret
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
    def preparation_for_diffusion(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
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

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop

        self._num_timesteps = len(timesteps)

        # coarse_timesteps = None
        self.initial_latents[timesteps[0].item()] = latents
        self.prompt_embeds = prompt_embeds
        self.timestep_cond = timestep_cond
        self.batch_size = batch_size
        self.num_images_per_prompt = num_images_per_prompt
        self.extra_step_kwargs = extra_step_kwargs
        self.added_cond_kwargs = added_cond_kwargs
        # self.coarse_num_warmup_steps = num_warmup_steps
        self.callback = callback
        self.callback_steps = callback_steps


    @torch.no_grad()
    def parasolver_forward(
            self,
            prompt: Union[str, List[str]] = None,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_images_per_prompt: Optional[int] = 1,
            num_inference_steps: int = 50,
            num_time_subintervals: int = 50,
            num_preconditioning_steps: int = 0,
            parallel: int = 10,
            tolerance: int = 0.05,
            guidance_scale: float = 7.5,
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
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds`. 
                Ignored when not using guidance (i.e., if `guidance_scale` is less than `1`).
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of 
                slower inference.
            num_time_subintervals (`int`, *optional*, defaults to the number of inference steps):
                The number of time subintervals. The number of time subintervals defaults to the number of inference steps.
                  These subintervals can be processed in parallel or sequentially, though the current implementation only 
                  supports sequential iteration. Thus, when the number of subintervals equals the number of inference steps, 
                  each subinterval contains just a single timestep.
            num_preconditioning_steps (`int`, *optional*, defaults to 0):
                The number of preconditioning steps for initialization. Increasing this value typically reduces the number 
                of parallel iterations needed but slows down inference in terms of wall-clock time.
            parallel (`int`, *optional*, defaults to 10):
                The size of the sliding window for parallel processing.
            tolerance (`int`, *optional*, defaults to 10):
                The tolerance for stopping.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). 
                Higher values encourage image generation closely linked to the text prompt, usually at the expense of 
                lower image quality.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. 
                Only applies to [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) 
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image 
                generation. Can be used to tweak the same generation with different prompts. If not provided, a 
                latents tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. 
                If not provided, text embeddings will be generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. If not provided, negative_prompt_embeds will be generated 
                from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): 
                `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead 
                of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will 
                be called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback 
                will be called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under 
                `self.processor` in [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            full_return (`bool`, *optional*, defaults to `False`):
                Whether to return additional information along with the generated images.
            mp_queues (`torch.FloatTensor`, *optional*):
                Queues for multiprocessing, if applicable.
            device (`str`, *optional*):
                The device to perform the computation on, e.g., 'cpu' or 'cuda'.
            num_consumers (`int`, *optional*):
                Number of consumer processes for parallel processing.

            Examples:

            Returns:
                [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
                When returning a tuple, the first element is a list with the generated images, and the second element is a
                list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
                (nsfw) content, according to the `safety_checker`.
            """
        print("parallel pipeline!", flush=True)


        #1.Reset all Settings to their initial state
        self.initialize_pipeline_variables()

        #2.Set base time steps and coarse time steps as well as related parameters
        num_time_subintervals = min(num_time_subintervals,num_inference_steps)
        num_preconditioning_steps = min(num_time_subintervals, num_preconditioning_steps)
        parallel = min(parallel, num_time_subintervals)
        self.scheduler.set_timesteps(num_inference_steps)   
        self.scheduler.set_coarse_and_fine_timesteps(num_time_subintervals=num_time_subintervals,num_preconditioning_steps=num_preconditioning_steps) #set base timesteps（corresponding to sequential timesteps), coarse timesteps (the starting timesteps of subintervals), initial timesteps (the preconditioning timesteps using model)


        #3. Using coarse scheduler setting for diffusion preparation
        self.scheduler.timesteps = self.scheduler.coarse_timesteps
        self.scheduler.num_inference_steps =len(self.scheduler.coarse_timesteps)
        self.preparation_for_diffusion(prompt=prompt,generator=generator,num_inference_steps=num_time_subintervals,timesteps=self.scheduler.coarse_timesteps)

        #4. Prepare_noise
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
                generator.manual_seed(cur_t)# We fix the random seed for each timestep to maintain determinism during parallel iteration.
                variance_noise = randn_tensor(
                    self.initial_latents[999].shape, generator=generator, device=self.initial_latents[999].device, dtype=self.initial_latents[999].dtype
                )
                variance = std_dev_t * variance_noise
                noise_array[j] = variance.clone()

        #5. Initialize initial  points of all subintervals
        self.scheduler.timesteps = self.scheduler.coarse_timesteps
        # # self.scheduler.sigmas = self.scheduler.coarse_timesteps_sigmas
        self.scheduler.num_inference_steps = num_time_subintervals
        stats_pass_count, stats_flop_count,initial_para_dur = self.preconditioning_initialization(num_inference_steps=self.scheduler.num_inference_steps,
                                        timesteps=self.scheduler.timesteps,
                                        generator=generator,
                                        eta=eta,
                                        image=list(self.initial_latents.values())[0],
                                        num_preconditioning_steps=num_preconditioning_steps)



        # 6. Parallel parameters initializatiom
        timesteps = self.scheduler.coarse_timesteps  
        latents_time_evolution_buffer = torch.stack(list(self.initial_latents.values()))  
        time_evolution_buffer_matrix = self.scheduler.fine_timesteps_matrix  
        block_prompt_embeds_total = torch.stack([self.prompt_embeds] * num_time_subintervals)
        self.initial_latents.clear()


        #7.Prepare stop parameters
        inverse_variance_norm = 1. / torch.tensor(
            [self.scheduler._get_variance_v1(timesteps[j].item(),timesteps[j + 1].item() if j < len(timesteps) - 1 else 0)
             for j in list(range(len(timesteps))) + [len(timesteps) - 1]]).to(torch.float32).to(device)


        latent_dim = noise_array[0, 0].numel()
        inverse_variance_norm = inverse_variance_norm[:, None] / latent_dim
        scaled_tolerance = (tolerance ** 2)


        #8.Set fine scheduler  for parallel iteration, such that  the scheduler can accurately find the previous timesteps or sigmas
        self.scheduler.timesteps = self.scheduler.base_timesteps
        self.scheduler.num_inference_steps = len(self.scheduler.base_timesteps)
        # self.scheduler.sigmas = self.scheduler.base_sigmas
        # self.scheduler._step_index = None


        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        end_i = parallel
        i = 0 #begin idx 
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
                block_prompt_embeds = block_prompt_embeds_total[:parallel_len]

                block_t = timestep_list[begin_idx][indices_started][:, None].repeat(1, self.batch_size * self.num_images_per_prompt).to(
                    block_latents_list.device)
                t_vec = block_t
                if self.do_classifier_free_guidance:
                    t_vec = t_vec.repeat(1, 2)

                # expand the latents if we are doing classifier free guidance
                block_latents = block_latents_list[indices_started]
                latent_model_input = (torch.cat([block_latents] * 2, dim=1) if self.do_classifier_free_guidance else block_latents)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t_vec)

                if parallel_len <= 2 or num_consumers == 1:
                    # #print("parallel_len", parallel_len, "num_consumers", num_consumers)
                    model_output = self.unet(
                        latent_model_input.flatten(0, 1).to(device),
                        t_vec.flatten(0, 1).to(device),
                        encoder_hidden_states=block_prompt_embeds.flatten(0, 1).to(device),
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        return_dict=False,
                    )[0]
                else:
                    chunks = torch.arange(parallel_len).tensor_split(num_consumers)
                    num_chunks = min(parallel_len, num_consumers)

                    for idx1 in range(num_chunks):
                        mp_queues[0].put(
                            (latent_model_input, t_vec, block_prompt_embeds, self.cross_attention_kwargs, chunks[idx1], idx1, begin_idx)
                        )

                    model_output = [None for _ in range(num_chunks)]

                    for _ in range(num_chunks):
                        ret = mp_queues[1].get()
                        model_output_chunk, idx = ret
                        model_output[idx] = model_output_chunk.to(device)
                        del ret

                    model_output = torch.cat(model_output)



                per_latent_shape = model_output.shape[1:]
                if self.do_classifier_free_guidance:
                    model_output = model_output.reshape(
                        parallel_len, 2, self.batch_size * self.num_images_per_prompt,
                        *per_latent_shape
                    )
                    noise_pred_uncond, noise_pred_text = model_output[:, 0], model_output[:, 1]
                    model_output = noise_pred_uncond + self.guidance_scale * (
                            noise_pred_text - noise_pred_uncond)
                model_output = model_output.reshape(
                    parallel_len * self.batch_size * self.num_images_per_prompt,
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
                parallel_len_outer, self.batch_size * self.num_images_per_prompt, -1)

            #Parallel correction of initial points.
            latents_time_evolution_buffer[i + 1:end_i + 1] = block_latents_new


            cur_error = torch.linalg.norm(cur_error_vec, dim=-1).pow(2)
            error_ratio = cur_error * inverse_variance_norm[i + 1:end_i + 1]
            error_ratio = torch.nn.functional.pad(
                error_ratio, (0, 0, 0, 1), value=1e9
            )  # handle the case when everything is below ratio, by padding the end of parallel_len dimension
            any_error_at_time = torch.max(error_ratio > scaled_tolerance, dim=1).values.int()
            ind = torch.argmax(any_error_at_time).item()
            # compute the new begin and end indexs for the window
            new_i = i + min(1 + ind, parallel)
            new_end_i = min(new_i + parallel, num_time_subintervals)

            #initialize new sample points outside current window via approximated reverse process
            if end_i < num_time_subintervals-1:
                X_0 = self.predicted_original_sample[-1]
                self.scheduler.timesteps = self.scheduler.coarse_timesteps
                # self.scheduler.sigmas = self.scheduler.coarse_timesteps_sigmas
                self.scheduler.num_inference_steps = num_time_subintervals
                # self.scheduler._step_index = None
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
                self.scheduler.timesteps = self.scheduler.base_timesteps
                # self.scheduler.sigmas = self.scheduler.base_sigmas
                self.scheduler.num_inference_steps = len(self.scheduler.base_timesteps)
                # self.scheduler._step_index = None
                self.initial_latents.clear()


            i = new_i
            end_i = new_end_i
            k = k + 1
            # if k >= num_time_subintervals: 
            #     self.save_all_process_image(num_time_subintervals,num_preconditioning_steps,output_type,tolerance,num_inference_steps,device,latents_time_evolution_buffer[:],k,i,end_i,parallel,"ParaSolver_DDIM")
            #     break
            # else:
            #     self.save_all_process_image(num_time_subintervals,num_preconditioning_steps,output_type,tolerance,num_inference_steps,device,latents_time_evolution_buffer[:i],k,i,end_i,parallel,"ParaSolver_DDIM")
            print(f"i={i}, k={k}, ind={ind}, stride={min(1 + ind, end_i)}")


        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()

        print("pass count", stats_pass_count)
        print("flop count", stats_flop_count)
        print("initial elapsed time:", initial_para_dur)
        print("noise elapsed time:", noises_dur_time)
        print("update elapsed time:", update_dur_time)
        print("parallel elapsed time:",start.elapsed_time(end) + initial_para_dur - noises_dur_time- update_dur_time)
        stats = {
            'pass_count': stats_pass_count,
            'flops_count': stats_flop_count,
            'time': start.elapsed_time(end) + initial_para_dur - noises_dur_time- update_dur_time,
        }
        print("total elapsed time:",stats['time'])
        print("done", flush=True)


        latents = latents_time_evolution_buffer[-1]
        def process_image(latents):
            if output_type == "latent":
                image = latents
                has_nsfw_concept = None
            elif output_type == "pil":
                # Post-processing
                # #print("post-processing", flush=True)
                image = self.decode_latents(latents)

                # Run safety checker
                # #print("safety check", flush=True)
                # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
                image, has_nsfw_concept = image, False

                #  Convert to PIL
                # #print("conver to PIL", flush=True)
                image = self.numpy_to_pil(image)
            else:
                # Post-processing
                image = self.decode_latents(latents)

                # Run safety checker
                image, has_nsfw_concept = self.run_safety_checker(image, device, self.prompt_embeds.dtype)

            # Offload last model to CPU
            if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                #print("offload hook", flush=True)
                self.final_offload_hook.offload()

            return image, has_nsfw_concept



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
        """Approximates the diffusion process using sequential sampling with epsilon prediction.

        Args:
            num_inference_steps (int): Total number of denoising steps.
            timesteps (torch.FloatTensor): Tensor of timesteps for the diffusion process.
            image (torch.FloatTensor): Input latent image tensor (4D: [batch, channels, height, width]).
            generator (Optional[Union[torch.Generator, List[torch.Generator]]], optional): Random number generator(s). Defaults to None.
            eta (float, optional): DDIM noise parameter (0=deterministic). Defaults to 0.0.
            use_clipped_model_output (Optional[bool], optional): Whether to clip model output. Defaults to None.
            num_preconditioning_steps (int, optional): Number of preconditioning steps to run with full model guidance. Defaults to 0.

        Returns:
            Tuple[int, int, float]:
                - stats_pass_count: Number of UNet forward passes performed
                - stats_flop_count: Estimated FLOP count (channel-dependent)
                - initial_para_dur: Initial parallel processing duration in milliseconds

        Note:
            - When num_preconditioning_steps > 0, first num_preconditioning_steps steps use sequential iteration, remaining steps use approximated originial sample for generation
            - Progressively stores intermediate latents in self.initial_latents
            - CUDA timing events are used for performance measurement
        """
        with torch.no_grad():
            latents = image
            initial_para_dur = 0
            stats_pass_count = 0
            stats_flop_count = 0
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if num_preconditioning_steps>0:
                        if i < num_preconditioning_steps:# perform num_preconditioning_steps preconditioning steps via sequential sampling
                            if self.interrupt:
                                continue
                            # expand the latents if we are doing classifier free guidance
                            latent_model_input = torch.cat(
                                [latents] * 2) if self.do_classifier_free_guidance else latents
                            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                            noise_pred = self.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=self.prompt_embeds,
                                timestep_cond=self.timestep_cond,
                                cross_attention_kwargs=self.cross_attention_kwargs,
                                added_cond_kwargs=self.added_cond_kwargs,
                                return_dict=False,
                            )[0]

                            # perform guidance
                            if self.do_classifier_free_guidance:
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + self.guidance_scale * (
                                            noise_pred_text - noise_pred_uncond)

                            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text,
                                                               guidance_rescale=self.guidance_rescale)

                            # compute the previous noisy sample x_t -> x_t-1
                            generator = generator.manual_seed(t.item())
                            latents, x_0 = self.scheduler.step(
                                noise_pred, t, latents, eta=eta, use_clipped_model_output=True,
                                generator=generator, return_dict=True
                            ).values()
                            latents = latents.to(noise_pred.dtype)

                            # model_output = noise_pred
                            self.predicted_original_sample = x_0.clone()

                            stats_pass_count += 1
                            stats_flop_count += 1 * latents.size()[1]

                            end.record()
                            torch.cuda.synchronize()
                            initial_para_dur = start.elapsed_time(end)

                        else:#Predicted the other initial points via predicetd reverse process
                            generator = generator.manual_seed(t.item())
                            latents, _ = self.scheduler.ddim_step(
                                self.predicted_original_sample, t, latents, eta=eta, use_clipped_model_output=True,
                                generator=generator, return_dict=True
                            ).values()
                    else:
                        model_output = randn_tensor(
                            latents.shape, generator=generator, device=latents.device,
                            dtype=latents.dtype) 
                        generator = generator.manual_seed(t.item())
                        latents, _ = self.scheduler.step(
                            model_output, t, latents, eta=eta, use_clipped_model_output=True,
                            generator=generator, return_dict=True
                        ).values()
                    progress_bar.update()
                    if i == num_inference_steps - 1:
                        self.initial_latents[-1] = latents
                    else:
                        self.initial_latents[timesteps[i+1].item()] = latents
        return  stats_pass_count,stats_flop_count,initial_para_dur

    @torch.no_grad()
    def initialize_points_outside_current_window(
        self,
        num_inference_steps: int,
        timesteps: torch.FloatTensor,
        image: torch.FloatTensor,
        X_0: torch.FloatTensor,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        use_clipped_model_output: Optional[bool] = None,
        start_point: int = 0,
        end_point: int = 0
    ) -> float:
        """Updates latent representations for non-convergence timesteps using DDIM steps.

        This function performs DDIM sampling steps between specified start and end points
        to refine latent representations that failed to converge during parallel processing.

        Args:
            num_inference_steps (int): Total number of diffusion steps (for progress bar).
            timesteps (torch.FloatTensor): 1D tensor of diffusion timesteps.
            image (torch.FloatTensor): Current latent image tensor (4D: [batch, channels, height, width]).
            X_0 (torch.FloatTensor): Predicted clean image tensor (same shape as `image`).
            generator (Optional[Union[torch.Generator, List[torch.Generator]]], optional): 
                Random number generator(s) for reproducibility. Defaults to None.
            eta (float, optional): DDIM noise parameter (0=deterministic). Defaults to 0.0.
            use_clipped_model_output (Optional[bool], optional): Whether to clip model outputs. Defaults to None.
            start_point (int, optional): First timestep index to update (inclusive). Defaults to 0.
            end_point (int, optional): Last timestep index to update (exclusive). Defaults to 0.

        Returns:
            float: Execution time in milliseconds for the update operations.

        Note:
            - Updates self.initial_latents dictionary with refined latent values
            - Uses CUDA events for precise timing measurement
            - Progress bar shows overall steps while only processing [start_point, end_point) range
            - For DDIM step details, see scheduler.step implementation
        """
        with torch.no_grad():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if i >= start_point and i < end_point:
                        # print("initialize_points_outside_current_window",i,start_point,end_point)
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
    
    def save_all_process_image(self,num_time_subintervals,num_preconditioning_steps,output_type,tolerance,num_inference_steps,device,latents_time_evolution_buffer,k,unconvegred_i,end_i,parallel,method):
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
                image, has_nsfw_concept = image, False

            # Offload last model to CPU
            if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                print("offload hook", flush=True)
                self.final_offload_hook.offload()

            return image, has_nsfw_concept
        file_savepath = f'{method}/iteration_results.txt'
        import os
        os.makedirs(os.path.dirname(file_savepath), exist_ok=True)
        # Save unconvegred_i and end_i to a text file
        with open(file_savepath, 'a') as f:  # 'a' mode appends to the file
            f.write(f"Iteration {k}: unconvegred_i={unconvegred_i}, end_i={end_i}\n")
        for i, latents in enumerate(latents_time_evolution_buffer):
            (image, has_nsfw_concept) = process_image(latents)
            generated_images = image
            one_pil_image = self.numpy_to_pil(generated_images[0])[0]
            image_savepath = f'{method}/parallel_{k}/In_Expr_Image_ppt_vcm_SD_num_{num_inference_steps}_NS_{num_time_subintervals}_NPS_{num_preconditioning_steps}_tor_{tolerance}_parallel_{parallel}_unconvegred_i_{unconvegred_i}_end_i_{end_i}_pass_{k}_{i+1}.png'
            import os
            os.makedirs(os.path.dirname(image_savepath), exist_ok=True)
            one_pil_image.save(image_savepath)  