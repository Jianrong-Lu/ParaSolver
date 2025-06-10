from typing import TYPE_CHECKING, List, Optional, Union

import torch
from diffusers.pipelines import DDIMPipeline
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor

if TYPE_CHECKING:
    import torch.jit
    import torch.jit._state


class SequentialDDIMDiffusionPipeline(DDIMPipeline):
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
    def sequential_paradigms_forward(
            self,
            batch_size: int = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            eta: float = 0.0,
            num_inference_steps: int = 50,
            use_clipped_model_output: Optional[bool] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
    ):

        print("Sequential ddim pipeline!", flush=True)

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

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        image = randn_tensor(image_shape, generator=generator, device=device, dtype=self.unet.dtype)
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        stats_pass_count = 0
        stats_flop_count = 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()


        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t).sample

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample


            stats_pass_count += 1
            print(image.size())
            stats_flop_count += 1 * image.size()[0]

        print("pass count", stats_pass_count)
        print("flop count", stats_flop_count)
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()

        print("sequential ddim elapsed time:",start.elapsed_time(end))
        stats = {
            'pass_count': stats_pass_count,
            'flops_count': stats_flop_count,
            'time': start.elapsed_time(end),
        }
        print("total elapsed time:",stats['time'])
        print("done", flush=True)

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
