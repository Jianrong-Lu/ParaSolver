import imageio
import numpy as np
from PIL import Image


import torch
import os
import torch.multiprocessing as mp
import pandas as pd

import socket





#Sequential
from diffusers.schedulers import DDIMScheduler,DDPMScheduler
from lsun_parasolver.classical_sequential_ddim import SequentialDDIMDiffusionPipeline
from lsun_parasolver.classical_sequential_ddpm import SequentialDDPMDiffusionPipeline

#ParaDiGMS and ParaSolver Scheduler
from lsun_parasolver.paraddpm_scheduler import ParaDDPMScheduler
from lsun_parasolver.paraddim_scheduler import ParaDDIMScheduler
#ParaDiGMS for LSUN Church
from lsun_parasolver.lsun_paradigms_ddim_mp import ParaDiGMSDDIMDiffusionPipeline
from lsun_parasolver.lsun_paradigms_ddpm_mp import ParaDiGMSDDPMDiffusionPipeline

#ParaSolver for LSUN Church
from lsun_parasolver.lsun_parasolver_ddpm_mp import ParaSolverDDPMDiffusionPipeline
from lsun_parasolver.lsun_parasolver_ddim_mp import ParaSolverDDIMDiffusionPipeline








TOPIC = "lsun_church"
MODEL_ID = "google/ddpm-ema-church-256"
HOME_DIR = "./"



def run(rank, total_ranks, queues,shared_list):
    gups,random_seed, method, SCHEDULER_CONFIGS = shared_list[0]
    model_str = MODEL_ID
    device = torch.device(f"cuda:{gups[0]}" if torch.cuda.is_available() else "cpu")
    method = SCHEDULER_CONFIGS[method][0]
    num_inference_steps = SCHEDULER_CONFIGS[method][1]
    generator = torch.Generator()
    torch_dtype = torch.float16


    if method in ["ParaDiGMS_DDPM", "ParaSolver_DDPM"]:
        scheduler = ParaDDPMScheduler.from_pretrained(model_str,  timestep_spacing="trailing",torch_dtype=torch_dtype)
        scheduler._is_ode_scheduler = SCHEDULER_CONFIGS[method][3]


    elif method in ["ParaDiGMS_DDIM","ParaSolver_DDIM"]:
        scheduler = ParaDDIMScheduler.from_pretrained(model_str, timestep_spacing="trailing",torch_dtype=torch_dtype)
        scheduler._is_ode_scheduler = SCHEDULER_CONFIGS[method][3]


    elif method == "DDPM":
        scheduler = DDPMScheduler.from_pretrained(model_str,  timestep_spacing="trailing",torch_dtype=torch_dtype)
        scheduler._is_ode_scheduler = SCHEDULER_CONFIGS[method][3]
    elif method == "DDIM":
        scheduler = DDIMScheduler.from_pretrained(model_str,  timestep_spacing="trailing",torch_dtype=torch_dtype)
        scheduler._is_ode_scheduler = SCHEDULER_CONFIGS[method][3]

    if method in ["ParaDiGMS_DDPM"]:
        pipe = ParaDiGMSDDPMDiffusionPipeline.from_pretrained(
            model_str, scheduler=scheduler, torch_dtype=torch_dtype,safety_checker=None
        )
    elif method in ["ParaDiGMS_DDIM"]:
        pipe = ParaDiGMSDDIMDiffusionPipeline.from_pretrained(
            model_str, scheduler=scheduler, torch_dtype=torch_dtype,safety_checker=None
        )


    elif method in ["ParaSolver_DDPM"]:
        pipe = ParaSolverDDPMDiffusionPipeline.from_pretrained(
            model_str, scheduler=scheduler, torch_dtype=torch_dtype,safety_checker=None
        )
    elif method in ["ParaSolver_DDIM"]:
        pipe  = ParaSolverDDIMDiffusionPipeline.from_pretrained(
            model_str, scheduler=scheduler, torch_dtype=torch_dtype,safety_checker=None
        )


    elif method in ["DDPM"]:
        pipe = SequentialDDPMDiffusionPipeline.from_pretrained(
            model_str, scheduler=scheduler, torch_dtype=torch_dtype,safety_checker=None
        )
    elif method in ["DDIM"]:
        pipe = SequentialDDIMDiffusionPipeline.from_pretrained(
            model_str, scheduler=scheduler, torch_dtype=torch_dtype,safety_checker=None
        )

    else:
        raise ValueError("not implemented")
    pipe.unet.eval()
    pipe.enable_xformers_memory_efficient_attention()
    if rank != -1:
        if method not in ["DDPM",  "DDIM"]:
            pipe = pipe.to(f"cuda:{gups[rank]}")
            # pipe.paradigms_forward = types.MethodType(paradigms_forward_worker, pipe)
            pipe.paradigms_forward_worker(mp_queues=queues, device=f"cuda:{gups[rank]}")
    else:
        pipe = pipe.to(device)
        # ngpu_sweep = [(i + 1) for i in range(torch.cuda.device_count())]
        ngpu_sweep = [x + 1 for x in gups]
        ngpu_sweep = ngpu_sweep[-1:]
        if method  in ["DDPM",  "DDIM"]:#Not a parallel method, only use one GPU.
            parallel_sweep = [1]
        else: #parallel method
            num_max = min(num_inference_steps,100)
            if num_inference_steps > 100:
                parallel_sweep =  [num_max - (i-1) for i in range(1, num_max + 1, 2)]
            else:
                parallel_sweep =  [num_max - (i-1) for i in range(1, num_max + 1, 1)]
            parallel_sweep = [8]


        # prepare a dict for storing the results
        time_results = {scfg[0]: torch.zeros(max(ngpu_sweep) + 1, max(parallel_sweep) + 1) for scfg in
                        [SCHEDULER_CONFIGS[method]]}
        pass_results = {scfg[0]: torch.zeros(max(ngpu_sweep) + 1, max(parallel_sweep) + 1) for scfg in
                        [SCHEDULER_CONFIGS[method]]}
        flops_results = {scfg[0]: torch.zeros(max(ngpu_sweep) + 1, max(parallel_sweep) + 1) for scfg in
                         [SCHEDULER_CONFIGS[method]]}




        chunked_prompts_lists = [
        "A highly detailed portrait of an elderly man with wrinkles, wearing a traditional woolen hat, cinematic lighting, 8K, ultra-realistic, photorealistic, depth of field, soft shadows, film grain",
        "A futuristic cityscape at night, neon lights reflecting on wet streets, cyberpunk style, towering skyscrapers with holographic advertisements, ultra-detailed, cinematic atmosphere",
        "Volcanic eruption at night, lava flowing down the mountain, ash clouds illuminated by lightning, dramatic and epic, hyper-realistic",
        "A modern minimalist living room with floor-to-ceiling windows overlooking a forest, Scandinavian design, natural wood and neutral tones, soft daylight",
        "A stunning portrait of an ethereal elf queen with intricate silver jewelry, glowing blue eyes, flowing white hair, soft cinematic lighting, highly detailed, by Alphonse Mucha and Artgerm, 8K.",
        "A mystical enchanted forest at twilight, towering ancient trees with bioluminescent leaves, glowing mushrooms, a crystal-clear stream reflecting the stars, ethereal fog, dreamlike atmosphere, Studio Ghibli style, 4k wallpaper.",
        "A majestic ice dragon perched on a frozen cliff, glowing blue scales, intricate icy horns, breath visible in the cold air, aurora borealis in the background, ultra-detailed fantasy artwork, digital painting by Greg Rutkowski, dramatic lighting, 4k.",
        "An anime-style girl with long pink hair and golden eyes, wearing a futuristic school uniform, cherry blossoms falling in the background, soft pastel colors, vibrant and clean line art,4k."]


        for num_consumers in ngpu_sweep:
            for parallel_id,parallel in enumerate(parallel_sweep):
                for scfg in [SCHEDULER_CONFIGS[method]]:
                    method, num_inference_steps,num_time_subintervals, fname,  tolerance, num_preconditioning_steps = scfg
                    for chunk_id, chunk_prompts in enumerate(chunked_prompts_lists[0:1]):
                            generator.manual_seed(random_seed)
                            print("len(chunk_prompts)", len(chunk_prompts))
                            if method in ["ParaSolver_DDPM", "ParaSolver_DDIM"]:
                                # warmup
                                # generator.manual_seed(random_seed)
                                _, _ = pipe.parasolver_forward(batch_size=1,num_inference_steps=8,
                                                                          num_time_subintervals=num_consumers, num_preconditioning_steps=num_preconditioning_steps
                                                                          parallel=parallel,
                                                                          tolerance=tolerance, output_type="np",
                                                                          full_return=False,
                                                                          mp_queues=queues, device=device,
                                                                          generator=generator,
                                                                          num_consumers=num_consumers
                                                                          )
                                generator.manual_seed(random_seed)
                                output, stats = pipe.parasolver_forward(batch_size=1,num_inference_steps=num_inference_steps,
                                                                          num_time_subintervals=num_time_subintervals, num_preconditioning_steps=num_preconditioning_steps,
                                                                          parallel=parallel,
                                                                          tolerance=tolerance, output_type="np",
                                                                          full_return=False,
                                                                          mp_queues=queues, device=device,
                                                                          generator=generator,
                                                                          num_consumers=num_consumers
                                                                          )


                            elif method in ["ParaDiGMS_DDPM",  "ParaDiGMS_DDIM"]:
                                # warmup
                                generator.manual_seed(random_seed)
                                _, _ = pipe.ParaDiGMS_paradigms_forward(
                                    batch_size=1,
                                    parallel=num_consumers,
                                    num_inference_steps=8,
                                    tolerance=tolerance,
                                    output_type="np",
                                    mp_queues=queues,
                                    generator=generator,
                                    num_consumers=num_consumers)
                                generator.manual_seed(random_seed)
                                output, stats = pipe.ParaDiGMS_paradigms_forward(
                                    batch_size=1,
                                    parallel=parallel,
                                    num_inference_steps=num_inference_steps,
                                    tolerance=tolerance,
                                    output_type="np",
                                    mp_queues=queues,
                                    generator=generator,
                                    num_consumers=num_consumers)

                            elif method in ["DDPM",  "DDIM"]:
                                # warmup
                                generator.manual_seed(random_seed)
                                _, _ = pipe.sequential_paradigms_forward(batch_size=1,
                                                                                  num_inference_steps=2*num_consumers,
                                                                                  output_type="np",
                                                                                  generator=generator
                                                                                  )
                                generator.manual_seed(random_seed)
                                output, stats = pipe.sequential_paradigms_forward(batch_size=1,
                                                                                  num_inference_steps=num_inference_steps,
                                                                                  output_type="np",
                                                                                  generator=generator
                                                                                  )

                            else:
                                raise ValueError(f"not supported method: {method}")


                            torch.cuda.empty_cache()
                            generated_images = output.images
                            one_pil_image = pipe.numpy_to_pil(generated_images[0])[0]
                            image_savepath = f'image_expr_name_CMP_LSUN/num_{num_inference_steps}_N_{num_time_subintervals}_parallel_{parallel}_M_{num_preconditioning_steps}_{method}_tor_{tolerance}/pass_{stats["pass_count"]}_flop_{stats["flops_count"]}_time_{int(stats["time"])}_prt_id_{chunk_id}.png'
                            os.makedirs(os.path.dirname(image_savepath), exist_ok=True)
                            one_pil_image.save(image_savepath) 
                            
                            # store the result tm in a dict with key (ngpu, parallel, scheduler)
                            time_results[method][num_consumers, parallel] = stats['time']
                            pass_results[method][num_consumers, parallel] = stats['pass_count']
                            flops_results[method][num_consumers, parallel] = stats['flops_count']


        # convert results to a dataframe
        stat_dfs = [time_results, pass_results, flops_results]
        stat_names = ['time', 'pass', 'flops']
        method, num_inference_steps,num_time_subintervals, fname,  tolerance, num_preconditioning_steps = scfg
        for stat_df, stat_name in zip(stat_dfs, stat_names):
            for scheduler_name in stat_df:
                df = pd.DataFrame(stat_df[scheduler_name].numpy())
                print(scheduler_name)
                print(df.to_string())

                df_savepath = f'{HOME_DIR}/stats/{TOPIC}/expr_name_CMP_LSUN/{method}_stepnum_{num_inference_steps}_N_{num_time_subintervals}_M_{num_preconditioning_steps}_tolr_{tolerance}/{stat_name}_{scheduler_name}.csv'
                os.makedirs(os.path.dirname(df_savepath), exist_ok=True)
                df.to_csv(df_savepath, index=True)

        # shutdown workers
        for _ in range(total_ranks):
            queues[0].put(None)

def main(gups,random_seed,method,SCHEDULER_CONFIGS):
    torch.autograd.set_detect_anomaly(True)
    mp.set_start_method('spawn', force=True)
    queues = mp.Queue(), mp.Queue(), mp.Queue()
    processes = []

    if  method in ["DDPM","DDIM"]:
        num_processes = 1
    else: num_processes = len(gups)

    with mp.Manager() as manager:
        shared_list = manager.list()
        shared_list.append((gups,random_seed,method,SCHEDULER_CONFIGS))
        for rank in range(-1, num_processes):
            p = mp.Process(target=run, args=(rank, num_processes, queues,shared_list))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()  # wait for all subprocesses to finish

    queues[2].put(None)









# CUDA_VISIBLE_DEVICES=0,2,3 python parareal_v4_sweep_batch_generation.py
if __name__ == "__main__":
    random_seed = 2
    num_inference_steps = 25  # setings: 25 and 50
    num_inference_steps = 50  # setings: 25 and 50
    num_inference_steps = 1000  # setings: 25 and 50


    gpus = [i for i in range(torch.cuda.device_count())]



    method = "ParaSolver_DDPM"
    # method = "ParaDiGMS_DDPM" 
    # method = "DDPM"


    # method = "ParaSolver_DDIM"
    # method = "ParaDiGMS_DDIM"
    # method = "DDIM"



    SCHEDULER_CONFIGS = {
        # method, num_inference_steps,num_time_subintervals, fname,  tolerance,preconditioning_step 
        "DDPM":
            ["DDPM", num_inference_steps, None,f"{HOME_DIR}/imgs/{TOPIC}/stepnum_{num_inference_steps}_tolr_%s/ddpm%s.png",  None, None],
        "ParaDiGMS_DDPM":
            ["ParaDiGMS_DDPM", num_inference_steps,None,f"{HOME_DIR}/imgs/{TOPIC}/stepnum_{num_inference_steps}_tolr_%s/ParaDiGMS_DDPM%s.png",  0.5, None],
        "ParaSolver_DDPM":
            ["ParaSolver_DDPM", num_inference_steps,num_inference_steps,f"{HOME_DIR}/imgs/{TOPIC}/stepnum_{num_inference_steps}_tolr_%s/ParaSolver_DDPM%s.png",  0.55,5],
            #Default to set the number of time subintervals as the number of inference steps as the solving of subintervals is implemented by sequential solvers.

        "DDIM":
            ["DDIM", num_inference_steps,None,f"{HOME_DIR}/imgs/{TOPIC}/stepnum_{num_inference_steps}_tolr_%s/ddim%s.png",  None, None],
        "ParaDiGMS_DDIM":
            ["ParaDiGMS_DDIM", num_inference_steps,None,f"{HOME_DIR}/imgs/{TOPIC}/stepnum_{num_inference_steps}_tolr_%s/ParaDiGMS_DDIM%s.png", 0.001, None],
        "ParaSolver_DDIM":
            ["ParaSolver_DDIM", num_inference_steps,num_inference_steps,f"{HOME_DIR}/imgs/{TOPIC}/stepnum_{num_inference_steps}_tolr_%s/ParaSolver_DDIM%s.png",  0.005,1],
            #Default to set the number of time subintervals as the number of inference steps as the solving of subintervals is implemented by sequential solvers.
    }



    main(gpus,random_seed, method,SCHEDULER_CONFIGS)