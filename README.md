
<br />
<p align="center">
  <h1 align="center">ParaSolver: A Hierarchical Parallel Integral Solver for Diffusion Models </h1>

  <p align="center">
    <br />
    <a href="https://scholar.google.com/citations?user=k-oe9TUAAAAJ&hl=zh-CN"><strong>Jianrong Lu</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=d1L0KkoAAAAJ&hl=en"><strong>Zhiyu Zhu*</strong></a>
    ·
    <a href="https://sites.google.com/site/junhuihoushomepage/"><strong>Junhui Hou</strong></a>
    ·
  </p>
</p>

<br />

[![Paper](https://img.shields.io/badge/Paper-ICLR%202025-blue)](https://openreview.net/forum?id=your-paper-id)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)






## 📢 News

- **2025-06-10** 📦 🎉 The codes are available now!
- **2025-06-06** ⚙️ 🛠️ We are organizing the code
- **2025-04-22** ⌛ 📅 The code is scheduled to be released between May 16 and June 20  
- **2025-01-23** 📜 🎓 Our paper has been accepted by ICLR 2025!





Official implementation of "ParaSolver: A Hierarchical Parallel Integral Solver for Diffusion Models" (ICLR 2025)

## 📺 Method Overview Video
<video width="100%" controls>
  <source src="parasolver.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## 🌟 Introduction
ParaSolver revolutionizes diffusion model sampling by transforming the sequential inference process into a hierarchical parallel computation. Our method achieves up to **12.1× speedup** without compromising sample quality.

![Method Overview](method_poster.png)

### Key Features:
- **Training-free acceleration**: Works with existing diffusion models
- **Hierarchical parallelism**: Efficiently utilizes computing resources
- **Quality preservation**: Maintains FID/CLIP scores while being faster
- **Flexible integration**: Compatible with DDPM, DDIM, DPMSolver, etc.

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/Jianrong-Lu/ParaSolver.git
cd ParaSolver
pip install -r requirements.txt
```

### Basic Usage 

#### Single GPU
To use ParaSolver to accelerate DDIM on a single GPU, here's a minimal example:

```python
import torch
from sd_parasolver.paraddim_scheduler import ParaDDIMScheduler
from sd_parasolver.stablediffusion_parasolver_ddim_mp import ParaSolverDDIMStableDiffusionPipeline

# Configuration
MODEL_ID = "stabilityai/stable-diffusion-2"
PROMPT = "A highly detailed portrait of an elderly man with wrinkles, wearing a traditional woolen hat, cinematic lighting, 8K, ultra-realistic, photorealistic, depth of field, soft shadows, film grain"
NUM_STEPS = 25
SEED = 42
DEVICE = f"cuda:{0}"
# Initialize scheduler and pipeline
scheduler = ParaDDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler", 
                                            timestep_spacing="trailing",
                                            torch_dtype=torch.float16)
pipe = ParaSolverDDIMStableDiffusionPipeline.from_pretrained(
    MODEL_ID, 
    scheduler=scheduler, 
    torch_dtype=torch.float16,
    safety_checker=None
).to(DEVICE)

# Generate image
generator = torch.Generator(device=DEVICE).manual_seed(SEED)
image = pipe.parasolver_forward(
    PROMPT,
    num_inference_steps=NUM_STEPS,
    num_time_subintervals=NUM_STEPS,  # Typically same as num_inference_steps
    num_preconditioning_steps=1,
    parallel=5,  # Parallelism degree
    tolerance=0.05,
    generator=generator
).images[0]

# Save result
image.save("output.png")
```

#### Multiple GPUs
For multi-GPU usage, please run [`Expr_CMP_SD.py`](Expr_CMP_SD.py) in the repository with appropriate configuration.

## ⚡ Performance Benchmarks

### Stable Diffusion v2 Acceleration
[Stable Diffusion Results](results/sd2_comparison.png)

### LSUN Church Acceleration
[LSUN Church Results](results/lsun_comparison.png)

## 🎨 Visual Results
| Method | Iteration 1 | Iteration 3 | Iteration 5 | Final Result |
|--------|------------|------------|------------|--------------|
| Sequential | [seq1](visuals/seq_1.png) | [seq3](visuals/seq_3.png) | [seq5](visuals/seq_5.png) | [seq_final](visuals/seq_final.png) |
| ParaSolver | [para1](visuals/para_1.png) | [para3](visuals/para_3.png) | [para5](visuals/para_5.png) | [para_final](visuals/para_final.png) |

## 🛠 Advanced Usage


### Integration with Popular Frameworks
Todo


## 📝 Citation
If you use ParaSolver in your research, please cite our paper:
```bibtex
@inproceedings{lu2025parasolver,
  title={ParaSolver: A Hierarchical Parallel Integral Solver for Diffusion Models},
  author={Lu, Jianrong and Zhu, Zhiyu and Hou, Junhui},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```

## 🤝 Contributing
We welcome contributions! 




## 🙏 Acknowledgement

We extend our gratitude to these wonderful projects and resources that made our work possible:

- 🌸 [Diffusers](https://huggingface.co/docs/diffusers/index) - The foundational library for diffusion models
- 🔥 [PyTorch](https://pytorch.org/) - Our deep learning framework of choice
- 🤗 [Hugging Face](https://huggingface.co/) - For model sharing infrastructure
- 🏛️ [Stable Diffusion](https://stability.ai/) - The base models we built upon
- 📚 [Our University/Institution] - For computational resources and support

*Special thanks to all contributors and open-source maintainers in the community!*