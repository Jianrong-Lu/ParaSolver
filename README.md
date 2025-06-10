
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
```python
from parasolver import ParaSolver

# Initialize with your favorite diffusion model
solver = ParaSolver(
    model=your_diffusion_model,
    n_subintervals=40,  # Number of temporal partitions
    precondition_steps=5,  # Warm-up steps
    tol=0.05  # Convergence tolerance
)

# Generate samples in parallel
samples = solver.sample(batch_size=8, steps=100)
```

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

### Custom Configuration
```python
# Advanced configuration example
solver = ParaSolver(
    model=your_model,
    n_subintervals=30,
    precondition_steps=10,
    tol=0.01,
    window_size=8,  # For memory optimization
    max_iters=50,   # Maximum parallel iterations
    device='cuda',
    verbose=True
)
```

### Integration with Popular Frameworks
```python
# With Diffusers
from diffusers import StableDiffusionPipeline
from parasolver import DiffusersWrapper

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
accelerated_pipe = DiffusersWrapper(pipe, parasolver_params={
    'n_subintervals': 40,
    'tol': 0.05
})

image = accelerated_pipe("A beautiful landscape").images[0]
```



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



Here's the enhanced `Acknowledgement` section with icons and improved formatting:


## 🙏 Acknowledgement

We extend our gratitude to these wonderful projects and resources that made our work possible:

- 🌸 [Diffusers](https://huggingface.co/docs/diffusers/index) - The foundational library for diffusion models
- 🔥 [PyTorch](https://pytorch.org/) - Our deep learning framework of choice
- 🤗 [Hugging Face](https://huggingface.co/) - For model sharing infrastructure
- 🏛️ [Stable Diffusion](https://stability.ai/) - The base models we built upon
- 📚 [Our University/Institution] - For computational resources and support

*Special thanks to all contributors and open-source maintainers in the community!*