# AdaSR: Efficient and Adaptive Super-Resolution Decoding ⚡

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

**Authors:** Jacob Maimon, Gal Zohar, Tomer Abram, Tomer Baziza, Yonatan Bansay  
**Based on:** ECDP Framework

---

## 📌 Executive Summary

**AdaSR** is an innovative extension of the Efficient Conditional Diffusion Model (ECDP) framework tailored for image super-resolution.

The core inefficiency of modern diffusion models is the **uniform application of computational resources** across all image regions, regardless of their complexity. Standard models process smooth backgrounds with the same intensity as complex textures.

**AdaSR** introduces **inference-level adaptivity**, achieving a superior speed–quality tradeoff. It reduces energy consumption while maintaining—and in several cases, **enhancing**—image reconstruction quality.

> **Conceptual Analogy:** Think of AdaSR as a **smart cleaning crew**. A standard crew spends exactly 10 minutes scrubbing every square inch of a floor, even the clean parts. The AdaSR crew quickly identifies the stains (complex regions), spends extra time scrubbing them, and moves on quickly from clean spots. Because they don't waste time on clean spots, they finish the whole job faster and can use that saved energy to make sure the tough stains are truly spotless.

---

## 🔬 Core Mechanisms & Mathematics

AdaSR implements two primary adaptive mechanisms that operate during the inference phase:

### A. Adaptive Decoding (Spatial Adaptivity)
Instead of processing every pixel equally, the model identifies "Easy" regions (like smooth backgrounds) and "Hard" regions (like eyes or fine textures).

1.  **Patch Complexity ($C_p$):** The image is divided into patches. The complexity of a patch is the average complexity of its constituent pixels:
    $$C_p = \frac{1}{|p|} \sum_{i,j \in p} C(x_{i,j})$$

2.  **Adaptive Step Allocation ($N_p$):** Each patch is assigned a specific step budget based on its complexity:
    $$N_p = N_{min} + (N_{max} - N_{min}) C_p$$
    * This ensures detailed areas receive rigorous refinement while simple areas are processed efficiently.

### B. Adaptive Early Stopping (Temporal Adaptivity)
This mechanism monitors the global refinement process and terminates the algorithm once the image reaches stability.

1.  **Convergence Rate ($\Delta_t$):** The model calculates the difference between the predicted clean image at the current step ($t$) and the subsequent step ($t+1$):
    $$\Delta_t = \| \hat{x}_0^{(t)} - \hat{x}_0^{(t+1)} \|$$

2.  **Termination Condition:** The process stops if a mandatory "warm-up" phase ($T_{min}$) is completed and the maximum convergence rate over a "Patience" period ($P$) falls below a threshold ($\epsilon$):
    $$\max(\Delta_t, \Delta_{t+1}, \dots, \Delta_{t+P}) < \epsilon$$

---

## 📊 Performance Results

Our experiments demonstrate that adaptivity allows for **higher-fidelity outputs** compared to fixed-step baselines, in addition to saving time.

### Quantitative Improvements (Case Studies)
| Image Subject | Baseline PSNR | **AdaSR PSNR** | Baseline SSIM | **AdaSR SSIM** | Improvement |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Moon** | 28.06 dB | **28.37 dB** | 0.804 | **0.815** | **+0.31 dB** |
| **Airplane** | 33.71 dB | **34.85 dB** | 0.817 | **0.894** | **+1.14 dB** |

### Efficiency Gains
* **Speedup:** Achieved a **1.18x speedup** on average.
* **Step Reduction:** Saved an average of **34.2 steps** per image compared to the fixed baseline.
* **Visual Quality:** Qualitative tests indicate human observers often cannot distinguish between a 30-step adaptive output and a 200-step standard output.

---

## 🚀 How to Run

### 1. Installation
```bash
git clone [https://github.com/JacobMaimon13/AdaSR-Efficient-Diffusion.git](https://github.com/JacobMaimon13/AdaSR-Efficient-Diffusion.git)
cd AdaSR-Efficient-Diffusion
pip install -r requirements.txt
2. Download Data
Automatically download the DIV2K dataset:

Bash

python -m src.data.download
3. Training
To train the model from scratch (or fine-tune):

Bash

python train.py --epochs 30 --batch_size 8
4. Evaluation & Comparison
Run the comparison script to generate the metrics table and visual gallery:

Bash

python test.py --model_path checkpoints/best_model.pt --num_images 5
📂 Project Structure
Plaintext

src/
├── adaptive/           # The core innovation (Adaptive Logic)
│   ├── complexity.py   # Gradient/Variance complexity estimators
│   ├── allocation.py   # Logic for assigning steps per patch
│   └── early_exit.py   # Temporal early stopping mechanism
├── models/             # Base Diffusion & UNet architecture
├── data/               # Data loading and transforms
├── training/           # Training loops and validation
└── utils/              # LPIPS/PSNR metrics and visualization
🔮 Future Directions
To combat the rising energy demands of Generative AI, we propose expanding AdaSR principles to:

Spatiotemporal Data: Adaptive sampling for video super-resolution.

Neural Audio Synthesis: Using early stopping for transient sound generation.

NLP: Applying adaptive inference to Large Language Models (LLMs) to reduce text generation costs.

📜 Acknowledgments
This project builds upon the ECDP (Efficient Conditional Diffusion Model) framework. We credit the original authors for their foundational work in diffusion-based super-resolution.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details
