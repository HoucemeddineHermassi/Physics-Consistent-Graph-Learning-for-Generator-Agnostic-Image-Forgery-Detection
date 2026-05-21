# PCGL: Physics-Consistent Graph Learning for Blind Image Forgery Detection

[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of the paper: **"Physics-Consistent Graph Learning for Blind Image Forgery Detection"**.

## Overview
As generative image synthesis (e.g., Diffusion Models, GANs) reaches photorealistic fidelity, traditional forensics methods relying on textural artifacts are increasingly susceptible to failures. **PCGL** is a principled framework that detects forgeries by verifying the underlying **structural consistency** of image formation. 

Instead of searching for ephemeral generator-specific fingerprints, PCGL models authenticity as the adherence to a globally consistent physical world model where illumination, reflectance, and geometry are physically linked.

### Key Contributions
- **Physics-Guided Structural Consistency Paradigm**: Interpreting forgery as a local structural consistency break in the physical image formation process.
- **Hybrid Physics-First Architecture**: A framework integrating a Multi-modal Hybrid Physics Proxy (incorporating reflectance, illumination, depth, and normals) with a highly efficient Causal Graph Attention Network (CGAT).
- **Uncertainty-Aware Inference**: Leverages Monte Carlo Dropout to provide calibrated, uncertainty-aware forgery predictions, severely reducing overconfidence rates common in purely semantic models.
- **PhysForge Benchmark \& Empirical Rigor**: A controlled dataset of 1,200 images isolating specific physical violations. The framework is extensively validated across five standard datasets (CASIA v2.0, CocoGlide, In-the-Wild, Columbia, PhysForge), demonstrating superior cross-domain generalization.

---

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA toolkit (for GPU acceleration)

### Installation
```bash
git clone https://github.com/YourUsername/PCGL-Forensics.git
cd PCGL-Forensics
pip install -r requirements.txt
```

---

## Repository Structure

- `models/`: Implementation of the Hybrid Physics Proxy and CGAT architecture.
- `data_loader.py`: Integration and pre-processing pipeline for CASIA v2.0, CocoGlide, Columbia, In-the-Wild, and PhysForge datasets.
- `run_simulation.py`: Main script to execute the end-to-end evaluation pipeline.
- `elsarticle (1)/elsarticle/`: LaTeX source code and compiled PDF of the full manuscript.
- `figs/`: Output directory containing generated diagrams, qualitative results, and plot visualizations.

---

## Reproducibility

We provide dedicated scripts to autonomously reproduce every figure and table presented in the manuscript. All scripts save their outputs to the `figs/` directory.

| Component / Figure | Script to Run | Description |
| :--- | :--- | :--- |
| **Full Simulation** | `run_simulation.py` | Runs end-to-end evaluation, computing F1, AUC, and calibration metrics. |
| **Fig. 1 (DAG)** | `generate_scm_diagram.py` | Physical Interdependence Graph visualization. |
| **Qualitative Heatmaps**| `generate_attention_maps.py` | Forgery localization maps across different datasets. |
| **Attention Comparison**| `generate_attention_comparison.py`| Local GNN vs. Global Transformer attention visualization. |
| **Robustness Curves** | `generate_robustness_plots.py` | F1-score degradation curves (JPEG, Blur, Resizing). |
| **Adversarial & Attacks**| `generate_attack_examples.py` | Visualizations of physics-targeted attacks and digital degradations. |
| **Reliability Diagrams**| `generate_calibration_plot.py` | MC Dropout uncertainty calibration and expected calibration error (ECE). |
| **Explainability** | `generate_explainability_viz.py` | Mechanistic interpretability analysis of structural breaks. |
| **Scalability Analysis**| `generate_scalability_plot.py` | Resource efficiency analysis (Latency vs. Graph complexity). |
| **Failure Modes** | `generate_failure_cases.py` | Representative failure cases (Low-light, extreme compression, dense textures). |

To reproduce the primary quantitative results reported in the manuscript:
```bash
python run_simulation.py
```

---

## Dataset: PhysForge
The **PhysForge** benchmark is designed to systematically evaluate physical reasoning. It provides a targeted testbed of 1,200 images isolating three key violation axes:
1. **Lighting Mismatch**
2. **Shadow Inconsistency**
3. **Reflectance Anomaly**

### Generation Approach & Code Name
**Code Name:** `PhysForge-Gen`

**Approach:** The dataset is generated using a 3D scene intervention pipeline (via Blender Python API). The process begins by rendering a physically consistent "authentic" base scene. To generate a forgery, controlled causal interventions are applied—such as displacing a light source exclusively for one object, modifying a local material's BRDF (reflectance), or artificially shifting shadow maps. This yields paired images where the forgery contains a mathematically precise physical violation without introducing traditional 2D image splicing or resampling artifacts.

*(Note: The full PhysForge dataset will be made publicly available via HuggingFace Datasets upon manuscript acceptance.)*

---


```

