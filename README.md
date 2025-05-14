<p align="center">
  <img src="./assets/zendo-logo-gradient-transparent-dark.svg" alt="ZendoAI Logo" width="700" />
</p>
<br><br>
A full-stack, modular generative AI platform for vision-based machine learning — created by [Dale Hutchinson](https://daletristanhutchinson.com).

# ZendoAI

**ZendoAI** is an experimental generative AI platform designed to support the entire image model development lifecycle — from dataset creation and annotation to model training, fine-tuning, and photorealistic image generation from text prompts.

This project is in its early foundational phase, focused on implementing core functionality in **Rust** and wrapping it in **Python** for integration with machine learning workflows. The long-term vision is to create a modular, performant, and elegant system combining Rust, Python, and Next.js.

---

## 🔍 Why ZendoAI?

- Designed for developers and researchers who want full control over their image-based ML pipelines
- Inspired by Zen principles of **clarity**, **intentionality**, and **technical elegance**
- Aims to bridge the performance of systems programming with the flexibility of Python and the accessibility of modern web interfaces

---

## 🧪 Current Features

- ✅ Core image-processing functions implemented in **Rust**:
  - Image resizing
  - Aspect-ratio-preserving fit
  - Cropping
- ✅ Python bindings generated for integration with ML workflows
- 🛠️ Under development:
  - CLI tools for image dataset preprocessing
  - CLI test harness for evaluating Rust bindings

---

## 🧱 Tech Stack

| Layer        | Technology                    |
|--------------|-------------------------------|
| Core Logic   | Rust                          |
| Python API   | PyO3                          |
| ML/Workflow  | PyTorch (planned)             |
| Frontend     | Next.js + WASM (planned)      |
| Deployment   | Docker (planned)              |

---

## 🛠️ Getting Started

> ⚠️ This project is under heavy development. Interfaces and folder structure may change frequently.

### 1. Clone the repository

```bash
git clone https://github.com/dale8nson/zendoai.git
cd zendoai
```

### 2. Build Rust library

```bash
cd rust
cargo build --release
```

### 3. Set up Python virtual environment

```bash
cd python
python3 -m venv venv
source venv/bin/activate
pip install maturin
maturin develop
```

### 4. Run Python test script

```bash
python test_bindings.py
```

---

## 🧭 Roadmap

- [x] Core Rust image operations
- [x] Python bindings (PyO3 / maturin)
- [ ] Dataset preprocessing CLI
- [ ] Test coverage for all functions
- [ ] FastAPI backend for prediction (planned)
- [ ] Canvas-based web frontend (Next.js + WASM)
- [ ] Generative model integration (CLIP + GAN/diffusion)
- [ ] Model versioning + metadata storage

---

## 🌱 Why I Built This

ZendoAI reflects my passion for building systems that are not only technically powerful but also thoughtfully designed. It’s a long-term research project to explore the full lifecycle of generative image models — from structured dataset creation to neural network training to interactive synthesis and evaluation.

My goal is to create an open, modular platform that combines:
- The performance of **Rust**
- The flexibility of **Python**
- The accessibility of **modern web UIs**
- The intentionality of **good design**

---
