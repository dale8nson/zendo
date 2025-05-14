<p align="center">
  <img src="./assets/zendo-logo-gradient-transparent-dark.svg" alt="ZendoAI Logo" width="700" />
</p>
<br><br>
A full-stack, modular generative AI platform for vision-based machine learning — created by [Dale Hutchinson](https://daletristanhutchinson.com).

# ZendoAI

**ZendoAI** is an experimental generative AI platform designed to support the entire image model development lifecycle — from dataset creation and annotation to model training, fine-tuning, and photorealistic image generation from text prompts.

This project represents an ongoing effort to combine **Rust**, **Python**, and **Next.js** into a unified, performant, modular system for generative AI research and tooling.

---

## 🔍 Why ZendoAI?

- Designed for **researchers, developers, and creators** working at the intersection of ML, UI/UX, and image synthesis
- Inspired by Zen principles of **clarity**, **precision**, and **intentionality** in design and engineering
- Prioritizes **modularity** and **customization** over one-size-fits-all black-box models

---

## 🚀 Key Features (WIP)

- 📁 **Image annotation and gallery interface** (Next.js + Tailwind)
- 🧠 **Canvas-based image editor** for cropping, labeling, preprocessing
- 🛠️ **FastAPI backend** for prediction and preprocessing
- 🧬 **CLIP integration** for vision–language embedding
- 🧪 **Rust–Python bridge** for performance-critical tasks (planned)
- 🌀 Future integration of **custom generative models** (GAN, diffusion, etc.)
- 🔄 Model versioning, loading/saving, and dataset switching (in progress)

---

## 🧱 Tech Stack

| Layer          | Technology                     |
|----------------|--------------------------------|
| Frontend       | Next.js (App Router), Tailwind CSS |
| State/Storage  | Local uploads (S3 planned)     |
| Backend        | FastAPI (Python)               |
| ML/Inference   | PyTorch, CLIP                  |
| Performance    | Rust + WASM (planned)          |
| Deployment     | Docker (WIP), Fly.io (planned) |

---

## 🛠️ Installation & Setup

> Note: This project is under active development. Setup steps may change.

### 1. Clone the repository

```bash
git clone https://github.com/dale8nson/zendoai.git
cd zendoai
```

### 2. Install frontend dependencies

```bash
cd frontend
npm install
npm run dev
```

### 3. Start backend (Python/FastAPI)

```bash
cd backend
source venv/bin/activate  # or use your preferred virtualenv tool
uvicorn app.main:app --reload
```

### 4. Access the app

Visit [http://localhost:3000](http://localhost:3000)

---

## 🧭 Roadmap

- [x] Image upload + persistent gallery
- [x] Canvas-based annotation and editor
- [x] Prediction API integration
- [ ] Model save/load support
- [ ] Rust–Python data bridge
- [ ] Dataset segmentation + label versioning
- [ ] Generative model fine-tuning + inference
- [ ] Cloud-based persistent storage (Backblaze B2 / S3)

---

## 🌱 Why I Built This

ZendoAI is a personal research project rooted in a broader vision: to create a robust, ethical, and modular system for exploring **AI-generated imagery** — from raw data to final render.

It merges my interests in:
- Systems programming (**Rust**)
- Machine learning experimentation (**Python**)
- Human-centered UI design (**React/Next.js**)
- Performance optimization and custom ML tooling

ZendoAI reflects my belief that **developer tools should be beautiful**, **flexible**, and **deeply thoughtful** — and that the future of generative AI lies in giving creators more understanding and control.
