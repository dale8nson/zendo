<p align="center">
  <img src="./assets/zendo-logo-gradient-transparent-dark.svg" alt="ZendoAI Logo" width="700" />
</p>
<br><br>
A full-stack, modular generative AI platform for vision-based machine learning — created by <a href="daletristanhutchinson.com">Dale Hutchinson</a>.

# ZendoAI

ZendoAI is a modular, full-stack generative AI platform for training and evaluating image–text models. It combines a Python backend (FastAPI), Rust modules (for performance-critical tasks), and a modern Next.js frontend for image annotation, model evaluation, and future image generation tools.

---

## 🚀 Features

- Upload, crop, and transform images in the browser
- Predict labels using CLIP embeddings via `/api/predict`
- Automatically save prediction metadata
- Python–Rust hybrid backend
- Benchmark scripts for performance comparisons
- Future plans for training workflows and image generation (e.g., StyleGAN2)

---

## 🧱 Project Structure

```
.
├── assets                  # Logo and branding assets
├── backend
│   ├── models             # Downloaded CLIP model (OpenCLIP)
│   ├── python             # FastAPI app and services
│   │   ├── app
│   │   └── requirements.txt
│   └── rust               # Rust performance modules and FFI bindings
├── benchmarks             # Benchmark scripts
├── frontend               # Next.js App Router frontend
│   ├── app
│   ├── public
│   └── styles and config
├── Dockerfile
├── fly.toml               # Deployment config for Fly.io
├── pyrightconfig.json
├── rust-project.json
└── README.md
```

---

## 🛠 Requirements

### Python Backend

- Python 3.11+
- FastAPI, Uvicorn, Pillow, pydantic
- Install dependencies from the `backend/python/requirements.txt` file

```bash
cd backend/python
pip install -r requirements.txt
```

### Rust Components

- Rust (stable)
- `tch`, `pyo3`, `image` crates
- You can build the Rust module from `backend/rust/` with:

```bash
cd backend/rust
cargo build --release
```

---

## ⚙️ Running the Backend

```bash
cd backend/python
uvicorn app.main:app --reload
```

The docs are available at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🖼 Using the Predict API

Send a POST request to `/api/predict` with this format:

```json
{
  "filename": "example.jpg",
  "transform": {
    "x": 0.1,
    "y": 0.2,
    "scale": 1.5,
    "fit": "contain"
  }
}
```

The response will include the predicted label and similarity score.

---

## 🧪 Running Benchmarks

```bash
cd benchmarks
python benchmark_clip_inference.py
python benchmark_transform.py
```

---

## 🧪 Future Development

- Add training interface for supervised fine-tuning
- Evaluate multiple CLIP variants
- Export dataset to Hugging Face format
- Generate photorealistic images using StyleGAN2

---

## 📄 License

MIT
