# Stanford Dogs Breed Classifier

A clean, end-to-end computer vision side project classifying **120 dog breeds** from the [Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset).

The project covers the complete lifecycle: extracting bounding boxes, training a fine-tuned ResNet-50 model in PyTorch, exporting it to ONNX for lightweight CPU inference, serving it with FastAPI, and packaging it into a multi-stage Docker container.

---

## 🛠️ Features

- **Model**: Fine-tuned ResNet-50 backbone on 120 canine breeds.
- **Inference**: Exported to **ONNX Runtime** for low-overhead CPU predictions.
- **API**: **FastAPI** web service with defensive image validation and a `/predict` endpoint.
- **Docker**: Multi-stage, non-root container (`appuser`) with standard-library health checks.
- **Tooling**: Dependency management and environment isolation powered by [`uv`](https://docs.astral.sh/uv/).

---

## 📁 Project Structure

```text
Stanford-Dogs/
├── app.py                  # FastAPI inference service
├── Dockerfile              # Multi-stage Docker build
├── pyproject.toml          # Project configuration & dependencies
├── pytorch_to_onnx.py      # PyTorch to ONNX export script
└── src/stanford_dogs/
    ├── __init__.py         # Package exports
    ├── config.py           # Paths & hyperparameter settings
    ├── crop_bbx.py         # XML bounding box extraction & cropping
    ├── dataset.py          # Dataset loading, train/val/test split & per-split transforms
    ├── inference.py        # PyTorch & ONNX inference classes 
    ├── model.py            # ResNet-50 architecture
    ├── train.py            # Training & evaluation loop
    ├── transforms.py       # Data augmentations & normalization
    └── utils.py            # Reproducibility seed & structured logging
```

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and install dependencies using `uv`:

```bash
git clone https://github.com/EmanAbdelhaleem/Stanford-Dogs.git
cd Stanford-Dogs

# Install dependencies into virtual environment
uv sync
```

### 2. Run the API Locally
Start the FastAPI server:

```bash
uv run uvicorn app:app --reload
```

Interactive Swagger documentation is available at **`http://localhost:8000/docs`**.

### 3. Make a Prediction
Test the `/predict` endpoint using `curl`:

```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@examples/n02091032_1396.jpg"
```

Response:
```json
{
  "predicted_class": "n02091032-Italian_greyhound"
}
```

---

## 🐳 Run with Docker

Build and run the container:

```bash
# Build image (~1.9 GB with CPU PyTorch)
docker build -t stanford-dogs .

# Run container
docker run -d -p 8000:8000 --name dog-app stanford-dogs
```

Check the health status:
```bash
curl http://localhost:8000/health
```

---

## 🏋️ Training & ONNX Export (Optional)

If you have the dataset in `data/processed/images/Images` and want to retrain or re-export:

```bash
# Train the model
uv run python -m stanford_dogs.train --n_epochs 10 --batch_size 32

# Export best checkpoint to ONNX
uv run --extra onnx python pytorch_to_onnx.py
```

---

## 👤 Author
**Eman Abdelhaleem** — [emanabdelhaleem4@gmail.com](mailto:emanabdelhaleem4@gmail.com)
