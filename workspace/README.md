# PyTorch CNN Classification Template

A small, production-style PyTorch template for image classification experiments.

It includes:

- Config-driven training
- Reproducible random seeds
- Train/validation split
- Mini-batch training with `DataLoader`
- Clean model, data, metrics, trainer, train, and evaluate modules
- Checkpoint saving/loading
- Device abstraction for CPU/CUDA
- Optional automatic mixed precision on CUDA

> This template uses a synthetic random image dataset by default so it runs immediately. Replace `datasets.py` with your real dataset later.

## Project layout

```text
pytorch_cnn_template/
├── README.md
├── requirements.txt
├── config.py
├── datasets.py
├── evaluate.py
├── metrics.py
├── models.py
├── train.py
├── trainer.py
└── checkpoints/
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

## Evaluate saved checkpoint

```bash
python evaluate.py
```

## Important note

If your labels are random, validation accuracy should stay close to chance level:

```text
1 / num_classes
```

For `num_classes = 100`, chance accuracy is about `1%`.

If training accuracy reaches 100% but validation accuracy stays near chance, the model is memorizing the training set.
