<div align="center">
<h1>Graph-CAFR: Context-Aware Semantic Graph Reasoning <br>for Compositional Zero-Shot Fruit Recognition</h1>

**A Research Project by Trần Vũ Quang Huy**
<br>
*Bridging the Gap between Visual Features and Semantic Attributes in Agricultural AI*

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-Graph_Neural_Networks-green.svg)](https://pytorch-geometric.readthedocs.io/)
[![CLIP](https://img.shields.io/badge/Backbone-Vision_Transformers-lightgrey.svg)](https://github.com/mlfoundations/open_clip)

</div>

---

## 📌 Project Overview

**Graph-CAFR** is an advanced framework designed for **Compositional Zero-Shot Fruit Recognition (CZFR)**. It enables mobile-based identification of diverse produce and their specific states (e.g., freshness, variety, color) without requiring exhaustive training data for every possible attribute-fruit combination.

### 🔍 The Problem: The Compositional Explosion
In agricultural retail, the variety of cultivars and ripening stages is vast. Traditional supervised models fail because:
1. **Massive Annotation Cost**: It is impractical to collect labeled data for every single "Attribute + Object" pair.
2. **Unseen Compositions**: Models must recognize combinations they have never seen during training (e.g., inferring a specific "ripe" state for a "newly introduced cultivar" based on prior knowledge).

### 💡 The Graph-CAFR Solution
Our framework introduces **Context-Aware Semantic Graph Reasoning** combined with **Explicit-Implicit Relation Learning**:
- **Unified Semantic Graph**: Establishes high-level correlations between fruit types and physical attributes.
- **Context-Aware Reasoning**: Captures the subtle interplay of how the same attribute (e.g., "red") manifests differently across species (e.g., "Red Apple" vs. "Red Cherry").
- **Explicit-Implicit Learning**: Models both direct visual-attribute links and latent semantic dependencies to accurately identify novel compositions.

---

## 🛠️ Tech Stack & Skills
- **Core Architecture**: Vision Transformers (CLIP Backbone) + Semantic Graph Reasoning Layers.
- **DL Frameworks**: PyTorch, PyTorch Geometric.
- **Domain Expertise**: Compositional Zero-Shot Learning (CZSL), Agricultural Computer Vision.
- **Research Infrastructure**: Optimized for mobile-ready inference and large-scale semantic embedding.

---

## 📊 Performance Benchmark (tlu_states dataset)

Evaluating the model's ability to recognize Unseen Compositions compared to traditional baselines:

| Methodology | Task Type | Accuracy (Top-1) |
| :--- | :--- | :---: |
| Baseline (CLIP Zero-Shot) | CZSL | 31.9% |
| VP-CMJL (Vanilla) | CZSL | 38.2% |
| **Graph-CAFR (Ours)** | **CZSL** | **43.5%** |

*\*Note: Graph-CAFR demonstrates superior reasoning capabilities in agricultural domains where attribute-object relationships are highly context-dependent, outperforming base architectures by a significant margin.*

---

## 🚀 Execution Guide

### 1. Training & Inference
To train the Graph-CAFR model on the TLU-States dataset using context-aware reasoning:
```bash
python train_cafr.py --dataset tlu_states \
                     --backbone ViT-B/32 \
                     --use_context_reasoning True \
                     --relation_learning explicit_implicit
```

### 2. Zero-Shot Evaluation
Test the model's performance on unseen fruit-attribute compositions (`--test_mode zero_shot`):
```bash
python evaluate.py --checkpoint models/graph_cafr_final.pth \
                   --test_mode zero_shot \
                   --show_top_k 5
```

---

## 📁 Repository Organization
- `models/graph_cafr.py`: Core implementation of the Context-Aware Semantic Graph.
- `layers/relation_learning.py`: Modules for Explicit-Implicit relation extraction.
- `data/tlu_states_loader.py`: Preprocessing and split logic for the TLU-States agricultural dataset.
- `scripts/benchmark.py`: Comparative analysis tools for CZSL baselines.

---
*Developed as a modern solution for Agricultural Computer Vision. For collaboration or inquiries, please reach out via GitHub or LinkedIn.*
