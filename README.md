# Contrastive Knowledge Distillation

This project implements a **Knowledge Distillation** framework using **Contrastive Learning** (CLIP loss) to transfer knowledge from a large monolingual teacher model (OpenCLIP EVA02) to a smaller multilingual student model (LaBSE).

## Project Structure

- **`Contrastive_Knowledge_Distillation.ipynb`**: The main notebook containing the implementation of data preprocessing, model definitions, training loop, and evaluation.

## Methodology

### 1. Teacher-Student Setup
- **Teacher Model:** `EVA02-E-14-plus` (pretrained OpenCLIP). Freeze this model to use it as a feature extractor.
- **Student Model:** `setu4993/smaller-LaBSE`. A smaller, efficient multilingual model finetuned to match the teacher's embedding space.

### 2. Contrastive Loss (CLIP)
We align the student's Persian text embeddings with the teacher's English text embeddings using a symmetric cross-entropy loss over the similarity matrix.

$$
\text{Loss} = \frac{1}{2N} \sum_{i=1}^{N} \left(\log \frac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ij}/\tau)} + \log \frac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ji}/\tau)}\right)
$$

Where $s_{ij}$ is the cosine similarity between the $i$-th candidate and $j$-th reference, scaled by temperature $\tau$.

## Dataset
 The project uses a paired English-Persian dataset.
- Training Data: `train.csv` (Downloaded in notebook).
- Validation Data: `val.csv` (Downloaded in notebook).

## Usage
1. Open the notebook `Contrastive_Knowledge_Distillation.ipynb`.
2. Install dependencies (uncomment the installation cell if running for the first time).
3. Run all cells to train the student model.
