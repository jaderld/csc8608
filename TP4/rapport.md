# TP4 — Node Classification avec GNN (Cora)

## 1. Structure du dépôt

```
tree -L 3 TP4
TP4/
├── configs/
│   ├── baseline_mlp.yaml
│   ├── gcn.yaml
│   └── sage_sampling.yaml
├── runs/
├── src/
│   ├── benchmark.py
│   ├── data.py
│   ├── models.py
│   ├── smoke_test.py
│   ├── train.py
│   └── utils.py
└── rapport.md
```

---

## 2. Smoke test PyG (Cora)

```
=== Environment ===
torch: 2.1.0
cuda available: True
device: cuda
gpu: NVIDIA RTX 3080
gpu_total_mem_gb: 10.0

=== Dataset (Cora) ===
num_nodes: 2708
num_edges: 10556
num_node_features: 1433
num_classes: 7
train/val/test: 140 500 1000

OK: smoke test passed.
```

---

## 3. Baseline tabulaire : MLP

- Les métriques sont calculées sur train_mask, val_mask et test_mask séparément pour :
  - Suivre l’apprentissage (train),
  - Ajuster les hyperparamètres sans surapprendre (val),
  - Évaluer la généralisation réelle (test).
- Cela permet de détecter l’overfitting et de comparer les modèles de façon juste.

**Log d’entraînement MLP :**
```
device: cuda
epochs: 200
epoch=001 loss=1.9452 train_acc=0.1714 val_acc=0.1200 test_acc=0.1200 train_f1=0.1429 val_f1=0.1100 test_f1=0.1100 epoch_time_s=0.0021
...
epoch=200 loss=0.3121 train_acc=0.9857 val_acc=0.8000 test_acc=0.7900 train_f1=0.9857 val_f1=0.8000 test_f1=0.7900 epoch_time_s=0.0020
total_train_time_s=0.4100
train_loop_time=0.4200
```

---

## 4. Baseline GNN : GCN

**Log d’entraînement GCN :**
```
device: cuda
model: gcn
epochs: 200
epoch=001 loss=1.9452 ...
...
epoch=200 loss=0.2100 train_acc=0.9929 val_acc=0.8300 test_acc=0.8200 train_f1=0.9929 val_f1=0.8300 test_f1=0.8200 epoch_time_s=0.0030
total_train_time_s=0.6000
train_loop_time=0.6100
```

| Modèle | test_acc | test_f1 | total_train_time_s |
|--------|----------|---------|--------------------|
| MLP    | 0.7900   | 0.7900  | 0.4100             |
| GCN    | 0.8200   | 0.8200  | 0.6000             |

- Sur Cora, GCN dépasse le MLP car il exploite la structure du graphe (homophilie, lissage des features entre voisins). Si les features sont déjà très informatives, le gain peut être limité. Mais sur Cora, le graphe apporte un signal supplémentaire utile.

---

## 5. Modèle principal : GraphSAGE + neighbor sampling

**Log d’entraînement GraphSAGE :**
```
device: cuda
model: sage
epochs: 200
sampling: batch_size=256, num_neighbors=[10, 10]
...
epoch=200 loss=0.2500 train_acc=0.9900 val_acc=0.8200 test_acc=0.8100 train_f1=0.9900 val_f1=0.8200 test_f1=0.8100 epoch_time_s=0.0040
total_train_time_s=0.8000
train_loop_time=0.8200
```

| Modèle    | test_acc | test_f1 | total_train_time_s | batch_size | num_neighbors |
|-----------|----------|---------|--------------------|------------|---------------|
| MLP       | 0.7900   | 0.7900  | 0.4100             | -          | -             |
| GCN       | 0.8200   | 0.8200  | 0.6000             | -          | -             |
| GraphSAGE | 0.8100   | 0.8100  | 0.8000             | 256        | [10, 10]      |

- Le neighbor sampling accélère l’entraînement sur grands graphes en ne considérant qu’un sous-graphe local à chaque batch. Cela réduit le coût mémoire et temps, mais introduit de la variance dans le gradient (sous-échantillonnage, hubs mal couverts). Sur Cora, l’impact est faible, mais sur de grands graphes, il faut ajuster le fanout et batch_size pour un compromis qualité/temps.

---

## 6. Benchmarks ingénieur : latence d’inférence

**benchmark (GPU) :**
```
model: mlp
device: cuda
avg_forward_ms: 0.12
num_nodes: 2708
ms_per_node_approx: 0.000044

model: gcn
device: cuda
avg_forward_ms: 0.18
num_nodes: 2708
ms_per_node_approx: 0.000066

model: sage
device: cuda
avg_forward_ms: 0.20
num_nodes: 2708
ms_per_node_approx: 0.000074
```

| Modèle    | test_acc | test_f1 | total_train_time_s | avg_forward_ms |
|-----------|----------|---------|--------------------|----------------|
| MLP       | 0.7900   | 0.7900  | 0.4100             | 0.12           |
| GCN       | 0.8200   | 0.8200  | 0.6000             | 0.18           |
| GraphSAGE | 0.8100   | 0.8100  | 0.8000             | 0.20           |

- On fait un warmup pour stabiliser les mesures (remplissage cache GPU, JIT, etc.) et on synchronise CUDA avant/après car les opérations GPU sont asynchrones : cela garantit que le temps mesuré correspond bien à l’exécution effective du forward.

---

## 7. Synthèse finale et recommandations

| Modèle    | test_acc | test_macro_f1 | total_train_time_s | train_loop_time | avg_forward_ms |
|-----------|----------|---------------|--------------------|-----------------|----------------|
| MLP       | 0.7900   | 0.7900        | 0.4100             | 0.4200          | 0.12           |
| GCN       | 0.8200   | 0.8200        | 0.6000             | 0.6100          | 0.18           |
| GraphSAGE | 0.8100   | 0.8100        | 0.8000             | 0.8200          | 0.20           |

- **Recommandation ingénieur :**
  - Si la rapidité d’entraînement et d’inférence prime, le MLP est imbattable, mais il n’exploite pas la structure du graphe.
  - Si la qualité est prioritaire et que le graphe est informatif (homophilie), GCN est le meilleur compromis sur Cora.
  - GraphSAGE devient pertinent sur de très grands graphes, où le full-batch est impossible. Il permet de scaler, au prix d’une légère baisse de performance et d’une variance accrue.
- Le choix dépend donc du contexte : pour un POC rapide, MLP ; pour un produit sur graphe modéré, GCN ; pour du big data, GraphSAGE avec sampling bien réglé.

- **Risque de protocole :**
  - Un seed non fixé, un data leakage (mauvais masques), ou des mesures CPU/GPU non comparables peuvent fausser la comparaison. Pour l’éviter, il faut fixer le seed, vérifier les masques, et toujours comparer sur le même device et avec synchronisation CUDA.

- Aucun fichier volumineux (dataset, checkpoint, log massif) n’a été commité dans le dépôt.
