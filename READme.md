1. The 99.57% OA is probably inflated by leakage (highest priority)

Random pixel split with overlapping patches. You split 70/30 at pixel level, then take 7×7 patches. Most test pixels have training pixels in their immediate neighbourhood, and their patches share many of the same pixels. Training accuracy was already 98.9% after epoch 1, which suggests the model is mostly interpolating between neighbours. This is the main reason published HSI numbers on Salinas are hard to compare.

The graph branch works as a location lookup table. graph_feat = g[seg_ids] gives every pixel in a superpixel the same 64-d vector. With only 170 nodes for 54k labelled pixels, and large homogeneous fields in Salinas, that branch can memorise “superpixel → majority class”. Because test pixels share superpixels with training pixels, this is a legitimate shortcut under your split. The model doesn’t need real spatial-spectral reasoning to score high.
Fixes

Use a spatially disjoint split: assign whole blocks (e.g. 32×32) or whole connected regions to train or test, and drop training patches that overlap test pixels.
Or use the standard few-shot protocol (e.g. 5–10% or 20–50 pixels per class for training), which is also less saturated than 70%.
Report both the random-split and disjoint-split numbers, so readers can see the gap.
Compute superpixel purity (fraction of labelled pixels in each superpixel sharing the majority label). It shows how much the graph branch can get from region identity alone.
2. Ablations and baselines are missing

You can’t currently show that the hybrid design earns its complexity. Add:

ViT only, graph only, and fusion (the graph-only run will likely score surprisingly high, which confirms point 1)
Graph branch with a shuffled or random adjacency
GCN only versus GCN + Graph Transformer
Simple baselines: SVM on PCA features, a 2D-CNN, a 3D-CNN, and ideally SpectralFormer or a similar published model
3. Training and validation protocol
Epochs mismatch: the markdown and the cell header say 15 epochs, but EPOCHS = 5.
No validation set: carve out ~10% of train, use it for early stopping and checkpoint selection, and touch the test set once.
No LR schedule: add cosine or OneCycle, plus weight decay (AdamW).
Single seed: run 5–10 seeds and report mean ± std for OA/AA/Kappa. A 0.1–0.5% difference between variants is meaningless without that.
It ran on CPU (Device: cpu). Switch the Colab runtime to GPU and consider mixed precision.
Dataloader speed: PatchDataset.__getitem__ slices numpy per sample. Precompute all patches into one tensor, or use num_workers.
Augmentation: none at present. Flips and 90° rotations of the patch, plus small spectral noise, help generalisation on small training sets.
4. Graph construction
Too coarse: N_SEGMENTS=200 gave 170 nodes, roughly 650 pixels each. Boundary superpixels will mix classes. Try 500–1500 nodes or multi-scale superpixels.
Edges ignore space: the KNN graph uses feature similarity only. Add spatially adjacent superpixels (region adjacency graph), or combine both edge types.
Node features are just the mean PCA vector: add per-superpixel std, centroid coordinates, and size.
Hard pixel-to-node assignment: consider combining a pixel’s own features with its superpixel embedding, or using soft assignment, so the graph branch can’t act as a pure ID lookup.
Dense adjacency: fine for 170 nodes, but use sparse ops or PyTorch Geometric if you increase the node count.
5. Model architecture
The Graph Transformer ignores the graph. nn.MultiheadAttention attends over all nodes with no mask or bias, so the structure the GCN built is discarded. Use the adjacency as an attention mask or additive bias:
python
# in GraphTransformer.forward(x, A_hat)
mask = (A_hat == 0)                       # True = blocked
a, _ = self.attn(x, x, x, attn_mask=mask)

Alternatively add Laplacian positional encodings.

The fusion is a plain concat plus MLP. Try gated or cross-attention fusion, where the pixel tokens query the graph embedding.
ViT details: there is no CLS token (mean pooling is fine, but test both). Try patch sizes 9, 11 and 13, and more depth. Keep a spectral-aware stem (e.g. a 3D conv or spectral tokens) rather than relying only on a 3×3 conv over 30 PCs.
PCA = 30 is an assumption. Sweep 10/20/30/50, or compare against a learned spectral projection or band selection.
6. Preprocessing hygiene
PCA, scaling, and the graph are fit on the whole image, including test pixels. This is unsupervised and common, but state it as transductive. For a strict version, fit the scaler and PCA on training pixels only.
Add a per-band standardisation check, since Salinas has water-absorption bands that were already removed in the “corrected” version.
7. Evaluation and reporting
The crop map in Cell 10 predicts all labelled pixels, including the 70% training pixels. It looks better than the test performance really is. Show the map with training pixels masked or marked, or predict only test pixels.
The comment in Cell 9 is stale: it mentions “Grapes_untrained → Lettuce_romaine_7wk”, but your actual top confusion is Grapes_untrained → Vinyard_untrained (48 samples). Fix the comment and discuss that pair, which is the genuinely hard one.
Add confidence intervals or a McNemar test when comparing variants.
Report parameter count, training time and inference time alongside accuracy.
8. Engineering
Cell 2 relies on google.colab.files.upload. Add a path-based loader with config for .mat keys, so it runs outside Colab.
Set torch.backends.cudnn.deterministic, seed the DataLoader generator, and save checkpoints and the config.
The DATASET flag only controls class names. For Indian Pines, the loader’s “first 3-D array” detection is fragile, so pass the key names explicitly.
