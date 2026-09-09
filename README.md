# GNN-BERT for Understanding Context from Music

CSE425 / EEE474 / CSE715 — Supervised Neural Network Project.
Hybrid **BERT + Graph Neural Network** system for music context understanding
(multi-label tagging, cross-modal alignment; emotion regression is out of scope,
see below).

| Task | Status | Model |
| --- | --- | --- |
| 1 (Easy) — BERT tag classifier | **done** | `bert-base-uncased`, masked + unmasked |
| 2 (Medium) — GNN on music structure graphs | **done** | GraphSAGE / GAT / GCN + CNN baseline |
| 3 (Hard) — GNN–BERT fusion | **done** | cross-attention fusion + 4-way ablation |
| 4 (Advanced) — Contrastive retrieval | **done** | dual encoder + InfoNCE |

### Where the deliverables live

| Deliverable | Path |
| --- | --- |
| Final report | [`report/final_report.pdf`](report/final_report.pdf) |
| Demo notebook (end-to-end inference) | [`notebooks/demo_context.ipynb`](notebooks/demo_context.ipynb) |
| EDA notebook | [`notebooks/eda.ipynb`](notebooks/eda.ipynb) |
| Evaluation tables + plots | `results/metrics/`, `results/metrics.json` (digest), `results/plots/`, `results/retrieval_examples/` |
| 20 preprocessed graph samples | `data/processed/graph_samples/` |
| Full run report & caveats | [`RESULTS.md`](RESULTS.md) |

**Scope note — emotion regression.** DEAM (valence/arousal) ships no text and
could not be joined to the fusion setup cleanly, so the multi-task weight
`λ = α = β = 0` and no emotion head is built. The loss machinery and MAE/R²
metric code are present but unused; see `RESULTS.md`.

---

## Running everything (Task 1 → 4)

```bash
python run_all.py
```

Runs all eleven stages in dependency order. It checks each stage's requirements
*before* starting it, skips stages whose output already exists, and stops at the
first genuine blocker instead of failing deep into a long run. Re-running the
same command resumes.

```bash
python run_all.py --dry-run
```

Prints the plan and what is blocking, without executing anything — do this
first.

Other flags: `--only task2`, `--from task3`, `--to task2`, `--skip-download`
(reuse clips already on disk), `--force` (redo completed stages).

**Prerequisites**, checked by the preflight report:

| Need | For | Install |
| --- | --- | --- |
| CUDA build of torch | everything | `pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124` |
| `librosa`, `soundfile` | Tasks 2–4 | `pip install librosa soundfile` |
| `torch-geometric` | Tasks 2–4 | `pip install torch-geometric` |
| `yt-dlp` + ffmpeg on PATH | Task 3 clips | `pip install yt-dlp` |
| GTZAN or FMA-small audio | Task 2 | manual download, see below |

Task 1 needs none of these beyond torch — it runs on a fresh checkout.

Rough GPU timings (estimates, not measured): Task 1 minutes; Task 2 ~30 min
mostly feature extraction; Task 3 **1–3 hours dominated by the clip download**;
Task 4 20–40 min. Feature caches and downloads are both resumable.

---

## Task 1 — BERT baseline for music tag understanding

Text-only multi-label classifier, no graph structure:

```
t     = BERT_CLS(X_text)
y_hat = sigmoid(W t + b)
L     = -(1/K) sum_k [ y_k log y_hat_k + (1 - y_k) log(1 - y_hat_k) ]
```

### Dataset choice: MusicCaps caption → tag proxy

The spec allows either MagnaTagATune top-50 tags or the MusicCaps caption → tag
proxy task. **MusicCaps** is used here for two reasons:

1. Task 1 is a *text-only* model. In MagnaTagATune the tags are the labels, and
   the dataset carries no accompanying text, so there would be nothing for BERT
   to encode. MusicCaps pairs each 10 s clip with an expert-written caption,
   which is genuine model input.
2. MusicCaps is the dataset Tasks 3 and 4 build on, so the label vocabulary and
   splits created here carry forward instead of being thrown away.

Only `musiccaps-public.csv` (2.9 MB) is required — Task 1 needs no audio.

| Property | Value |
| --- | --- |
| Clips | 5,521 |
| Unique aspects | 13,219 |
| Aspects per clip | 10.67 mean |
| Label vocabulary | top-50 aspects by frequency (`data.top_k_tags`) |
| Text input | expert caption, max 128 word-piece tokens |
| Test split | official `is_audioset_eval` flag |
| Val split | 10% of the remainder, seeded, split by `ytid` |

### Known caveat: label-in-caption leakage

MusicCaps aspects were written by the same annotator as the caption, so many
appear verbatim in the text (`'sad'`, `'medium tempo'`, `'acoustic drums'`).
Left alone, part of the proxy task degenerates into substring matching and the
F1 numbers look better than the model's actual understanding.

`data.mask_aspect_spans: true` (or `--mask-aspects`) deletes those exact spans
from the input before tokenisation. **Report both settings** — the gap between
them is the honest measure of how much the encoder generalises rather than
copies, and it makes for a much stronger discussion section than a single
inflated score.

### Running it

```bash
pip install -r requirements.txt
```

Build the corpus (downloads the CSV on first run, writes splits + label vocab):

```bash
python src/data_musiccaps.py --config config.yaml
```

Fine-tune (CUDA, mixed precision, discriminative LRs, linear warmup):

```bash
python src/train.py --config config.yaml
```

Evaluate — test metrics, baselines, F1 curves, 5 qualitative predictions:

```bash
python src/evaluate.py --config config.yaml
```

Useful variants:

```bash
python src/train.py --config config.yaml --model bert-base-uncased --epochs 10
```

```bash
python src/train.py --config config.yaml --mask-aspects
```

All commands are run **from the project root** — paths in `config.yaml` are
relative to it.

### What gets produced

| Path | Contents |
| --- | --- |
| `data/processed/musiccaps_task1.csv` | text, multi-hot labels, split assignment |
| `data/processed/label_vocab.json` | the K tag strings, frequency-ordered |
| `data/splits/task1_splits.json` | train / val / test `ytid` lists |
| `results/checkpoints/task1_bert_best.pt` | best epoch by val Macro-F1 |
| `results/metrics/task1_history.json` | per-epoch loss / Macro-F1 / Micro-F1 / AUC-PR |
| `results/metrics/task1_test_metrics.json` | test table incl. baselines |
| `results/metrics/task1_per_tag.json` | per-tag P / R / F1 / AP / support |
| `results/metrics/task1_examples.json` | 5 example predictions |
| `results/plots/task1_f1_curves.png` | **F1 vs. epoch curves** (deliverable) |
| `results/plots/task1_per_tag_f1.png` | per-tag F1, 25 most frequent tags |
| `results/plots/task1_attention_*.png` | `[CLS]` attention heat-maps (deliverable) |

### Baselines reported (spec Section 8)

* **B1a** random predictor — each tag sampled from its training prior
* **B1b** majority predictor — always the most frequent tags
* **TF-IDF + one-vs-rest logistic regression** — non-neural text reference,
  isolates how much the pretrained encoder actually contributes
* **B3** = this model (the BERT-only row of the Task 3 ablation)

### Methodology notes for the report

* **Thresholds.** Per-tag decision thresholds are swept on *validation* only and
  frozen for test. A global 0.5 cut-off badly underperforms on rare tags; tuning
  on test would leak labels and inflate the result.
* **Undefined metrics.** Tags with no positives in a split score F1 = 0
  (`zero_division=0`) and are excluded from the AUC-PR mean.
* **Loss.** `BCEWithLogitsLoss` on raw logits for numerical stability; optional
  per-tag `pos_weight = neg/pos` (clipped at 20) for imbalance.
* **Optimisation.** AdamW, encoder LR 2e-5 / head LR 1e-3, no weight decay on
  biases and LayerNorm, 10% linear warmup then linear decay, grad-norm clip 1.0.
* **Artist leakage.** Splits are by `ytid`, so no clip straddles two splits.
  MusicCaps releases no artist metadata, so the spec's "no artist leakage"
  requirement holds only to the extent the released data permits — state this in
  the report rather than claiming more.
* **Train-side F1 curves** are accumulated from the training forward passes, so
  they are a running within-epoch estimate, not a clean end-of-epoch pass.

---

## Task 2 — GNN on music structure graphs

Audio-only node features, no text. GraphSAGE update and mean-pool readout as
specified:

```
h_i^(l+1) = sigma( W^(l) . CONCAT( h_i^(l), MEAN_{j in N(i)} h_j^(l) ) )
g         = (1/|V|) sum_i h_i^(L)
y_hat     = sigma(W g + b)
```

### Getting the audio

Task 2 needs real audio — neither dataset is bundled here.

* **GTZAN** (1.2 GB, 1,000 × 30 s, 10 genres) — the spec's "standard easy
  baseline". Unpack so that `data/raw/gtzan/genres_original/<genre>/*.wav` exists.
* **FMA-small** (7.2 GB, 8,000 × 30 s, 8 genres) — heavier but has proper
  artist-disjoint official splits. Needs `fma_small/` plus `fma_metadata/tracks.csv`.

Switch between them with `task2.dataset` or `--dataset`; nothing else changes.

### Two graph constructions

| | Segment graph (default) | Chord-transition graph |
| --- | --- | --- |
| Nodes | 5 s windows, 50% overlap (or beat-synchronous) | unique chords from 24 triad templates |
| Node features | 83-d: MFCC μ/σ, chroma μ/σ, spectral contrast, centroid/bandwidth/rolloff/ZCR/RMS μ/σ, position, duration | 27-d: mean chroma, duration, count, minor flag, root one-hot |
| Edges | temporal adjacency + cosine similarity > τ | observed transitions |
| Edge weight | 1.0 (temporal) / similarity | normalised transition count |

Similarity edges are **capped at `max_sim_neighbors` per node**. A bare
threshold makes a homogeneous track's graph almost complete, which washes out
message passing and inflates memory — the cap keeps the structure meaningful.

### Running it

```bash
pip install librosa soundfile torch-geometric
```

Index the tracks and assign splits:

```bash
python src/audio_datasets.py --config config.yaml
```

Extract features and build graphs (slow the first time, cached afterwards):

```bash
python src/graph_builder.py --config config.yaml
```

Train the GNN and the CNN baseline under an identical protocol:

```bash
python src/train_gnn.py --config config.yaml --model both
```

Swap the encoder or the graph family:

```bash
python src/train_gnn.py --config config.yaml --model gnn --conv gat
```

```bash
python src/graph_builder.py --config config.yaml --graph-type chord
```

### What gets produced

| Path | Contents |
| --- | --- |
| `data/processed/task2_index_<dataset>.csv` | track_id, path, label, artist, split |
| `data/processed/audio_cache/*.npz` | cached mel / chroma / node features per track |
| `data/processed/graphs/task2_graphs_<dataset>_<type>.pt` | all graphs, split-grouped |
| `data/processed/graph_samples/` | **20 example `.pt` + `.json` graphs** (submission req. 2) |
| `results/metrics/task2_graph_stats.json` | node / edge / degree statistics |
| `results/metrics/task2_results.json` | test metrics, per-class report, confusion matrices |
| `results/plots/task2_curves.png` | accuracy and Macro-F1 vs. epoch, both models |
| `results/plots/task2_confusion_*.png` | row-normalised confusion matrices |

### Methodology notes for the report

* **Fair comparison.** The GNN and the CNN read the same cached audio, the same
  splits, the same epoch budget, optimiser family and metric code. The CNN is
  excluded from any track the graph pipeline skipped, so both are scored on an
  identical track set — otherwise the comparison silently measures data coverage.
* **Fixed crops vs. variable length.** The CNN needs a fixed 256-frame crop;
  the graph encoder consumes whole tracks natively. That is a genuine advantage
  of the graph representation and belongs in the discussion.
* **Node scaling.** Mean/std are fitted on the **train split only** and applied
  to val/test.
* **Splits.** FMA's official splits are artist-disjoint. GTZAN ships no artist
  metadata, so the code falls back to a stratified random split and prints a
  warning — report it as a limitation rather than claiming no leakage.
* **GTZAN caveats** (Sturm, 2013): duplicate clips, several mislabelled tracks,
  repeated artists, and a corrupt `jazz.00054.wav`. The pipeline skips
  unreadable files and reports how many were dropped.
* **Edge weights.** Only `GCNConv` consumes scalar edge weights; `SAGEConv`
  ignores them and `GATConv` learns its own attention. The weights are passed
  selectively rather than silently dropped — worth stating when comparing convs.
* **Chord estimation** is template matching on CQT chroma with a majority
  filter, not a trained chord recogniser. It is a structural prior, not a
  transcription claim.

---

## Task 3 — GNN–BERT fusion for multi-context understanding

Cross-attention fusion exactly as specified — the graph vector is a single query
attending over the caption's tokens, so the model learns *which words the audio
structure is talking about*:

```
A = softmax( Q K^T / sqrt(d) ),   Q = g W_Q,  K = H_text W_K
z = CONCAT( g, A H_text ),        y_hat = sigma( W z )
L = L_tags + alpha ||v - v_hat||^2 + beta ||a - a_hat||^2
```

With `fusion_heads: 1` and no value projection, `CrossAttentionFusion` is
literally that equation. Multi-head is available; reported attention is then
averaged over heads.

### Dataset: MusicCaps, paired

Task 3 needs the full tuple `(X_audio, X_text, G, y)`. MusicCaps is the only
listed source that supplies all four for the same clip — caption as text, aspect
tags as labels, and 10 s audio for the graph. **FMA-medium is also supported**
(`task3.dataset: fma_medium`, text synthesised from artist/album/title/tags,
labels from `genres_all`) since the spec names it for the results table.

> **Dataset of record for the reported runs: MagnaTagATune** (`task3.dataset:
> magnatagatune`, the current `config.yaml` default). MusicCaps ships YouTube ids
> rather than audio, and only ~150 of 5,521 clips are still downloadable here
> (bulk `yt-dlp` fetching is blocked by YouTube bot-verification), which is far
> too few to train the fusion model. MagnaTagATune gives 21,108 real paired
> examples; the spec permits it for Task 3 explicitly. Its `Artist/Album/Title`
> metadata is a much weaker text signal than a caption — see
> `RESULTS.md` → *MagnaTagATune vs. MusicCaps* for the full rationale and its
> measured effect on Tasks 3 and 4.

The MusicCaps label vocabulary is **reused from Task 1**, so the `bert_only`
ablation row is the Task 1 model and the numbers are directly comparable across
tasks.

### Ablation (required deliverable)

All four modes share one classifier head, one training loop, one seed and one
protocol, so the table isolates the fusion mechanism rather than head capacity:

| Mode | z |
| --- | --- |
| `bert_only` | `t` — the Task 1 model |
| `gnn_only` | `g` — the Task 2 encoder |
| `concat` | `CONCAT(g, t)` — early fusion |
| `cross_attention` | `CONCAT(g, A H_text)` — recommended |
| `gated` | `CONCAT(g, gate ⊙ t)` — extra row, not required |

### Running it

Fetch the clips (slow, and expect attrition — see below):

```bash
python src/download_musiccaps_audio.py --config config.yaml --workers 4
```

Build the paired dataset (index + graphs + node scaler):

```bash
python src/data_fusion.py --config config.yaml
```

Train the full ablation:

```bash
python src/train_fusion.py --config config.yaml --ablation all
```

t-SNE, case studies and graph coherence:

```bash
python src/analyze_fusion.py --config config.yaml
```

### What gets produced

| Path | Contents |
| --- | --- |
| `data/processed/fusion/task3_fusion_<dataset>.pt` | paired index + graphs + label vocab |
| `results/metrics/musiccaps_download_log.json` | how many clips were actually obtainable |
| `results/metrics/task3_results.json` | full ablation table, per-mode histories |
| `results/metrics/task3_analysis.json` | coherence, distributions, case studies |
| `results/plots/task3_ablation.png` | val curves per mode + test bar chart |
| `results/plots/task3_tsne_genre.png` | **t-SNE of z by genre** (deliverable) |
| `results/plots/task3_tsne_mood.png` | **t-SNE of z by mood** (deliverable) |
| `results/plots/task3_case_*.png` | **3 case studies** (deliverable) |

### Methodology notes for the report

* **YouTube attrition is the headline caveat.** MusicCaps ships ids, not audio,
  and a meaningful share are now private, deleted or region-locked. The
  downloader logs every failure and `data_fusion.py` prints how many rows it
  dropped — quote those counts in the report instead of the nominal 5,521.
* **Clip length drives segmentation.** MusicCaps clips are 10 s, so `task3.audio`
  uses 2 s segments. Reusing Task 2's 5 s windows would give two or three nodes
  per graph and there would be nothing for message passing to do.
* **Padding is masked before the softmax.** Otherwise short captions leak
  padding tokens into the fused vector.
* **Warm start.** The text branch loads the fine-tuned Task 1 encoder by default.
  It is free, and it makes the `bert_only` row a like-for-like comparison rather
  than a differently-initialised model.
* **Partial BERT freezing** (`freeze_text_layers: 2`) follows Algorithm 3's
  "(partial) BERT": the lower layers encode generic syntax that a few thousand
  captions cannot improve, and freezing them keeps training stable at batch 16.
* **Masked multi-task loss.** Tag and emotion terms are each averaged over only
  the samples that carry those targets, so a batch can mix MusicCaps (tags, no
  emotion) with DEAM (emotion, no text) and neither term's scale depends on how
  the batch happened to be composed.
* **DEAM asymmetry.** DEAM has no text, so for those rows the fusion degenerates
  to the graph branch. That is a real limitation of the joint setup, not an
  oversight — state it if you enable `use_deam`. With MusicCaps alone,
  `alpha = beta = 0` and no emotion head is built.
* **Node importance** for the case studies is the projection `h_i · ĝ`. Under
  mean pooling every node contributes equally in magnitude, so raw norms would
  say almost nothing; the projection measures how much a node pushes `g` in the
  direction the classifier reads.
* **t-SNE colour groups are derived, not supervised.** Genre and mood buckets
  come from keyword matching over the aspect vocabulary; the model never sees
  them, and unmatched clips are labelled `other` rather than guessed at.

---

## Task 4 — Cross-modal MusicCaps alignment

Dual encoder trained with InfoNCE over in-batch negatives:

```
g_i   = Normalize( GNN(G_i) )
t_i   = Normalize( BERT_CLS(caption_i) )
S_ij  = g_i^T t_j / tau
L_NCE = -(1/N) sum_i log( exp(S_ii) / sum_j exp(S_ij) )
```

Both branches project into one shared 256-d space, so retrieval is a
nearest-neighbour lookup in either direction. **No new data is needed** — Task 4
reuses the Task 3 paired file, dropping any row without a caption.

### Running it

```bash
python src/train_contrastive.py --config config.yaml
```

Batch size is a modelling choice here, not just a memory knob — every other item
in the batch is a negative, so 64 gives 63 negatives per step:

```bash
python src/train_contrastive.py --config config.yaml --batch-size 128
```

### What gets produced

| Path | Contents |
| --- | --- |
| `results/metrics/task4_results.json` | retrieval table, zero-shot metrics, history |
| `results/plots/task4_retrieval_curves.png` | InfoNCE loss + R@K vs. epoch |
| `results/plots/task4_similarity_matrix.png` | test similarity matrix — a bright diagonal means alignment |
| `results/retrieval_examples/task4_retrieval_examples.json` | **10 query captions → top-3 clips** (deliverable) |
| `results/retrieval_examples/task4_retrieval_examples.html` | the same, readable, with YouTube links |
| `results/retrieval_examples/human_eval_sheet.csv` | **blank rating sheet, 5 raters** (spec Section 6) |

### Methodology notes for the report

* **Symmetric loss by default.** The spec writes the graph→caption direction
  only, but both directions are *reported*. Training one direction and grading
  the other measures something the loss never optimised, so the default averages
  both; `symmetric_loss: false` reproduces the one-directional equation.
* **Pessimistic tie-breaking.** Every candidate scoring ≥ the correct one counts
  as ranked ahead. Counting only strictly-greater candidates hands a perfect
  R@1 to a collapsed model that emits identical scores — verified: a constant
  similarity matrix now scores R@1 = 0 with median rank N.
* **R@K depends on the candidate pool.** Recall over a 300-clip test split is not
  comparable to recall over 3,000. Always report `n_candidates` (it is in the
  JSON) beside the recall numbers.
* **fp32 for the similarity matrix.** The N×N logits are computed outside
  autocast — fp16 with τ = 0.07 overflows.
* **Zero-shot decision rule.** There is no validation set to calibrate a
  threshold on — that is what makes it zero-shot — so tagging takes the top-n
  tags, n = mean true tags per clip. AUC-PR is the fairer headline since it is
  threshold-free.
* **Zero-shot has a text-side control.** The same tag prompts are also scored
  against *caption* embeddings. If graph→tag barely beats caption→tag, the
  alignment is carrying little audio information — report both rows.
* **The human-eval sheet hides ground truth** on purpose. Showing raters which
  clip is the true pair would bias every rating collected.
* **`drop_last=True` on the train loader.** A final batch of size 1 has no
  negatives and its InfoNCE term is degenerate.

### File map

```
src/
  data_musiccaps.py   # Task 1: download, label vocab, splits, torch Dataset
  bert_encoder.py     # Task 1: BERT encoder + multi-label head (re-used by Task 3)
  train.py            # Task 1: Algorithm 1 training loop
  evaluate.py         # Task 1: test metrics, baselines, plots, attention examples

  audio_features.py   # Task 2: mel / chroma / MFCC, segmentation, feature cache
  audio_datasets.py   # Task 2: GTZAN + FMA indexing, artist-aware splits
  graph_builder.py    # Task 2: segment + chord-transition graphs -> PyG Data
  gnn_model.py        # Task 2: GraphSAGE / GAT / GCN encoder, pooled readout
  cnn_baseline.py     # Task 2: baseline B2, CNN on mel-spectrogram
  train_gnn.py        # Task 2: Algorithm 2 loop + GNN vs. CNN comparison

  download_musiccaps_audio.py  # Task 3: yt-dlp clip fetcher
  data_fusion.py      # Task 3: paired (G, text, y) index, masks, collate
  fusion_model.py     # Task 3: cross-attention fusion + all ablation modes
  train_fusion.py     # Task 3: Algorithm 3 loop + ablation runner
  analyze_fusion.py   # Task 3: t-SNE, case studies, graph coherence

  contrastive.py      # Task 4: dual encoder + InfoNCE
  retrieval_eval.py   # Task 4: R@K, examples, human-eval sheet, zero-shot tagging
  train_contrastive.py # Task 4: Algorithm 4 loop + all evaluation deliverables

  metrics.py          # shared: Macro/Micro-F1, AUC-PR, thresholds, baselines,
                      #         emotion regression, multiclass scores
```
