# Moodsic

Emotion-aware music recommendation proof-of-concept that pairs VEATIC video analytics with the DEAM music catalogue. The repository bundles a Flask backend, a React dashboard, dataset preparation scripts, training utilities, and VEATIC evaluation pipelines to support rapid research iteration.

## Installation

### Python environment
1. (Optional) Create a virtual environment if `.venv` is missing:
   ```bash
   python3 -m venv .venv
   ```
2. Activate it before running any Python command:
   ```bash
   source .venv/bin/activate.fish
   ```
3. Install the core dependencies with `uv`:
   ```bash
   uv pip install -r requirements.txt
   ```
4. For inference/evaluation tooling install the extra set (includes pyarrow, fastai, etc.):
   ```bash
   uv pip install -r requirements_inference.txt
   ```

### Frontend dependencies
Install the React dependencies once:
```bash
cd frontend
npm install
```

### Dataset setup
- FindingEmo imagery lives under `data/Run_1` and `data/Run_2`. To re-download:
  ```bash
  source .venv/bin/activate.fish
  python scripts/findingemo_parallel_download.py --target-dir data --workers 100 --timeout 15
  ```
- The Flask API expects DEAM audio at `data/DEAM/MEMD_audio/` and VEATIC assets at `data/VEATIC/` (`videos/`, `rating_averaged/`, and `shortlisted_videos/`).
- Scripts such as `scripts/train_scene_model.py` and `scripts/evaluation/run_inference_pipeline.py` read and write from `data/` and `results/`; keep those directories writable.

## Running the Application

`run_app.sh` activates the virtualenv (when present), starts the Flask backend (`backend/run.py`), and launches the React development server:

```bash
./run_app.sh
```

The script assumes you have already run `npm install` in `frontend/`. To start the services manually:

```bash
source .venv/bin/activate.fish
python backend/run.py  # http://localhost:5000
```

```bash
cd frontend
npm start  # http://localhost:3000
```

## Project Structure
```
emo-rec/
├── README_APP.md                        # Focused app usage guide
├── backend/                             # Flask API + helpers
│   ├── app.py
│   ├── constants.py
│   ├── helpers/
│   └── run.py
├── configs/                             # YAML configs for training
│   ├── base_config.yaml
│   ├── face_models/
│   └── scene_models/
├── data/                                # Local datasets (FindingEmo, DEAM, VEATIC, splits)
├── frontend/                            # React dashboard (src/App.js, styles)
├── notebooks/                           # Research notebooks + exported artifacts
├── results/                             # Saved inference/evaluation outputs
├── scene/                               # Pretrained checkpoints used by the pipelines
├── scripts/                             # CLI tools (dataset prep, training, evaluation, clustering)
├── src/                                 # Python package with models/data/utils modules
├── tests/                               # Pytest suite for fusion + recommendation logic
├── run_app.sh
├── requirements.txt
├── requirements_inference.txt
└── AGENTS.md
```

## Data Assets

- `data/Run_1` and `data/Run_2`: FindingEmo labelled images used by the scene model trainer.
- `data/VEATIC`: VEATIC videos (`videos/`), evaluation label averages (`rating_averaged/`), and `shortlisted_videos/` clips served by the demo.
- `data/DEAM`: DEAM audio under `MEMD_audio/` consumed by `GET /api/song/<song_id>`.
- `results/inference` and `results/evaluation`: Pipeline outputs referenced by the backend (e.g., `pipeline_results_20251006_144126.parquet`) and evaluation notebooks.

## Scene Model Training

The training CLI lives at `scripts/train_scene_model.py` and relies on YAML configs under `configs/`. Always activate the virtualenv first:

```bash
source .venv/bin/activate.fish
python scripts/train_scene_model.py \
  --config configs/scene_models/scene_model_clip_vit_b32_frozen_auto_lr_config.yaml \
  --data.findingemo_path /path/to/FindingEmo
```

Key notes:
- Update `data.findingemo_path` either via CLI override (shown above) or by editing the config to point at your local dataset.
- The provided CLIP ViT-B/32 config inherits from `configs/base_config.yaml` and enables automatic LR discovery, WMSE/CCE losses, and Emo8 auxiliary supervision.
- Other backbones (DINOv3, pretrained ResNet) are supported by the codebase; create a new YAML file that inherits from `base_config.yaml` and set `model.backbone_type` (`clip`, `dinov3`, or `imagenet`), `model.clip_model_name`, or `model.imagenet_backbone_name` as needed.
- Use additional CLI overrides such as `--training.batch_size 16` or `--training.learning_rate 0.0001` to experiment without editing the YAML.

Typical outputs land in `experiments/checkpoints/` and `logs/` directories defined in the config.

## Evaluation Pipelines

### Batch VEATIC inference
The exporter mirrors the research notebook workflow and writes JSON payloads plus a consolidated Parquet file.

```bash
source .venv/bin/activate.fish
python scripts/evaluation/run_inference_pipeline.py \
  --video-dir data/VEATIC/videos \
  --output-root results/inference \
  --stabilizer-mode both
```

Optional flags include `--scene-weight`, `--face-weight`, `--no-variance-weighting`, `--video-limit`, and overlay capture toggles.

### Aggregate metrics
Summarise a pipeline run into per-video and aggregate CSVs:

```bash
source .venv/bin/activate.fish
python scripts/evaluation/aggregate_veatic_metrics.py \
  results/inference/pipeline_results_20251006_144126.parquet
```

Outputs default to `results/evaluation/` (CSV + JSON metadata).

### Linking fused valence/arousal to DEAM clusters
Annotate fused predictions with DEAM GMM assignments:

```bash
source .venv/bin/activate.fish
python scripts/clustering/deam_clusters.py \
  --bundle-dir results/clustering/deam_gmm_20251006_151857 \
  --parquet results/inference/pipeline_results_20251006_144126.parquet \
  --output results/inference/pipeline_results_20251006_144126_clusters.parquet
```

Programmatic access is available by importing `annotate_parquet_with_clusters` from `scripts/clustering/deam_clusters.py`.

## Testing

Pytests cover fusion behaviour, overlay generation, and song matching. Run them from the project root (after installing requirements):

```bash
source .venv/bin/activate.fish
pytest -q
```

If you need a headless OpenCV build for CI, install `opencv-python-headless` via `uv pip install "opencv-python-headless==4.12.0.88" --no-deps` and remove the GUI build from `requirements.txt`.

# Frontend UI

<placeholder>

# Evaluation

## Results: Scene Model Ablation

1. **Metrics and outcomes**

   | Model (Notebook) | Train Valence MAE ↓ | Test Valence MAE ↓ | Train Arousal MAE ↓ | Test Arousal MAE ↓ | Train Avg MAE ↓ | Test Avg MAE ↓ | Test Spearman ρ (Val / Aro / Avg) | Reference |
   | --- | --- | --- | --- | --- | --- | --- | --- | --- |
   | CLIP ViT-B/32 + multi-head aux (`CLIP_ViT-B32_improved_fixed.ipynb`) | 0.3834 | 0.3746 | 0.4519 | 0.4351 | 0.4176 | 0.4048 | 0.6641 / 0.2225 / 0.4433 | `docs/Stage III — Training/scene_model_ablation.md:22`-`docs/Stage III — Training/scene_model_ablation.md:25` |
   | CLIP ViT-B/32 + linear head (`CLIP_ViT-B32.ipynb`) | 0.3657 | 0.3699 | 0.4394 | 0.4329 | 0.4025 | 0.4014 | 0.6500 / 0.2970 / 0.4735 | `docs/Stage III — Training/scene_model_ablation.md:22`-`docs/Stage III — Training/scene_model_ablation.md:25` |

2. **Rationale for metric selection**

   Mean Absolute Error (MAE) remains the primary objective because the pipeline must deliver calibrated valence/arousal predictions within the bounded reference space [-1, 1]; MAE is directly interpretable in that scale and resilient to outliers relative to MSE. Spearman’s ρ is reported to assess monotonic ordering, which is pertinent when the downstream fusion layer ranks pathway estimates before weighting them.

3. **Analysis**

   CLIP_ViT-B32_improved_fixed keeps MAE within ≈0.003–0.005 of the linear-head baseline while nudging valence ordering to ρ = 0.6641 (+0.014) [docs/Stage III — Training/scene_model_ablation.md:22](docs/Stage%20III%20—%20Training/scene_model_ablation.md#L22)-[docs/Stage III — Training/scene_model_ablation.md:25](docs/Stage%20III%20—%20Training/scene_model_ablation.md#L25). Despite DINOv3's newer self-distillation pipeline, its features stem from label-free invariance objectives and curated image crops, leaving our head to infer affect semantics from scratch; in contrast, CLIP pretrains on 400 M image–text pairs where captions bake in descriptors like "warm sunset" or "tense alley," so its frozen embeddings already align with valence/arousal cues when fine-tuned on the ≈13 k FindingEmo samples ([arXiv:2103.00020](https://arxiv.org/abs/2103.00020)). DINOv3's scaled self-supervision emphasises cross-view consistency and smoothness ([arXiv:2304.07193](https://arxiv.org/abs/2304.07193)), which excels at object recognition but can mute color-tone and contextual signals that drive emotional regression, explaining the weaker Spearman lift even with the MLP head. The improved CLIP run further layers dropout and an auxiliary emo8 branch [docs/Stage III — Training/scene_model_ablation.md:11](docs/Stage%20III%20—%20Training/scene_model_ablation.md#L11), pairing with the production-ready checkpoint export [docs/Stage III — Training/scene_model_ablation.md:26](docs/Stage%20III%20—%20Training/scene_model_ablation.md#L26) to deliver more stable optimisation. The simpler linear head remains a strong fallback with marginally lower MAE and higher arousal ranking, echoing the deployment notes captured in [docs/project_overview.md:181](docs/project_overview.md#L181)-[docs/project_overview.md:203](docs/project_overview.md#L203).

## Overall Performance on Held-Out Dataset (VEATIC)

1. **Metrics and outcomes**

   | Pathway (Stabilized) | Valence MAE ↓ | 95% CI | Arousal MAE ↓ | 95% CI | Coverage Mean | Coverage Std | Reference |
   | --- | --- | --- | --- | --- | --- | --- | --- |
   | Face | 0.208 | 0.185–0.234 | 0.168 | 0.148–0.189 | 0.861 | 0.171 | `docs/Stage VI - Evaluation/eval_res_round_i.md:34`-`docs/Stage VI - Evaluation/eval_res_round_i.md:44` |
   | Scene | 0.238 | 0.210–0.268 | 0.203 | 0.178–0.230 | – | – | `docs/Stage VI - Evaluation/eval_res_round_i.md:34`-`docs/Stage VI - Evaluation/eval_res_round_i.md:46` |
   | Fusion (scene weight 0.6, face weight 0.4) | 0.193 | 0.169–0.217 | 0.161 | 0.140–0.184 | – | – | `docs/Stage VI - Evaluation/eval_res_round_i.md:34`-`docs/Stage VI - Evaluation/eval_res_round_i.md:46` |

2. **Rationale for metric selection**

   MAE is reported to quantify absolute deviations in the common reference space shared by scene and face pathways, enabling direct comparison across modalities. Confidence intervals communicate statistical stability across the dataset. Coverage is provided for the face pathway because it remains conditioned on successful detections and therefore needs an availability metric alongside accuracy.

3. **Analysis**

   Fusion achieves the lowest MAE on both axes (≈7% lower than face-only and ≈19% lower than scene-only), confirming that variance-weighted blending is superior to single-expert predictions under the current weighting scheme. The near-zero deltas between stabilized and unstabilized runs (≤0.001 MAE change) show that the temporal smoothing window improves user experience without compromising accuracy. Sustained face coverage at 0.861 ± 0.171 indicates that most clips enjoy dual-expert support, explaining why fusion rarely degrades to the less accurate scene-only regime. The archived artefacts (`results/evaluation/veatic_*_20251006_144126.*`) therefore provide a reliable baseline for future weight-sweep experiments without necessitating a fresh inference run.

## Results: Clustering of Music Dataset (DEAM)

1. **Metrics and outcomes**

   | Algorithm | Clusters (excl. noise) | Noise Fraction | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | Avg. Posterior Entropy ↓ | Weighted Avg. Purity ↑ | Source |
   | --- | --- | --- | --- | --- | --- | --- | --- | --- |
   | Gaussian Mixture Model (diag, k=5) | 5 | 0.00% | 0.325 | 0.899 | 1 538.6 | 0.686 | 0.229 | `notebooks/Dataset - DEAM/deam_clustering (des).ipynb` |
   | HDBSCAN (min_cluster_size=35) | 6 | 59.99% | 0.294 | 0.702 | 615.8 | 1.222 | 0.250 | `notebooks/Dataset - DEAM/deam_clustering (des).ipynb` |

2. **Rationale for metric selection**

   The silhouette coefficient captures cohesion–separation balance, while the Davies–Bouldin index penalises overlapping clusters; together they assess structural quality independent of cluster count. Calinski–Harabasz responds to cluster compactness relative to overall variance, useful for gauging the explanatory power of the latent stations. Posterior entropy and weighted average purity quantify assignment confidence and genre concentration respectively, highlighting whether clusters remain interpretable for recommendation logic. Noise fraction is reported for density-based methods to make the coverage trade-off explicit when large segments of the catalogue remain unclustered.

3. **Analysis**

   The GMM delivers stronger global separation (higher silhouette, lower Davies–Bouldin, and a Calinski–Harabasz score more than twice that of HDBSCAN) while retaining full catalogue coverage. Its lower entropy indicates that posterior assignments are decisive, which simplifies downstream gating. HDBSCAN achieves slightly higher weighted purity within its accepted clusters, implying more genre-homogeneous stations, but this improvement is offset by the 60% noise rate that would require fallback logic for a majority of songs. For the production-facing recommender, the GMM therefore offers a superior balance between coverage, numerical stability, and ease of integration, with HDBSCAN reserved for exploratory analyses where high-confidence islands are preferable to full coverage.
