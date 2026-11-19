# Pokéwars: Joint Visual Recognition, Global Cavity Inference, and Antisymmetric Score Completion

Code and compact validation summaries for the CS 3780 / 5780 competition [The Pokéwar](https://www.kaggle.com/competitions/cs-3780-5780-fa-26-the-pokewar/). Given two noisy $32 \times 32$ grayscale sprites $(x_1, x_2)$, the task is to predict the outcome $y \in [0, 10]$ of a 1,024-round battle simulation between their latent Pokémon species.

Competition archives, model checkpoints, and per-image prediction arrays stay outside Git (`.gitignore`); verified cross-validation and end-to-end evaluation summaries are tracked under `results/`.

---

## 1. Problem Formulation and Risk Decomposition

Let $\mathcal{C} = \{1, \dots, 1025\}$ denote the anonymous species set and $\mathcal{X} = \{0, \dots, 255\}^{32 \times 32}$ the space of noisy grayscale sprites. Across the competition files, images partition into three strictly disjoint regimes:

| Split | Battles | Unique images | Species labels | Battle outcome $y$ | Empirical combinatorial structure |
|---|---:|---:|---|---|---|
| $\mathcal{D}_{\mathrm{lab}}$ (`train.csv`, labeled) | $184{,}800$ | $160{,}925$ | observed $(c_1, c_2)$ | observed | Exactly $157$ images/species; pairwise distinct unordered pairs |
| $\mathcal{D}_{\mathrm{unl}}$ (`train.csv`, unlabeled) | $79{,}200$ | $68{,}675$ | latent | observed | Exactly $67$ images/species; mean degree $2.31$; distinct unordered pairs |
| $\mathcal{D}_{\mathrm{test}}$ (`test.csv`) | $80{,}000$ | $160{,}000$ | latent | target | Degree $1$ per image; disjoint from $\mathcal{D}_{\mathrm{train}}$; distinct unordered pairs |

For an ordered species pair $(a, b) \in \mathcal{C}^2$ ($a \neq b$) with true single-round win probability $p_{ab} \in [0, 1]$, the battle outcome is the scaled empirical win frequency

$$
y = 10 \cdot \frac{W}{1024}, \qquad W \sim \mathrm{Binomial}(1024, p_{ab}), \qquad p_{ba} = 1 - p_{ab}.
$$

Predictions $\hat{y}$ are evaluated by normalized RMSE:

$$
\mathrm{Score}(\hat{y}, y) = 100 \left(1 - \frac{\mathrm{RMSE}(\hat{y}, y)}{D}\right), \qquad D = 4.382900214041612.
$$

### Bayes-Optimal Prediction and Conditional MSE Audit

Given an estimated win-score matrix $T \in [0, 10]^{1025 \times 1025}$ ($T_{ab} \approx 10 p_{ab}$) and a joint posterior $Q_e(c_1, c_2)$ over the candidate Cartesian product $\mathcal{K}(x_{e,1}) \times \mathcal{K}(x_{e,2})$ for battle $e$, the squared-error risk minimizer is the posterior expectation

$$
\hat{y}_e = \sum_{c_1 \in \mathcal{K}(x_{e,1})} \sum_{c_2 \in \mathcal{K}(x_{e,2})} Q_e(c_1, c_2)\, T_{c_1, c_2}.
$$

Decomposing fold-0 test MSE across three mutually exclusive events at the `73.50 LB` checkpoint (`results/score_fold0_constrained_v2.json`, $\mathrm{MSE} = 1.590$, $\mathrm{Score} = 71.23$) identifies the remaining bottlenecks:

| Error regime (Fold-0 held-out battles) | Battle share | Conditional RMSE | MSE contribution | Share of total MSE |
|---|---:|---:|---:|---:|
| Both top-1 correct — table floor ($T_{a,b}$ oracle) | $93.15\%$ | $0.296$ | $0.0814$ | $5.1\%$ |
| Both top-1 correct — posterior uncertainty hedge | $93.15\%$ | — | $0.2730$ | $17.2\%$ |
| True pair inside top-20 pool, but top-1 wrong | $5.90\%$ | $4.201$ | $1.0417$ | $65.5\%$ |
| At least one true species missing from top-20 pool | $0.95\%$ | $4.529$ | $0.1941$ | $12.2\%$ |

Reliability checks confirm that post-cavity posteriors are well-calibrated ($\mathbb{E}[w_{\max}] = 0.9955$ vs. empirical accuracy $0.9966$ for $w_{\max} \ge 0.95$; $0.601$ vs. $0.603$ for $w_{\max} \in [0.5, 0.7)$), so post-hoc sharpening degrades MSE. All remaining score gains must come from (i) expanding candidate pool recall, (ii) raising visual/ranking accuracy on noisy sprites, and (iii) tightening global combinatorial constraints.

---

## 2. Multi-Backbone Visual Recognition & Deep-Feature kNN

### Fold Protocol and Input Standardization

All $160{,}925$ labeled sprites are split by unique image path into $3$ class-stratified folds (`stratified_folds`, seed $3780$). Every backbone, pseudo-label bank, kNN gallery, and ranker for fold $f$ is fit strictly outside fold $f$. Each image is standardized per instance via

$$
\tilde{x} = \frac{x / 255 - \mu(x)}{\max(\sigma(x),\, 0.08)}.
$$

Four architecture families are trained (`train_large_image.py`, `train_large_image_external.py`) with AdamW, label smoothing $\alpha = 0.03$, on-the-fly affine jitter, and two-stage cosine schedules:

1. **Native $32 \times 32$ `WideResNet-28-10`** ($37.1\text{M}$ params): trained across five fold-safe prefixes (`wrn28_10`, `wrn28_10_mild`, `wrn28_10_mild_pseudo`, `wrn28_10_mild_pseudo_synth`, `wrn28_10_mild_pseudo_ctrl`), producing a 15-checkpoint weighted ensemble.
2. **Native $32 \times 32$ `WideResNet-40-10`**: deeper residual backbone ($93.35\%$ OOF top-1 with $\pm 5^\circ$ TTA).
3. **CIFAR-stem `DenseNet-169`**: $3 \times 3$ stride-$1$ stem without initial pooling ($93.75\%$ OOF top-1 with $\pm 5^\circ$ TTA).
4. **`ConvNeXt V2 Nano`**: bicubic $64 \times 64$ input preserving spatial resolution at stride $32$.

All four backbones are evaluated under deterministic rotation TTA $\Theta = \{-5^\circ, 0^\circ, +5^\circ\}$ (`export_probs_rot5.py`):

$$
p_{\mathrm{tta}}(c \mid x) = \frac{1}{|\Theta|} \sum_{\theta \in \Theta} \mathrm{softmax}\!\left(z(R_\theta \tilde{x})\right)_c.
$$

### Expanded $\pm 5^\circ$ kNN Gallery and Noise-Aware Listwise Ranking (`Phase 1`)

To eliminate missing-pool penalties, `extract_gallery_features_v2.py` forms a top-$K$ ($K = 40$) candidate pool $\mathcal{K}(x)$ from the four-backbone probability blend (`0.60 WRN-28 + 0.12 WRN-40 + 0.16 DenseNet-169 + 0.12 ConvNeXt V2`), raising OOF pool recall from $99.42\%$ ($K=20$) to **$99.74\%$** ($K=40$). Penultimate embeddings are averaged across $\Theta$ before $\ell_2$-normalization,

$$
u(x) = \frac{\sum_{\theta \in \Theta} \phi(R_\theta \tilde{x}) / \|\phi(R_\theta \tilde{x})\|_2}{\left\|\sum_{\theta \in \Theta} \phi(R_\theta \tilde{x}) / \|\phi(R_\theta \tilde{x})\|_2\right\|_2},
$$

and scored against a variable-size gallery $\mathcal{G}_{\setminus f}$ augmented with fold-safe high-confidence pseudo-labeled images to produce `knn1`, `knn5`, `knn15`, class prototype, and SNR-weighted clean prototype similarities alongside per-image physical degradation statistics (Immerkaer Laplacian noise $\hat{\sigma}_{\mathrm{lap}}$, foreground–background contrast $b - f$, Gaussian $\hat{\sigma}$, and impulse rate).

In `train_listwise_ranker.py`, two complementary rankers score all $K = 40$ candidates using these $45$ features:

1. **Permutation-Equivariant DeepSets Ranker (`ListwiseSetRanker`)**: maps candidate features + species embedding to $h_{ik}$, concatenates global candidate-set context $[h_{ik} \,\|\, \max_j h_{ij} \,\|\, \mathrm{mean}_j h_{ij}]$, and minimizes exact $K$-way multiclass cross-entropy over $\mathcal{K}(x_i)$.
2. **Noise-Aware LightGBM Ranker**: 300-tree gradient-boosted candidate scorer with categorical species index.

An OOF multiclass NLL calibrator blends both rankers with the 4-backbone log-probability, reaching **$95.59\%$** 3-fold OOF top-1 (`results/phase1_ranker_cv.json`).

---

## 3. Antisymmetric Binomial Matrix Completion (`PairMLP`)

In `train_pair_mlp.py`, each species $a \in \mathcal{C}$ has an embedding $e_a \in \mathbb{R}^{24}$ passed through a 3-layer SiLU/BatchNorm/Dropout MLP $g_\theta : \mathbb{R}^{72} \to \mathbb{R}$:

$$
h_\theta(a, b) = g_\theta\!\left([e_a \,\|\, e_b \,\|\, e_a \odot e_b]\right), \qquad
\ell_\theta(a, b) = h_\theta(a, b) - h_\theta(b, a), \qquad
T_\theta(a, b) = 10\,\sigma\!\left(\ell_\theta(a, b)\right).
$$

Exact skew-symmetry $\ell_\theta(b, a) = -\ell_\theta(a, b)$ guarantees $T_\theta(a, b) + T_\theta(b, a) = 10$ identically. Training minimizes weighted binomial cross-entropy with random orientation flipping $(a, b, y) \mapsto (b, a, 10 - y)$:

$$
\mathcal{L}_{\mathrm{BCE}}(\theta) = -\sum_{(a, b, y, w)} w \left[\frac{y}{10} \log \sigma(\ell_\theta(a, b)) + \left(1 - \frac{y}{10}\right) \log\big(1 - \sigma(\ell_\theta(a, b))\big)\right].
$$

---

## 4. Global Unordered-Pair Exclusivity & Bidirectional Cavity Inference

### Combinatorial Exclusivity Law

Across all $\binom{1025}{2} = 524{,}800$ unordered species pairs $\{a, b\}$ ($a \neq b$):

- Every unordered pair appears **at most once** across the entire competition (`184,800` known pairs have $0$ unordered collisions; `79,200` unlabeled battles collide on $<0.004\%$ of pairs).
- Thus $\mathcal{D}_{\mathrm{lab}} \cup \mathcal{D}_{\mathrm{unl}}$ already exhausts $264{,}000 / 524{,}800 = \mathbf{50.3\%}$ of all unordered pairs, and $\mathcal{D}_{\mathrm{test}}$ internally exhausts another **$15.2\%$**—excluding **$65.5\%$** of candidate pairings for any ambiguous test battle.

### Constrained Loopy BP on $\mathcal{D}_{\mathrm{unl}}$ + Bidirectional Bethe Cavity (`constrained_inference.py`, `joint_constrained_inference.py`)

For each battle $e = (u, v)$ with candidate sets $\mathcal{K}_u, \mathcal{K}_v$, let $P_e(\{a, b\}) = \sum_{(i, j): \{c_{ui}, c_{vj}\} = \{a, b\}} Q_e(i, j)$ denote the unordered-pair probability assigned by battle $e$ (merging reversed orientations $(a, b)$ and $(b, a)$ within the same battle to prevent self-inhibition). Under the Bethe approximation to the global all-different factor $\prod_{\{a,b\}} \mathbb{I}\!\left[\sum_e \mathbb{I}[e = \{a,b\}] \le 1\right]$, the log-availability message received by battle $e$ for pair $\{a, b\}$ from all *other* battles is

$$
\Lambda_{-e}(\{a, b\})
= \log A_{\mathrm{known}}(\{a, b\})
+ \sum_{e' \neq e} \log\!\Big(1 - P_{e'}(\{a, b\})\Big),
$$

where $A_{\mathrm{known}}(\{a, b\}) = 0$ if $a = b$ or $\{a, b\} \in \mathcal{D}_{\mathrm{lab}}$, and $1$ otherwise.

1. **Bayes Class-Prior Debiasing**: before applying the uniform pair-availability factor, classifier class-frequency skew is divided out via $\tilde{q}(c \mid x) \propto q(c \mid x) / \bar{q}(c)$.
2. **Constrained BP on $\mathcal{D}_{\mathrm{unl}}$**: combines the Gaussian–uniform battle factor $\psi_{uv}(c_1, c_2; y_{uv}) = (1-\varepsilon)\mathcal{N}(y_{uv}; T_{c_1, c_2}, \sigma^2) + \varepsilon/10$ ($\sigma = 0.5, \varepsilon = 0.05$), the known-pair mask $A_{\mathrm{known}}$, Sinkhorn dual multipliers $\beta_c \leftarrow \beta_c + \tfrac{1}{2}\log(67 / \sum_u b_u(c))$ enforcing the exact $67$ images/species cardinality of $\mathcal{D}_{\mathrm{unl}}$, and internal unordered-pair cavity messages $\Lambda_{-e}$.
3. **Bidirectional Coupling ($\mathcal{D}_{\mathrm{unl}} \leftrightarrow \mathcal{D}_{\mathrm{test}}$)**: two outer rounds alternate constrained BP on the $79{,}200$ unlabeled battles (conditioned on $\Lambda_{\mathrm{test}}$) with damped Bethe cavity updates on the $80{,}000$ test battles (conditioned on $A_{\mathrm{known}}$ and $\Lambda_{\mathrm{unl}}$):

$$
Q_e^{(t+1)}(i, j) \propto \tilde{q}_u(c_{ui})\, \tilde{q}_v(c_{vj})\, \exp\!\big(\Lambda_{-e}^{(t)}(\{c_{ui}, c_{vj}\})\big).
$$

On `test.csv`, this raises unlabeled BP exact-67 species recovery to **$67.9\%$** (mean edge confidence **$99.54\%$**, $79{,}193 / 79{,}200$ unique unordered pairs) and yields **$79{,}917 / 80{,}000$** unique top-1 test pairs with $0$ collisions against $\mathcal{D}_{\mathrm{lab}}$ (`results/submission_phase1_fastattack.json`). On the leak-free fold-0 benchmark, global cavity inference lifts end-to-end score by **$+4.56$ to $+5.32$ points** ($66.67 \to 71.23 \to 71.99$), driving the Kaggle Public LB from **$68.21 \to 73.50$** (and further improvements ready in `submission_phase1_fastattack.csv`).

---

## 5. Exact Physical Impulse Discovery (`25 / 245`) & Transductive Phase 2

Full 0–255 intensity-histogram inspection across all three splits revealed that sprite corruption consists of contrast attenuation, additive Gaussian noise (right-clipped at $255$), and **exact discrete impulse spikes at intensities $25$ ($20.8\times$ local density) and $245$ ($3.9\times$ local density)** with identical rates across splits:

| Split | $[24, \mathbf{25}, 26]$ per $10^4$ px | $[244, \mathbf{245}, 246]$ per $10^4$ px |
|---|---|---|
| `labeled` | $[3.6,\ \mathbf{77.8},\ 4.3]$ | $[27.3,\ \mathbf{99.4},\ 24.4]$ |
| `unlabeled` | $[3.3,\ \mathbf{78.1},\ 3.9]$ | $[27.4,\ \mathbf{100.1},\ 24.5]$ |
| `test` | $[3.4,\ \mathbf{78.2},\ 3.9]$ | $[27.5,\ \mathbf{101.0},\ 24.8]$ |

Replacing pixels satisfying $x_{ij} \in \{25, 245\}$ with their $3 \times 3$ non-impulse neighborhood median (`clean_impulses_gpu` in `degradation.py`, strictly per-image with zero dataset leakage) produces an immediate **zero-training** gain on an existing fold-0 `WRN-28-10` checkpoint (single view, no TTA):

- Raw image: `94.227%` overall / `87.170%` on the noisiest $20\%$ deciles
- Exact `25/245` $3 \times 3$ median repair: **`94.559%` (`+0.332%`)** overall / **`87.925%` (`+0.755%`)** on the noisiest $20\%$
- `0.3 raw + 0.7 repaired`: **`94.594%` (`+0.367%`)** overall / **`88.017%` (`+0.847%`)** on the noisiest $20\%$

`run_phase2_impulse_hammer.sh` combines this exact impulse repair (`--clean-impulses`) with strictly fold-safe ~195k-image transductive banks (`build_fold_transductive_banks.py`, combining high-confidence unlabeled and test sprites from bidirectional cavity inference) to upgrade all four backbones (`DenseNet-169`, `ConvNeXt V2`, `WRN-40-10`, `WRN-28-10`) across all $3$ folds.

---

## 6. Quantitative Progression

### Image Recognition Cross-Validation ($160{,}925$ Held-Out Labeled Sprites)

| Stage | Tracked summary | OOF Top-1 | Candidate Pool Recall |
|---|---|---:|---:|
| Small contrastive CNN (local baseline) | `outputs/image_shift1_20e_cv.json` | $68.42\%$ | — |
| `WRN-28-10` + mild ensemble + $\pm 5^\circ$ TTA (6 ckpts) | `results/wrn_pair_tta_cv.json` | $93.89\%$ | $98.73\%$ (top-10) |
| + 3-arch (`WRN + ConvNeXtV2 + DenseNet`) LightGBM ranker | `results/meta_rerank_three_arch_top20_cv.json` | $94.91\%$ | $99.24\%$ (top-20) |
| 9-WRN BP pseudo-label ensemble (`o + m + 2p`) + rebuilt ranker | `results/meta_rerank_pseudo_top20_cv.json` | $95.25\%$ | $99.35\%$ (top-20) |
| 15-WRN continuation ensemble (`v2`) + rebuilt ranker | `results/meta_rerank_v2_top20_cv.json` | $95.43\%$ | $99.42\%$ (top-20) |
| **4-backbone rot5 `top-40` + expanded kNN + Listwise DeepSets/LGB (`Phase 1`)** | `results/phase1_ranker_cv.json` | **$95.59\%$** | **$99.74\%$ (top-40)** |

### End-to-End Battle Score Progression

Fold-0 end-to-end scores strictly hold out all fold-0/fold-0 battles (rankers fit on folds $1 + 2$; score tables exclude fold-0/fold-0 battles; hyperparameters selected on tune half, metrics reported on test half).

| Pipeline configuration | Tracked summary | Fold-0 Both-Correct | Fold-0 Soft Score | Kaggle Public LB |
|---|---|---:|---:|---:|
| Earlier baseline submission | — | — | — | $58.90425$ |
| 6-WRN 3-arch ranker + supervised `PairMLP` | `results/score_fold0_mlp_soft_v2.json` | — | $64.20$ | — |
| + Semi-supervised `PairMLP` | `results/score_fold0_e2e_oldranker_semi_old.json` | — | $65.13$ | — |
| + 9-WRN pseudo-label ensemble & ranker | `results/score_fold0_e2e_newranker_semi_old.json` | $90.11\%$ | $66.28$ | $68.21380$ |
| + 15-WRN `v2` ranker (independent soft) | `results/score_fold0_e2e_v2ranker_semi_new.json` | $90.87\%$ | $66.67$ | — |
| + Global unordered-pair exclusivity & Bethe cavity (`top-20`) | `results/score_fold0_constrained_v2.json` | $93.15\%$ | $71.23$ | **$73.50$** |
| **+ 4-backbone `top-40` + Listwise DeepSets + bidirectional cavity (`Phase 1`)** | `results/score_fold0_phase1_fastattack.json` | **$93.53\%$** | **$71.99$** | *(ready: `submission_phase1_fastattack.csv`)* |

---

## 7. Reproducing the Pipeline

Install dependencies (`pip install -r requirements_large.txt -r requirements_rerank.txt`) and cache labeled/unlabeled $32 \times 32$ arrays (`image_experiment.py`, `cache_unlabeled_images.py`).

1. **Base backbones & BP semi-supervised `WRN-28-10` + `PairMLP`:**
   ```bash
   PYTHON_BIN=python3 ./run_pseudo_wrn_queue.sh
   PYTHON_BIN=python3 ./run_pseudo_ensemble_queue.sh
   PYTHON_BIN=python3 ./run_synth_all_folds_queue.sh
   ```
2. **Global constrained Bethe cavity inference (`73.50 LB`):**
   ```bash
   PYTHON_BIN=python3 ./run_constrained_v3_queue.sh
   ```
3. **Phase 1 Fast Attack (`71.99` Fold-0 / `95.59%` OOF) & Phase 2 Impulse-Repaired Hammer:**
   ```bash
   PYTHON_BIN=python3 ./run_fast_and_hammer_queue.sh
   PYTHON_BIN=python3 ./run_phase2_impulse_hammer.sh
   ```

---

## 8. Key Repository Modules

- `train_large_image.py`, `train_large_image_external.py`: 3-fold training for `WRN-28-10`, `WRN-40-10`, `ConvNeXt V2 Nano`, and CIFAR-stem `DenseNet-169` with external/transductive banks and `--clean-impulses` physical impulse repair.
- `degradation.py`, `build_degradation_data.py`, `check_degradation_realism.py`, `evaluate_noise_bins.py`: physical degradation estimation, GPU $3 \times 3$ non-impulse median repair (`clean_impulses_gpu`), realism audit, and noise-decile evaluation.
- `export_probs_rot5.py`, `extract_gallery_features_v2.py`, `train_listwise_ranker.py`: 4-backbone $\pm 5^\circ$ TTA probability export, top-40 variable-gallery kNN/prototype extraction, and permutation-equivariant DeepSets + LightGBM listwise ranking.
- `train_pair_mlp.py`, `train_pair_semi.py`, `joint_identity_inference.py`: antisymmetric binomial `PairMLP` and unlabeled battle-graph BP.
- `constrained_inference.py`, `joint_constrained_inference.py`, `build_fold_transductive_banks.py`: global unordered-pair exclusivity, Sinkhorn cardinality-constrained BP, bidirectional $\mathcal{D}_{\mathrm{unl}} \leftrightarrow \mathcal{D}_{\mathrm{test}}$ Bethe cavity inference, and fold-safe transductive bank builder.
