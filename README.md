# Pokéwars: Joint Visual Recognition, Graphical Identity Inference, and Antisymmetric Score Completion

Code and validation summaries for the CS 3780 / 5780 competition [The Pokéwar](https://www.kaggle.com/competitions/cs-3780-5780-fa-26-the-pokewar/). Given two noisy $32 \times 32$ grayscale sprites $(x_1, x_2)$, the task is to predict the outcome $y \in [0, 10]$ of a 1,024-round simulation between their latent Pokémon species.

Competition archives, model checkpoints, and per-image prediction arrays are excluded from Git (`.gitignore`); compact cross-validation and ablation summaries are tracked under `results/`.

---

## 1. Problem Formulation and Risk Decomposition

Let $\mathcal{C} = \{1, \dots, 1025\}$ denote the anonymous species set and $\mathcal{X} = \{0, \dots, 255\}^{32 \times 32}$ the space of noisy grayscale sprites. Across the competition files, the data partition into three disjoint image regimes:

| Split | Battles | Unique images | Species labels | Battle outcome $y$ | Empirical structure |
|---|---:|---:|---|---|---|
| $\mathcal{D}_{\mathrm{lab}}$ (`train.csv`, labeled) | $184{,}800$ | $160{,}925$ | observed $(c_1, c_2)$ | observed | $157$ images per species; distinct unordered pairs |
| $\mathcal{D}_{\mathrm{unl}}$ (`train.csv`, unlabeled) | $79{,}200$ | $68{,}675$ | latent | observed | $67$ images per species; mean degree $2.31$ |
| $\mathcal{D}_{\mathrm{test}}$ (`test.csv`) | $80{,}000$ | $160{,}000$ | latent | target | $1$ battle per image; disjoint from $\mathcal{D}_{\mathrm{train}}$ |

For an ordered species pair $(a, b) \in \mathcal{C}^2$ with true single-round win probability $p_{ab} \in [0, 1]$, the recorded battle target is the scaled empirical win count

$$
y = 10 \cdot \frac{W}{1024}, \qquad W \sim \mathrm{Binomial}(1024, p_{ab}), \qquad p_{ba} = 1 - p_{ab}.
$$

Even with known identities $(a, b)$, binomial sampling induces an irreducible floor $\mathbb{E}[(y - 10 p_{ab})^2] = \frac{100}{1024}\, p_{ab}(1 - p_{ab})$ (root-mean-square $\approx 0.12$–$0.15$). Predictions $\hat{y}$ are scored on Kaggle by normalized RMSE:

$$
\mathrm{Score}(\hat{y}, y) = 100 \left(1 - \frac{\mathrm{RMSE}(\hat{y}, y)}{D}\right), \qquad D = 4.382900214041612.
$$

### Bayes-Optimal Prediction under Identity Uncertainty

Let $T \in [0, 10]^{1025 \times 1025}$ denote an estimate of the conditional expectation $T_{ab} = 10\, p_{ab}$, and let $q(c_1, c_2 \mid x_1, x_2)$ be the posterior distribution over latent species given the two images. Under squared-error loss, the pointwise risk-minimizing predictor is the posterior expectation rather than the argmax plug-in $T_{\hat{c}_1, \hat{c}_2}$:

$$
\hat{y}_{\mathrm{soft}}(x_1, x_2)
= \mathbb{E}_{(c_1, c_2) \sim q(\cdot \mid x_1, x_2)}\!\left[T_{c_1, c_2}\right]
\approx \sum_{c_1 \in \mathcal{K}(x_1)} \sum_{c_2 \in \mathcal{K}(x_2)} q_1(c_1 \mid x_1)\, q_2(c_2 \mid x_2)\, T_{c_1, c_2},
$$

where $\mathcal{K}(x)$ is the top-$K$ ($K = 20$) candidate set retained per image. Decomposing MSE by the event $E = \{(\hat{c}_1, \hat{c}_2) = (c_1, c_2)\}$ explains why both visual accuracy and posterior expectation are decisive:

$$
\mathrm{MSE} = \mathbb{P}(E)\,\mathbb{E}\!\left[(\hat{y} - y)^2 \mid E\right] + \big(1 - \mathbb{P}(E)\big)\,\mathbb{E}\!\left[(\hat{y} - y)^2 \mid E^c\right].
$$

On held-out fold-0 battles (`results/score_fold0_e2e_newranker_semi_old.json`), battles with at least one misclassified sprite comprise only $1 - \mathbb{P}(E) = 9.89\%$ of pairs ($1{,}045 / 10{,}571$), yet exhibit hard-plug-in conditional RMSE $5.53$ versus $0.281$ when both sprites are correct—accounting for $>95\%$ of hard-prediction MSE. Marginalizing over top-20 posteriors reduces conditional RMSE on $E^c$ from $5.53$ to $4.28$, lifting the end-to-end score by $+6.39$ points without changing any model weights.

---

## 2. Visual Recognition and Candidate Reranking

### Fold Protocol

All $160{,}925$ labeled sprites are partitioned by unique image path into $3$ class-stratified folds (`stratified_folds` in `image_experiment.py`, seed $3780$). Every backbone, kNN feature bank, pseudo-label bank, and gradient-boosted ranker for fold $f \in \{0, 1, 2\}$ is fit strictly on folds $\{0, 1, 2\} \setminus \{f\}$.

### Architectures, Standardization, and TTA

Each raw image $x \in [0, 255]^{32 \times 32}$ is normalized per instance to stabilize contrast drift across noise levels:

$$
\tilde{x} = \frac{x / 255 - \mu(x)}{\max(\sigma(x),\, 0.08)}.
$$

Three architecture families are trained on $\mathcal{D}_{\mathrm{lab}}$ (`train_large_image.py`, `train_large_image_external.py`) using AdamW, label smoothing $\alpha = 0.03$, gradient clipping ($\|\nabla\|_2 \le 1$), on-the-fly affine jitter, and a two-stage cosine schedule ($40$ epochs from $8 \times 10^{-4}$, followed by continuation from $10^{-4}$ to $10^{-7}$):

1. **Native $32 \times 32$ `WideResNet-28-10`** ($37.1\text{M}$ parameters, single-channel $3 \times 3$ stride-$1$ stem, trained under two augmentation regimes: `wrn28_10` and milder `wrn28_10_mild`, plus semi-supervised and noise-augmented continuations).
2. **`ConvNeXt V2 Nano`** (bicubic upsampling to $64 \times 64$ to preserve spatial resolution at stride $32$).
3. **CIFAR-stem `DenseNet-169`** ($3 \times 3$ stride-$1$ stem with initial pooling removed).

At inference, WRN logits are converted to probabilities and averaged across deterministic rotation test-time augmentation $\Theta = \{-5^\circ, 0^\circ, +5^\circ\}$:

$$
p_{\mathrm{tta}}(c \mid x) = \frac{1}{|\Theta|} \sum_{\theta \in \Theta} \mathrm{softmax}\!\left(z(R_\theta \tilde{x})\right)_c.
$$

### Deep-Feature kNN Gallery and Multi-Architecture Candidate Ranker

For each WRN checkpoint trained outside fold $f$, penultimate global-pooled features $u(x) = \phi(x) / \|\phi(x)\|_2 \in \mathbb{S}^{639}$ are extracted over its training gallery $\mathcal{G}_{\setminus f}$. For each candidate $c \in \mathcal{K}(x)$ in the WRN top-20 pool (`feature_rerank.py`), we compute:

$$
s_{\mathrm{knn1}}(x, c) = \max_{i \in \mathcal{G}_{\setminus f, c}} u(x)^\top u(x_i), \qquad
s_{\mathrm{knn5}}(x, c) = \frac{1}{5} \sum_{i \in \mathcal{N}_5(x, c)} u(x)^\top u(x_i), \qquad
s_{\mathrm{proto}}(x, c) = u(x)^\top \bar{u}_c.
$$

A cross-fitted LightGBM binary classifier (`evaluate_meta_rerank.py`, `predict_meta_top20_test.py`) scores all $20$ candidates per image using $19$ features combining log-probabilities, rank indicators, and margins from WRN, ConvNeXt V2, and DenseNet-169 together with $(s_{\mathrm{knn1}}, s_{\mathrm{knn5}}, s_{\mathrm{proto}})$ and the categorical species index $c$.

---

## 3. Antisymmetric Binomial Pair Model (`PairMLP`)

Because each unordered species pair $\{a, b\}$ appears at most once in $\mathcal{D}_{\mathrm{lab}}$ ($184{,}800$ observed pairs out of $\binom{1025}{2} = 524{,}800$, density $35.2\%$), estimating $T \in [0, 10]^{1025 \times 1025}$ is a structured matrix-completion problem. Low-rank bilinear or Bradley–Terry parameterizations underfit nontransitive type interactions (rock-paper-scissors cycles across elemental types).

In `train_pair_mlp.py`, each species $a \in \mathcal{C}$ is represented by an embedding $e_a \in \mathbb{R}^d$ ($d = 24$), mapped through a 3-layer SiLU/BatchNorm/Dropout MLP $g_\theta : \mathbb{R}^{3d} \to \mathbb{R}$:

$$
h_\theta(a, b) = g_\theta\!\left([e_a \,\|\, e_b \,\|\, e_a \odot e_b]\right), \qquad
\ell_\theta(a, b) = h_\theta(a, b) - h_\theta(b, a), \qquad
T_\theta(a, b) = 10\,\sigma\!\left(\ell_\theta(a, b)\right).
$$

By construction, $\ell_\theta(b, a) = -\ell_\theta(a, b)$, which enforces exact skew-symmetry $T_\theta(a, b) + T_\theta(b, a) = 10$ and $T_\theta(a, a) = 5$ for all $\theta$. Because $y / 10$ is an empirical binomial proportion, $\theta$ is trained by minimizing weighted binary cross-entropy with random orientation flipping $(a, b, y) \mapsto (b, a, 10 - y)$:

$$
\mathcal{L}_{\mathrm{BCE}}(\theta) = -\sum_{(a, b, y, w)} w \left[\frac{y}{10} \log \sigma(\ell_\theta(a, b)) + \left(1 - \frac{y}{10}\right) \log\big(1 - \sigma(\ell_\theta(a, b))\big)\right].
$$

Switching from factorized bilinear regression to `PairMLP` trained with $\mathcal{L}_{\mathrm{BCE}}$ reduces held-out known-ID RMSE from $\approx 0.90$ to $0.371$ (and to $0.250$ after adding graphical pseudo-battles below).

---

## 4. Graphical Identity Inference and Semi-Supervised Co-Training

### Loopy Belief Propagation on $\mathcal{D}_{\mathrm{unl}}$

The $79{,}200$ unlabeled training rows form a factor graph $\mathcal{G} = (\mathcal{V}_{\mathrm{unl}}, \mathcal{E}_{\mathrm{unl}})$ over $|\mathcal{V}_{\mathrm{unl}}| = 68{,}675$ unlabeled sprites, where each edge $(u, v) \in \mathcal{E}_{\mathrm{unl}}$ carries an observed battle score $y_{uv} \in [0, 10]$ (mean node degree $2 |\mathcal{E}_{\mathrm{unl}}| / |\mathcal{V}_{\mathrm{unl}}| = 2.307$). Restricting each node $u$ to its top-$K$ ($K = 20$) visual candidates $\mathcal{K}_u$, the joint posterior over latent assignments $\mathbf{c} = (c_u)_{u \in \mathcal{V}_{\mathrm{unl}}}$ factorizes as

$$
P(\mathbf{c} \mid \mathbf{x}, \mathbf{y}) \propto \prod_{u \in \mathcal{V}_{\mathrm{unl}}} \psi_u(c_u) \prod_{(u, v) \in \mathcal{E}_{\mathrm{unl}}} \psi_{uv}(c_u, c_v;\, y_{uv}),
$$

with unary visual prior $\psi_u(c) = p_{\mathrm{img}}(c \mid x_u)^\gamma$ and outlier-robust Gaussian-uniform pairwise likelihood (`joint_identity_inference.py`):

$$
\psi_{uv}(c_1, c_2;\, y_{uv}) = (1 - \varepsilon)\,\frac{1}{\sqrt{2\pi}\sigma} \exp\!\left(-\frac{(y_{uv} - T_{c_1, c_2})^2}{2\sigma^2}\right) + \frac{\varepsilon}{10},
$$

using $(\sigma, \varepsilon, \gamma) = (0.5, 0.05, 1.0)$. Damped sum-product message passing in log-domain ($\lambda = 0.5$, $20$ iterations) updates directed edge messages according to

$$
\tilde{m}_{u \to v}^{(t)}(c_v) \propto \sum_{c_u \in \mathcal{K}_u} \psi_{uv}(c_u, c_v;\, y_{uv})\,\psi_u(c_u) \prod_{w \in \mathcal{N}(u) \setminus \{v\}} m_{w \to u}^{(t-1)}(c_u),
\qquad
\log m_{u \to v}^{(t)} = (1 - \lambda)\log m_{u \to v}^{(t-1)} + \lambda \log \tilde{m}_{u \to v}^{(t)}.
$$

On a leak-free fold-0 simulation (`results/joint_identity_sim.json`, $54{,}325$ fold-0 images and $21{,}141$ fold-0/fold-0 battles scored with `pair_table_no_fold0.npy`), belief propagation lifts top-1 accuracy from $94.00\%$ to $95.44\%$ overall, scaling monotonically with battle degree: $94.12\%$ (deg $0$), $96.03\%$ (deg $1$), $96.91\%$ (deg $2$), and $98.09\%$ (deg $\ge 3$).

### Closed-Loop Semi-Supervised Co-Training

Belief propagation beliefs $b_u(c_u) \propto \psi_u(c_u) \prod_{v \in \mathcal{N}(u)} m_{v \to u}(c_u)$ drive two coupled semi-supervised updates:

1. **Semi-supervised score completion (`train_pair_semi.py`).** Setting $\hat{c}_u = \arg\max_c b_u(c)$ and confidence $\hat{\pi}_u = b_u(\hat{c}_u)$, unlabeled battles with joint confidence $w_{uv} = \hat{\pi}_u \hat{\pi}_v \ge 0.5$ ($77{,}549 / 79{,}200$ battles, mean weight $0.975$) are appended to $\mathcal{D}_{\mathrm{lab}}$ with loss weight $w_{uv}$. On the strict 80/20 holdout protocol (`results/pair_semi_clean.json`), one round reduces held-out `PairMLP` RMSE from $0.4666 \to 0.3264$; on the fold-0 end-to-end split, true-ID score rises from $89.39 \to 93.61$.
2. **Fold-safe semi-supervised WRN training (`build_pseudo_label_bank.py`, `run_pseudo_wrn_queue.sh`).** To prevent indirect validation leakage, fold $f$'s unlabeled bank is generated exclusively from fold-$f$ WRN checkpoints (trained without fold-$f$ images). Unlabeled sprites with $\hat{\pi}_u \ge 0.90$ ($\approx 63{,}280$ images per fold) are mixed into each training mini-batch at fraction $\rho_{\mathrm{ext}} = 0.35$, producing `wrn28_10_mild_pseudo_f{0,1,2}_stage2_best.pt`. Combining out-of-fold probabilities with weights $(0.25, 0.25, 0.50)$ across `(wrn28_10, wrn28_10_mild, wrn28_10_mild_pseudo)` raises WRN OOF top-1 from $93.89\% \to 94.61\%$ (`results/multibackbone_wrn_pseudo_rot5_oof.json`) and LightGBM ranker OOF top-1 from $94.91\% \to 95.25\%$ (`results/meta_rerank_pseudo_top20_cv.json`). This pipeline produced the **68.21380** Kaggle leaderboard submission (`run_pseudo_ensemble_queue.sh`).

---

## 5. Parametric Degradation Modeling and 15-Checkpoint Continuation (`v2`)

Stratifying validation accuracy by Immerkaer's Laplacian noise estimator $\hat{\sigma}_{\mathrm{lap}}(x)$ reveals that the cleanest $40\%$ of sprites already achieve $99.3\%$–$99.7\%$ top-1 accuracy, whereas $89\%$ of classification errors lie in the noisier half (`results/synth_pilot_noise_bins.json`). Pixel-level inspection shows a structured degradation process (`degradation.py`):

$$
x_{\mathrm{clean}} \;\mapsto\; x_0 = \mathrm{clip}\!\left(\mathrm{round}\big(b - (b - f)\, s(x_{\mathrm{clean}}) + \sigma_{\mathrm{add}}\,\eta\big),\, 0, 255\right), \qquad \eta \sim \mathcal{N}(0, I),
$$

followed by discrete impulse corruption that overwrites pixels with exact intensities $25$ (rate $p_{\mathrm{low}}$) and $245$ (rate $p_{\mathrm{high}}$), alongside positive coupling between noise severity and sprite scale/translation. `build_degradation_data.py` extracts fold-safe clean silhouette maps $s \in [0, 1]^{32 \times 32}$ and empirically estimated 5-tuples $(b, f, \sigma_{\mathrm{add}}, p_{\mathrm{low}}, p_{\mathrm{high}})$ from noisy training images.

Fine-tuning all three folds under both 50% on-the-fly re-degradation (`wrn28_10_mild_pseudo_synth`) and matched continuation (`wrn28_10_mild_pseudo_ctrl`) yields a 5-prefix, 15-checkpoint WRN ensemble (`run_synth_all_folds_queue.sh`) with weights $(1, 1, 2, 2, 2)$, reaching **94.85%** WRN OOF top-1 ($99.42\%$ top-20 recall, `results/multibackbone_wrn_v2_rot5_oof.json`), **95.43%** LightGBM ranker OOF top-1 (`results/meta_rerank_v2_top20_cv.json`), and **66.67** on the held-out fold-0 end-to-end benchmark (`results/score_fold0_e2e_v2ranker_semi_new.json`).

---

## 6. Quantitative Progression

### Image Recognition Cross-Validation ($160{,}925$ Held-Out Labeled Sprites)

| Stage | Tracked summary | OOF Top-1 | OOF Top-5 | Candidate Pool Recall |
|---|---|---:|---:|---:|
| Small contrastive CNN (local M4 baseline) | `outputs/image_shift1_20e_cv.json` | $68.42\%$ | $84.41\%$ | — |
| `WRN-28-10` (stage 2, single view) | `results/wrn28_10_3fold_topk.json` | $92.51\%$ | $97.20\%$ | $98.19\%$ (top-10) |
| `WRN-28-10` mild augmentation | `results/wrn28_10_mild_3fold_topk.json` | $93.19\%$ | $97.57\%$ | $98.45\%$ (top-10) |
| 6-WRN ensemble (`orig + mild`) + $\pm 5^\circ$ TTA | `results/wrn_pair_tta_cv.json` | $93.89\%$ | $97.95\%$ | $98.73\%$ (top-10) |
| + Fold-safe penultimate kNN-1 reranking | `results/feature_knn_rerank_cv.json` | $94.32\%$ | $98.10\%$ | $98.73\%$ (top-10) |
| + 3-backbone (`WRN + ConvNeXtV2 + DenseNet169`) LightGBM ranker | `results/meta_rerank_three_arch_top20_cv.json` | $94.91\%$ | — | $99.24\%$ (top-20) |
| 9-WRN ensemble with BP pseudo-labels (`o + m + 2p`, $\pm 5^\circ$ TTA) | `results/multibackbone_wrn_pseudo_rot5_oof.json` | $94.61\%$ | $98.22\%$ | $99.35\%$ (top-20) |
| + Rebuilt kNN & 3-backbone LightGBM ranker (**68.21 LB**) | `results/meta_rerank_pseudo_top20_cv.json` | $95.25\%$ | — | $99.35\%$ (top-20) |
| 15-WRN ensemble (`o + m + 2p + 2synth + 2ctrl`, $\pm 5^\circ$ TTA) | `results/multibackbone_wrn_v2_rot5_oof.json` | $94.85\%$ | $98.37\%$ | $99.42\%$ (top-20) |
| + Rebuilt 15-WRN kNN & 3-backbone LightGBM ranker (`v2`) | `results/meta_rerank_v2_top20_cv.json` | **$95.43\%$** | — | **$99.42\%$** (top-20) |

### End-to-End Battle Score Progression

Fold-0 end-to-end evaluations strictly hold out all battles where both sprites belong to image fold 0 (`evaluate_score_fold0_mlp.py`: ranker fit on folds $1 + 2$; score table fit without fold-0/fold-0 pairs; first half used for hyperparameter selection, second half reported).

| Pipeline configuration | True-ID Score | Hard Top-1 Score | Soft Expected Score (Fold 0) | Kaggle Public LB |
|---|---:|---:|---:|---:|
| Earlier baseline submission | — | — | — | $58.90425$ |
| 6-WRN 3-arch ranker + supervised `PairMLP` (`score_fold0_mlp_soft_v2.json`) | $89.39$ | $57.15$ | $64.20$ | — |
| + Semi-supervised `PairMLP` (`score_fold0_e2e_oldranker_semi_old.json`) | $93.61$ | $57.77$ | $65.13$ | — |
| + 9-WRN pseudo-label ensemble & rebuilt ranker (`score_fold0_e2e_newranker_semi_old.json`) | $93.61$ | $59.89$ | $66.28$ | **$68.21380$** |
| + 15-WRN synth/ctrl continuation & `v2` ranker (`score_fold0_e2e_v2ranker_semi_new.json`) | $93.45$ | $59.52$ | **$66.67$** | *(ready: `submission_v2_ensemble.csv`)* |

---

## 7. Reproducing the Pipeline

### Environment and Data Cache

Place the competition archive at `data/competition.zip` and install dependencies (`pip install -r requirements_large.txt -r requirements_rerank.txt`). Cache the labeled, unlabeled, and test $32 \times 32$ arrays:

```bash
python image_experiment.py --fold 0 --epochs 1 --run-name cache_only
python cache_unlabeled_images.py
```

### Core Execution Queues

1. **Multi-backbone training (`WRN-28-10`, `ConvNeXt V2 Nano`, `DenseNet-169`):**
   ```bash
   python train_large_image.py --model wrn28_10 --fold 0 --epochs 40
   ./run_wrn_mild_queue.sh
   ```
2. **Initial supervised & semi-supervised `PairMLP` and fold-safe pseudo-label WRN training:**
   ```bash
   python train_pair_mlp.py --loss bce --epochs 1500 --patience 100
   python predict_score_submission.py --output outputs/large/submission_pair_mlp.csv
   ./run_pseudo_wrn_queue.sh
   python export_wrn_oof_probs.py --prefix wrn28_10_mild_pseudo \
     --output outputs/large/oof_wrn28_10_mild_pseudo_rot5.npy
   ```
3. **Rebuild OOF priors, BP semi-supervised pair tables, kNN features, LightGBM ranker, and 68.21 LB submission:**
   ```bash
   PYTHON_BIN=python3 ./run_pseudo_ensemble_queue.sh
   ```
4. **Parametric noise-degradation continuations across all 3 folds and `v2` ensemble submission:**
   ```bash
   PYTHON_BIN=python3 ./run_synth_all_folds_queue.sh
   ```

---

## 8. Key Repository Modules

- `train_large_image.py`, `train_large_image_external.py`: 3-fold training for `WRN-28-10`, `WRN-40-10`, `ConvNeXt V2 Nano`, and CIFAR-stem `DenseNet-169` with external pseudo-label banks and on-the-fly parametric noise synthesis.
- `degradation.py`, `build_degradation_data.py`, `check_degradation_realism.py`, `evaluate_noise_bins.py`: silhouette extraction, 5-parameter noise/impulse estimation, CNN discriminator realism audit, and noise-decile evaluation.
- `export_wrn_oof_probs.py`, `combine_oof_probs.py`, `predict_wrn_topk.py`: $\pm 5^\circ$ TTA probability export and weighted multi-prefix ensembling.
- `feature_rerank.py`, `predict_test_top20_features.py`, `evaluate_meta_rerank.py`, `export_ranker_fold0_scores.py`, `predict_meta_top20_test.py`: fold-safe penultimate kNN/prototype galleries and cross-fitted LightGBM candidate ranker.
- `train_pair_mlp.py`, `train_pair_semi.py`, `joint_identity_inference.py`, `build_pseudo_label_bank.py`: antisymmetric `PairMLP`, damped log-domain loopy belief propagation on the unlabeled battle graph, and semi-supervised pair/image bank generation.
- `evaluate_score_fold0_mlp.py`, `predict_score_submission.py`: leak-free fold-0 end-to-end score evaluation and posterior-expectation test submission generator.
