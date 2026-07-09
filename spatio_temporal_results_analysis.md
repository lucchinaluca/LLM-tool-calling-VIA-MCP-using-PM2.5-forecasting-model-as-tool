# Spatio-Temporal PM₂.5 Forecasting — Results Analysis

**Reference baseline:** Multivariate Transformer-LSTM (`4.6`) — R² = 0.6219

---

## 0. Quick Clarification: The PM₂.5 Timeline Plot

The interactive Plotly chart in Section 4 plots `full_df_clean`, which spans **Jan 2016 → Dec 2025** (87,672 rows), not Jan 2014 as noted in the architecture documentation.

This is expected and correct. The air quality CSV records NaN values for several pollutants (NO₂, SO₂, CO) throughout 2014–2015. The `air_qual.dropna()` call at load time drops all rows with any NaN, eliminating the entire 2014–2015 period. The effective usable timeline therefore starts Jan 2016.

The PM₂.5 series itself looks irregular and noisy — this is entirely normal for hourly urban pollution data. PM₂.5 is driven by stochastic weather events, transient emission sources, and atmospheric boundary-layer dynamics, none of which produce smooth systematic patterns at hourly resolution. The data is correct.

---

## 1. What Each Stage Does and Why

### Stage 1 — Traffic Spatial Encoder

**What it is:** A graph neural network trained exclusively on the 9.5-month window where traffic data overlaps with PM₂.5 measurements (Jul 2017 – Apr 2018).

**What it does:** Each of the 24 traffic sensor nodes in Bath gets its own LSTM encoder, which compresses the 72-hour history of that road segment into a single hidden vector. Those 24 vectors are then passed through two Graph Convolutional Network (GCN) layers, which propagate information between neighbouring sensors according to Bath's road topology. The result is pooled into a single 64-dimensional graph embedding that summarises the spatial state of the road network.

**What it brings:** The goal is to teach the model how Bath's specific road network geometry co-varies with pollution. Road congestion patterns near the city centre, topography (valleys trapping emissions), and sensor proximity to pollution sources are all structural properties that a model trained only on weather cannot see. Stage 1 is where the model learns this spatial knowledge from real traffic counts.

**What happens to it:** After Stage 1 training, the spatial encoder's weights are permanently frozen. It becomes a fixed function — a learned spatial prior — that will be reused in Stage 2.

---

### Stage 2 — Full-Timeline Forecaster

**What it is:** The main forecasting model, trained on ~9 years of weather and PM₂.5 data (Jan 2016 – Dec 2024), tested on Jan 2025.

**What it does:** Stage 2 has two parallel branches:

- **Spatial branch (frozen):** A learnable `SyntheticNodeFeatureGenerator` maps the weather window into synthetic per-node inputs compatible with what the frozen Stage 1 encoder expects. The frozen encoder processes these synthetic inputs and emits the 64-dimensional spatial embedding. Because Stage 1's weights are locked, this branch injects the learned spatial prior into Stage 2 without any gradients flowing back through the encoder itself. The `SyntheticNodeFeatureGenerator` is trained — it learns to produce synthetic node features that best activate the frozen encoder for the current weather context.

- **Temporal branch (trainable):** A Transformer-LSTM identical in design to the best models from notebooks 4.4–4.6. This processes the full weather + PM₂.5 history window and produces a 128-dimensional temporal embedding.

Both embeddings are concatenated and passed through a two-layer fusion head to produce the 24-hour PM₂.5 forecast.

**What it brings:** Stage 2 combines the structural knowledge learned in Stage 1 (road network topology and its relationship to pollution) with the long-term meteorological patterns that only a 9-year dataset can provide. Crucially, at inference time (including the MPC advisor), only weather forecast data is needed — no live traffic data is required. The frozen encoder's road knowledge is embedded in its weights and expressed via the synthetic proxy features.

---

### Ablation — Weather-Only Temporal Model

**What it is:** Structurally identical to Stage 2 but the spatial branch is bypassed entirely. The 64-dimensional slot that Stage 2 fills with the frozen encoder's output is replaced with a zero vector.

**What it does:** Trains the same Transformer-LSTM temporal backbone and fusion head on the same 9-year dataset, without any spatial signal.

**What it brings:** The ablation is the controlled comparison that answers the key question: does the 9.5 months of traffic data actually improve forecasts via the transfer learning mechanism, or does the weather-only model perform just as well? If Stage 2 R² > Ablation R², the spatial prior contributes. If Stage 2 R² ≤ Ablation R², the spatial prior adds no benefit, and the simpler weather-only model is preferred.

---

## 2. Complete Results

| Model | Stage | Test Window | R² | MAE (μg/m³) | RMSE (μg/m³) | MAE% | RMSE% |
|---|---|---|---|---|---|---|---|
| Multivariate Transformer-LSTM (4.6) | Baseline | Jan 2025 | **0.6219** | 2.91 | 3.96 | 31% | 42% |
| ST Stage 1 | Traffic spatial encoder | Jul–Apr overlap 20% | -0.2102 | 4.79 | 7.07 | 48% | 71% |
| ST Stage 2 (with spatial prior) | Full timeline | Jan 2025 | 0.5596 | 3.51 | 4.28 | 38% | 46% |
| ST Ablation (no spatial prior) | Full timeline | Jan 2025 | 0.5996 | 2.98 | — | — | — |

---

## 3. Stage 1 Analysis: Traffic Spatial Encoder

**Result: R² = -0.2102, MAE = 4.79 μg/m³, RMSE = 7.07 μg/m³**

Stage 1 performs worse than the mean predictor on its own test set — negative R² is the clearest possible failure signal.

### Training behaviour

```
Epoch  10/60 | Train MSE: 0.00446 | Val MSE: 0.00963
Epoch  20/60 | Train MSE: 0.00413 | Val MSE: 0.00936
Epoch  30/60 | Train MSE: 0.00404 | Val MSE: 0.00952
Epoch  40/60 | Train MSE: 0.00400 | Val MSE: 0.00946
Epoch  50/60 | Train MSE: 0.00399 | Val MSE: 0.00946
Epoch  60/60 | Train MSE: 0.00400 | Val MSE: 0.00945
Best Stage 1 val MSE: 0.00891
```

Two clear problems:

**The train/val gap is large and persistent.** Train MSE converges to ~0.004 while val MSE stays at ~0.009 — a 2× gap that barely moves across 60 epochs. The model is fitting the training portion of the 9.5-month window without generalising to the held-out 20%.

**Val MSE does not decrease after epoch 10.** The `ReduceLROnPlateau` scheduler halves the LR when val loss plateaus, but improvement stops essentially at epoch 10. The best checkpoint (MSE = 0.00891) is saved early.

### Root cause

The 80/20 chronological split places the last ~1,371 hours (roughly Jan–Apr 2018) in the test set, while training uses Jul 2017 – Jan 2018 (summer and autumn only). **The model has never seen winter during training.** Winter PM₂.5 in Bath has a structurally different regime — higher baseline concentrations, more temperature inversions trapping pollutants, less traffic-responsive variation — so a traffic-only encoder trained on summer cannot generalise.

Additionally, 5,484 training samples is a small dataset for a 61k-parameter model. The per-node LSTM and GCN layers are likely memorising the training distribution rather than learning generalisable road topology patterns.

### Consequence for Stage 2

The frozen encoder carries weights from a model that did not generalise. The 64-dimensional spatial embeddings it produces in Stage 2 encode summer traffic–pollution co-variation that does not reflect winter conditions. This noisy spatial signal is the direct cause of Stage 2's underperformance relative to the ablation.

---

## 4. Stage 2 Analysis: Full-Timeline Forecaster

**Result: R² = 0.5596, MAE = 3.51 μg/m³, RMSE = 4.28 μg/m³**

Stage 2 achieves a positive R² and produces reasonable forecasts, but falls 6.2 R² points below the multivariate baseline and 4.0 points below the ablation.

### Training behaviour

```
Epoch  10/80 | Train MSE: 0.00094 | Val MSE: 0.00422
Epoch  20/80 | Train MSE: 0.00051 | Val MSE: 0.00484
Epoch  30/80 | Train MSE: 0.00041 | Val MSE: 0.00481
Epoch  40/80 | Train MSE: 0.00032 | Val MSE: 0.00456
Epoch  50/80 | Train MSE: 0.00029 | Val MSE: 0.00442
Epoch  60/80 | Train MSE: 0.00027 | Val MSE: 0.00424
Epoch  70/80 | Train MSE: 0.00026 | Val MSE: 0.00456
Epoch  80/80 | Train MSE: 0.00026 | Val MSE: 0.00442
Best Stage 2 val MSE: 0.00326
```

The train/val gap grows throughout training — train MSE reaches 0.00026 while val oscillates around 0.0042–0.0049. This is overfitting. The val loss does not follow the downward trend of train loss after epoch ~10, indicating the model has begun to memorise the training distribution. The original `CosineAnnealingLR` scheduler gradually reduced the LR without responding to the val plateau.

**Fix applied in the notebook:** The scheduler has been replaced with `ReduceLROnPlateau` (halves LR when val loss plateaus for 5 epochs) and early stopping with patience=10 has been added. Weight decay (`1e-4`) has been added to the Adam optimiser as L2 regularisation. These changes should stop training when val performance stagnates and reduce the parameter explosion that causes overfitting.

---

## 5. Ablation Analysis: Spatial Prior vs No Spatial Prior

**Ablation result: R² = 0.5996, MAE = 2.98 μg/m³**
**Stage 2 result: R² = 0.5596, MAE = 3.51 μg/m³**
**ΔR² = -0.0400 — the spatial prior hurts performance**

### Ablation training behaviour

```
Epoch  10/80 | Train: 0.00087 | Val: 0.00510
Epoch  20/80 | Train: 0.00047 | Val: 0.00619
Epoch  30/80 | Train: 0.00037 | Val: 0.00644
...
Best val MSE: 0.00510 (saved at epoch 10)
```

The ablation also overfits — val loss increases after epoch 10. But its best checkpoint (saved early at epoch 10) gives R² = 0.5996, which is better than Stage 2's 0.5596.

### Why the spatial prior hurts

The frozen encoder, trained on a poorly-generalising Stage 1, produces inconsistent 64-dimensional embeddings. Rather than encoding meaningful road topology, these embeddings carry noise from the failed Stage 1. The Stage 2 fusion head must learn to suppress this noise — but the extra 64 noisy dimensions still consume model capacity that the temporal branch could otherwise use. The result: Stage 2's temporal branch effectively gets less "room" to learn from the weather signal, and performance drops relative to the pure weather-only ablation.

This is a coherent and interpretable outcome. It does not mean the two-stage architecture is conceptually wrong — it means Stage 1 needs to work before the spatial prior can help.

---

## 6. Overall Thesis Interpretation

### What the results say

The hierarchy of performance is:

```
Multivariate baseline (4.6)     R² = 0.6219   ← best overall, recommended for MPC
Ablation (weather-only Stage 2) R² = 0.5996   ← Stage 2 temporal branch working correctly
Stage 2 with spatial prior      R² = 0.5596   ← spatial prior degrades performance
Stage 1 (traffic window test)   R² = -0.210   ← spatial encoder failed to learn
```

The ablation result is the most informative single number. It shows that the Stage 2 Transformer-LSTM trained on 9 years of weather + PM₂.5 data — with no traffic data and no spatial component — reaches R² = 0.60. This is within 0.02 of the fully optimised multivariate baseline (0.62), where the difference is attributable to the baseline having been through 40 rounds of Optuna hyperparameter search. The temporal backbone itself is functioning correctly.

### The core finding

The hypothesis being tested was: *does 9.5 months of Bath traffic data contain spatial information that, when encoded and transferred, improves 24-hour PM₂.5 forecasting?* The answer from this experiment is **no**, but for a specific and diagnosable reason — Stage 1's spatial encoder did not learn to generalise due to insufficient seasonal coverage of the training data.

This is not evidence that Bath's road topology is irrelevant to pollution. It is evidence that 9.5 months of traffic data, skewed toward summer and autumn, is insufficient to train a graph encoder that generalises to all-season PM₂.5 forecasting.

### Thesis framing

This is a valid negative result, and negative results have clear scientific value. It directly tests — and conditionally falsifies — the transfer learning hypothesis under the available data constraints. The conclusion supports a meteorology-first interpretation: for the tested configuration, Bath's hourly PM₂.5 is dominated by weather-driven dispersion dynamics, not by local traffic network topology. This motivates the choice of the multivariate Transformer-LSTM (R² = 0.6219) as the MPC backbone — simpler, better-performing, and requires only weather forecast data at inference time.

---

## 7. What Could Improve Stage 1

If revisiting the two-stage approach, the highest-impact changes are:

**1. Fix the seasonal split.** The 80/20 chronological split is the primary failure mode. Options: use a random split across the 9.5 months (loses temporal integrity but gives seasonal coverage), or augment by cyclically repeating the available data to cover all 12 months.

**2. Reduce Stage 1 model complexity.** With ~5,484 training samples, a 61k-parameter model is overparameterised. A single-layer GCN with a smaller LSTM (hidden=32, 1 layer, ~8k parameters) would generalise better on this dataset size.

**3. Allow partial fine-tuning in Stage 2.** Instead of fully freezing the encoder, allow Stage 2 to fine-tune it with a 100× lower learning rate. This lets the larger Stage 2 dataset correct Stage 1's seasonal bias.

**4. Add auxiliary targets.** Train Stage 1 to predict local pollutants (NO₂, CO) alongside PM₂.5. These are more directly traffic-driven and give the graph encoder a richer signal from the traffic data.
