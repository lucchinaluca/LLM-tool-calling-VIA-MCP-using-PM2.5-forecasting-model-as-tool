# Thesis Empirical Benchmarking Report: Univariate Temporal Baseline Validations

**Project Focus:** Hourly $PM_{2.5}$ Urban Forecasting in Bath, UK (Full Continuous 1-Year Evaluation Window)  
**Reference Benchmark Study:** Ling Zeng et al., *iScience* (2026) — Chengdu Basin Meteorological/Pollutant Horizon Analysis

---

## 1. Overview of Experimental Paradigm

Unlike standard sequence-to-sequence evaluation setups that evaluate performance on isolated seasonal blocks or short time horizons, this benchmark assesses model robustness over an uninterrupted, continuous **1-year calendar cycle**. 

### Core Configuration Parameters:
* **Temporal Horizon:** 72-Hour Historical Sliding Lookback -> 24-Hour Non-Overlapping Forward Vector Forecast.
* **Data Resolution:** High-fidelity **hourly measurements** (capturing micro-scale diurnal fluctuations, traffic peak-idling windows, and overnight boundary-layer variations), contrasting with the lower-variance daily aggregates used in regional studies like Zeng et al. (2026).
* **Target Mode:** Strict **Univariate Baseline Sequence Tracking** (conditioned exclusively on historical $PM_{2.5}$ data and cyclic temporal embeddings).

---

## 2. Empirical Benchmark Matrix

The table below compiles the verified performance metrics across your five univariate configurations, ordered by incremental structural complexity and optimization.

| Model Configuration | Global $R^2$ Score | Global MAE ($mu g/m^3$) | Global RMSE ($mu g/m^3$) | MAE% (Normalized) | RMSE% (Normalized) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Darts Transformer** | 0.5526 | 2.4352 | 3.8541 | 32.0% | 51.0% |
| **Darts Transformer (Optimized)** | 0.5887 | 2.4413 | 3.6950 | 33.0% | 49.0% |
| **Base LSTM** | 0.5916 | 2.9622 | 4.1198 | 32.0% | 44.0% |
| **Transformer-LSTM Hybrid** | 0.6108 | 2.8617 | 4.0218 | 31.0% | 43.0% |
| **Transformer-LSTM Hybrid (Optimized)** | **0.6234** | 2.8119 | 3.9561 | **30.0%** | **42.0%** |

---

## 3. Deep Architectural & Scientific Insights

### A. The Transformer-LSTM Hybrid Superiority
The empirical results reveal that the **Transformer-LSTM Hybrid (Optimized)** achieves the highest variance-explanation capacity ($R^2 = 0.6234$) and the lowest scale-normalized errors ($MAE\% = 30\%$, $RMSE\% = 42\%$). This directly validates the architectural intuition established by Zeng et al. (2026):
* **Complementary Dynamics:** The multi-head self-attention layers within the Transformer encoder successfully map long-range seasonal patterns and global trend dynamics, effectively bypassing the memory-dilution risks that plague deep pure recurrent networks.
* **Sequential Regularization:** The trailing LSTM layer refines these macro-features by enforcing strict local step-to-step sequential constraints, stabilizing the 24-hour forecasting head against erratic phase shifts or extreme gradient variances.

### B. The MAE vs. RMSE Disconnect (The Residual Variance Phenomenon)
A fascinating structural behavior appears when comparing the **Darts Transformer** variants to the **LSTM**-driven models:
* The Transformers consistently yield significantly **lower absolute MAE values** (~2.43 vs. ~2.81-2.96 $mu g/m^3$).
* However, their **RMSE% values are noticeably higher** (49%-51% vs. 42%-44%), which drags down their overall $R^2$ scores.
* **The Scientific Explanation:** This indicates that the pure Transformer models excel at tracking the tight, low-amplitude baseline background pollution signals characteristic of spring and summer. However, because RMSE heavily penalizes large errors, the higher RMSE implies the pure Transformer suffers from localized, catastrophic overshoot or undershoot errors during highly chaotic, sporadic pollution events (e.g., winter atmospheric temperature inversions or abrupt micro-scale rush-hour gridlock). The recurrent memory of the LSTM, conversely, acts as an error-smoothing regularizer, capping peak error magnitude.

---

## 4. Methodological Roadmap: Strategic Next Steps

Your proposed progression provides a textbook structural framework for a rigorous Master's thesis methodology and discussion chapter:

1. **Replicating the SOTA Multivariate Configuration:** By adding the top three verified physical driving covariates from Zeng et al. (2026)—**Temperature, Surface Pressure, and Wind Speed**—directly into the Optimized Transformer-LSTM framework, you will isolate the exact predictive lift provided by local meteorology vs. pure temporal autoregression.
2. **Establishing the Localized Information Ceiling:** If your multivariate $R^2$ approaches or crosses the ~0.65 threshold across a *full continuous year* of hourly data, it proves your architecture has successfully achieved seasonal invariance, outperforming the paper's models which degraded sharply when exposed to data gaps exceeding three months.
3. **The GNN Transition Justification:** Any residual unexplained variance from Phase 2 will provide the ultimate mathematical justification for your Spatio-Temporal GNN. You can definitively argue that a purely localized model—even when augmented with weather inputs—cannot resolve the *open-system limitation* of urban air pollution. Passing spatial messages along physical topology (e.g., upwind sensor nodes, urban traffic networks, topography) will be positioned as the necessary step to break through the localized sequence tracking limits.
