# Time Series Forecasting (`time-series-forecasting`)

## Overview
Time Series Forecasting predicts future sequential values based on historical data points observed at regular time intervals. Modern architectures employ patch-based temporal transformers, frequency-domain neural operators, and autoregressive foundation models to capture seasonal rhythms, trend shifts, and cyclical patterns. Unlike standard tabular models, time-series architectures preserve temporal ordering and causality across extended prediction horizons. These models handle multi-variate dependencies and irregular sensor sampling intervals. This task is essential for electrical grid demand forecasting, supply chain inventory management, macroeconomic metric projections, server traffic capacity planning, and weather forecasting.

## Representative Model: `Datadog/Toto-2.0-2.5B`

### Model profile
Toto-2.0-2.5B is a 2.454B-parameter time-series foundation model (FP32, ~9.1 GB) released in 2026 under Apache-2.0. It takes a numeric multivariate context `(batch, variates, time)` with optional mask and series IDs and returns nine quantiles per forecast step. There is no text or covariate input.

### Architecture tailored to forecasting
It is a decoder-only patched transformer: 48 layers, 2048 hidden, 32 heads, d_ff 5464, patch size 32. Attention alternates between time-causal and variate-full patterns, with contiguous patch masking, a nine-level quantile head, an arcsinh scaler, u-μP parameterization, and the NorMuon optimizer. Causal time attention preserves forecasting order, while variate-full attention captures cross-series dependencies. See [transformers](roadmap.md#transformers-and-attention).

### Niche mechanism
Toto-2 patches the time axis (patch 32) and alternates time-causal attention (no future leaks) with variate-full attention (cross-series mixing). A nine-quantile head and pinball loss give calibrated uncertainty directly; arcsinh scaling tames heavy tails, and u-μP/NorMuon stabilize training from a 10M proxy. **Design takeaway:** for forecasting, encode time as patches, keep the time axis causal, mix variates deliberately, and predict quantiles.

### Training and task tailoring
Training used 5.04T points for the 313M/1B/2.5B family, split 42.5% Datadog observability and 57.5% synthetic TempoPFN, with no public pretraining data. Contexts are 4,096 steps with up to 32 variates, batch 64, and 600,500 steps, optimized with pinball loss. u-μP transfers from a 10M proxy; the 2.5B-FT variant adds 45% public data.

### How it is used
Install `toto-models`, load `Toto2Model.from_pretrained("Datadog/Toto-2.0-2.5B")`, and call `forecast({"target", "target_mask", "series_ids"}, horizon=96, decode_block_size=768)`. Output is `(9, batch, variates, horizon)` at quantiles 0.1–0.9. It is FP32 and takes about 36.2 ms for 1,024 steps at batch 8 on an A100.

### Fit and caveats
The tag is correct, but this is not a Transformers model; it needs the custom `toto2` runtime. FP32 weights are 9.1 GB, the training context is 4,096 steps, input is numeric only, and the pretraining data is not public. Reported BOOM CRPS/MASE is 0.349/0.581, GIFT-Eval 0.476/0.696, and TIME 0.532/0.640, with claimed #1 foundation-model status (CRPS ranks 3.88 and 3.43).

## Related Reading
- [Roadmap: transformers](roadmap.md#transformers-and-attention) and [Toto-2 family](roadmap.md#model-family-notes).
- [Tabular Regression](tabular-regression.md) · [Tabular Classification](tabular-classification.md) · [Table Question Answering](table-question-answering.md)
