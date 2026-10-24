# Tabular Regression (`tabular-regression`)

## Overview
Tabular Regression predicts continuous numerical outcomes from structured rows containing tabular features. The model maps non-linear interactions across mixed data types, such as continuous measurements, categories, and timestamps, to an estimated numerical value. Modern transformer-based tabular architectures learn self-attention matrices across columns to uncover hidden synergies between financial ratios, sensor metrics, and demographic indicators. Typical applications include financial asset price forecasting, real estate valuation, demand forecasting, medical dosage optimization, and industrial sensor parameter prediction. Tabular regression models allow organizations to turn historical tabular data into high-value quantitative predictions.

## Representative Model: `pcoloc/autotrain-dragino-7-7-max_495m-1860863627`

### Model profile
This repository is a 1.18 MB `model.joblib` plus a 99-byte `config.json`, created in October 2022 by Hugging Face AutoTrain. The "495m" in the name is not a parameter count. It performs single-column regression from two numeric features, `rssi` and `snr`, to one continuous `target`. No license is stated.

### Architecture and training
The artifact is a joblib-serialized XGBoost regressor (`model_type: "xgboost"`), not a neural network or transformer. It has no tokenizer, GPU path, or pretrained backbone. Training used the `pcoloc/autotrain-data-dragino-7-7-max_495m` dataset with problem type "Single Column Regression"; hyperparameters, data size, split, objective, and training duration are not documented. The AutoTrain CO2 estimate is 0.0112 g.

### Niche mechanism
This is a two-feature XGBoost tree ensemble, not a transformer: `rssi` and `snr` are renamed to `feat_0`/`feat_1`, and the joblib model splits on thresholds rather than learning embeddings. Its entire capacity is a small set of trees, with no tokenizer or GPU path. **Design takeaway:** for tiny tabular data, start with gradient-boosted trees; a 495M/1B transformer is not automatically better.

### How it is used
Load it with `joblib.load('model.joblib')`, read `config.json`, select the `features`, rename columns to `feat_<n>`, and call `model.predict(data)`. It is CPU-only and has no HF pipeline or serving integration.

### Fit and caveats
The metadata's `library_name: transformers` is wrong: this is a tiny XGBoost artifact, not a transformer. It is a narrow two-feature observability model with weak validation performance (loss 72.730, R² 0.386, MSE 5,289.6, MAE 60.23, RMSLE 0.436 on one split), no baseline, and no license. It is unsuitable as a general tabular-regression foundation model and is included only because the Hub maps the task to it.

## Related Reading
- [Roadmap: model family notes](roadmap.md#model-family-notes) and [task-tag caveats](roadmap.md#task-tag-caveats).
- [Tabular Classification](tabular-classification.md) · [Time Series Forecasting](time-series-forecasting.md) · [Table Question Answering](table-question-answering.md)
