# Tabular Classification (`tabular-classification`)

## Overview
Tabular Classification predicts categorical target labels from structured tabular datasets composed of heterogeneous numerical and categorical columns. While gradient boosted decision trees historically dominated this domain, modern tabular deep learning architectures utilize transformer encoders with column embeddings to capture complex inter-feature relationships. The architecture processes missing values, normalizes continuous features, and encodes high-cardinality categories into expressive shared embedding spaces. Tabular classification is a cornerstone of enterprise operations, driving automated credit risk underwriting, customer churn forecasting, medical diagnosis categorization, fraud detection in banking, and ad click-through rate optimization.

## Representative Model: `stanford-star/rt-j`

### Model profile
RT-J is an 85.56M-parameter relational transformer (BF16, ~1.7 GB) released in 2026 under Apache-2.0. It consumes a relational database—linked tables, schema, keys, and a time column—plus labeled in-context rows, and produces entity-level binary classification **or regression**. It is not designed for a single flat CSV.

### Architecture tailored to relational tables
The model has 12 blocks, d_model 512, 8 heads, d_ff 2048, and d_text 384. Text columns are embedded with `all-MiniLM-L12-v2`; cells and rows become tokens, and time/relation-aware attention lets information flow across linked tables. A Huber loss supports both classification and regression heads. This relational attention is the key difference from flat tabular transformers. See [transformers](roadmap.md#transformers-and-attention).

### Niche mechanism
RT-J tokenizes cells and rows, not columns: text cells are embedded by MiniLM (384-d), numeric cells are projected, and time/relation-aware attention lets a row attend to linked rows across tables. It is trained for in-context prediction—retrieve labeled rows, place them in the 512–8,192-cell context, and read the entity head. **Design takeaway:** relational schemas and retrieved labels are the architecture; a flat CSV transformer loses the joins.

### Training and task tailoring
It is pretrained on synthetic PluRel, then on THE JOIN (639 databases, 6,023 LLM tasks), with 512–8,192-cell contexts, random-walk retrieval, and context tuning/ensembling. In-context prediction requires no per-task training: labeled examples are retrieved and placed in the context.

### How it is used
Install `relational-transformer`, call `load_rt_model("stanford-star/rt-j")`, and preprocess data into RelBench Parquet plus a `manifest.yaml`. Use frozen in-context prediction or fine-tune; BF16 works on A100/L4.

### Fit and caveats
The `tabular-classification` tag is too narrow: the model also regresses and expects a relational schema, not a flat table. It needs in-context labels and compatible preprocessing. Reported results are #1 in-context on RelBench, with ~10× fewer examples than RDBLearn/LLM-SQL + TabICLv2/LightGBM, 2.3–5.0 better nMAE, +3.1 AUROC, and +4.1 nMAE/6.1 AUROC from retrieval; driver-DNF AUROC is 0.826. These are author-reported.

## Related Reading
- [Roadmap: transformers](roadmap.md#transformers-and-attention) and [contrastive learning](roadmap.md#contrastive-and-metric-learning).
- [Tabular Regression](tabular-regression.md) · [Table Question Answering](table-question-answering.md) · [Time Series Forecasting](time-series-forecasting.md)
