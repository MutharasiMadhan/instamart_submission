# Part B — Customer Segments and Lapse Prediction

## RFM checks

At snapshot `2025-10-01 00:00:00`, there are 1,175 customers with completed orders. Recency median/min/max is 19/0/180 days; frequency mean/max is 9.46/33; monetary mean/max is INR 2,552.06/INR 14,766.55.

## K-means choice

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

K=4 is selected because it has the highest silhouette score and the K=3 to K=4 inertia reduction is substantial while four segments remain actionable. Inertia always decreases with more clusters because each observation can be assigned to a closer centroid; the lowest inertia alone rewards complexity and does not tell us whether the segmentation is stable or useful to a campaign. PCA explained-variance ratios: 0.634 and 0.218, total 0.852.

## Segment profiles

See `segment_profiles.csv` for the exact profiles and recommended actions. Segment IDs are arbitrary labels, not ordered ranks. The Premium flag is assigned only to the segment with the highest mean monetary value.

## Lapse cohort and baseline

The cutoff is `2025-09-01`. Cohort: 1,156 customers; lapsed: 443 (38.32%); train/test: 924/232. The always-not-lapsed baseline has accuracy 0.616 and lapsed precision/recall/F1 of 0/0/0. No order dated on or after the cutoff enters model features, and Task 3 segment labels are not used as features. All encoders and scalers are fitted on training rows only; oversampling is also restricted to the training rows.

## Tuned models

| Model | Best parameters | Best CV F1 | Test confusion matrix (TN, FP / FN, TP) | Lapsed precision | Lapsed recall | Lapsed F1 |
|---|---|---:|---|---:|---:|---:|
| Logistic regression | C=100 | 0.651 | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | max_depth=8, min_samples_leaf=10 | 0.683 | [[114, 29], [31, 58]] | 0.667 | 0.652 | 0.659 |

The tuned random forest is selected because it has the stronger test F1 and precision while matching logistic regression's recall. Missing a lapsing customer risks losing future revenue, but wrongly targeting a loyal customer wastes campaign budget and may train customers to wait for discounts; the balance should be monitored after deployment.

## Oversampling experiment

Training-only oversampling balances the training classes at 570 each. Logistic regression before oversampling: precision 0.644, recall 0.652, F1 0.648. After oversampling: precision 0.589, recall 0.854, F1 0.697. Recall changes most, finding more potential lapsers while lowering precision and increasing the likely number of unnecessary contacts.

## Interpretation and artifacts

- Random-forest leading features: frequency, monetary, recency, average delivery minutes, and cancellation rate. See `random_forest_top_features.csv` for importances.
- Logistic coefficient signs and values are in `logistic_coefficient_signs.csv`. Positive means higher lapse log-odds, negative means lower lapse log-odds, conditional on other features.
- The selected pipeline is saved to `lapse_model.joblib`; reloaded test predictions match exactly.
- The top ten Premium-segment test customers by predicted lapse probability are in `retention_list.csv`; 2 of the 10 were actually lapsed in this run.

**Limitation:** The retention list is illustrative and not a leakage-free evaluation because its Premium flag comes from whole-window RFM, which includes the September label window; the model's own features remain pre-cutoff only.
