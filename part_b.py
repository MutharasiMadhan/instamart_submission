"""Part B: RFM segmentation and leakage-aware lapse prediction.
Run from project root: python part_b.py
"""
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
SNAPSHOT = pd.Timestamp('2025-10-01 00:00:00')
CUTOFF = pd.Timestamp('2025-09-01 00:00:00')

orders = pd.read_csv(DATA / 'clean' / 'orders.csv', parse_dates=['order_ts'])
customers = pd.read_csv(DATA / 'clean' / 'customers.csv', parse_dates=['signup_date'])
products = pd.read_csv(DATA / 'clean' / 'products.csv')
orders['status'] = orders['status'].astype(str).str.lower().str.strip()
completed = orders[orders.status.eq('completed')].copy()

# Task 1: RFM over the full clean history, as at the snapshot.
rfm = completed.groupby('customer_id').agg(
    last_order_ts=('order_ts', 'max'),
    frequency=('order_id', 'nunique'),
    monetary=('order_amount', 'sum')
).reset_index()
rfm['recency'] = (SNAPSHOT - rfm['last_order_ts']).dt.days
rfm = rfm[['customer_id', 'recency', 'frequency', 'monetary']].copy()
print('\n=== TASK 1: RFM ===')
print(f'Customers={len(rfm)}; recency median/min/max={rfm.recency.median():.0f}/{rfm.recency.min()}/{rfm.recency.max()}; '
      f'frequency mean/max={rfm.frequency.mean():.2f}/{rfm.frequency.max()}; monetary mean/max={rfm.monetary.mean():.2f}/{rfm.monetary.max():.2f}')

# Task 2: scale RFM and compare K=2..8.
rfm_cols = ['recency', 'frequency', 'monetary']
scaler_rfm = StandardScaler()
X_rfm = scaler_rfm.fit_transform(rfm[rfm_cols])
k_rows = []
models = {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_rfm)
    sil = silhouette_score(X_rfm, labels)
    k_rows.append({'K': k, 'inertia': km.inertia_, 'silhouette': sil})
    models[k] = (km, labels)
k_table = pd.DataFrame(k_rows)
k_table.to_csv(OUT / 'kmeans_metrics.csv', index=False)
print('\n=== TASK 2: K-MEANS METRICS ===')
print(k_table.to_string(index=False, formatters={'inertia':'{:.1f}'.format,'silhouette':'{:.3f}'.format}))
chosen_k = 4
km, chosen_labels = models[chosen_k]
rfm['segment_id'] = chosen_labels

# Task 3: profile and give actionable, deterministic names; Premium is highest monetary mean.
profiles = rfm.groupby('segment_id').agg(
    size=('customer_id','size'), mean_recency=('recency','mean'),
    mean_frequency=('frequency','mean'), mean_monetary=('monetary','mean')
).reset_index()
premium_id = int(profiles.loc[profiles.mean_monetary.idxmax(), 'segment_id'])
# Naming is assigned from measurable segment profile, not the arbitrary KMeans label.
profiles['segment_name'] = ''
profiles['recommended_action'] = ''
for ix, row in profiles.iterrows():
    sid = int(row.segment_id)
    if sid == premium_id:
        name, action = 'Premium High-Value', 'Offer a personal loyalty benefit and early access; protect the relationship.'
    elif row.mean_frequency == profiles.mean_frequency.max():
        name, action = 'Frequent Loyalists', 'Use replenishment reminders and a subscription or streak reward.'
    elif row.mean_recency == profiles.mean_recency.max():
        name, action = 'Cooling Customers', 'Send a targeted win-back offer and ask what changed.'
    elif row.mean_monetary == profiles.mean_monetary.min():
        name, action = 'Occasional Value Seekers', 'Use relevant bundles and low-friction second-order incentives.'
    else:
        name, action = 'Occasional Value Seekers', 'Use relevant bundles and low-friction second-order incentives.'
    profiles.loc[ix, 'segment_name'] = name
    profiles.loc[ix, 'recommended_action'] = action
# Ensure unique names in any profile tie / overlap.
if profiles.segment_name.duplicated().any():
    for ix in profiles[profiles.segment_name.duplicated(keep=False)].index:
        profiles.loc[ix, 'segment_name'] = profiles.loc[ix, 'segment_name'] + f' (Group {int(profiles.loc[ix,"segment_id"])})'
name_map = dict(zip(profiles.segment_id.astype(int), profiles.segment_name))
rfm['segment_name'] = rfm.segment_id.map(name_map)
rfm['is_premium'] = rfm.segment_id.eq(premium_id)
segments_path = OUT / 'customer_segments.csv'
rfm[['customer_id','recency','frequency','monetary','segment_id','segment_name','is_premium']].to_csv(segments_path, index=False)
profiles.to_csv(OUT / 'segment_profiles.csv', index=False)
# PCA is fit to scaled full-window RFM, solely for descriptive visualisation.
pca = PCA(n_components=2)
pca.fit(X_rfm)
print('\n=== TASK 3: SEGMENT PROFILES (cluster IDs are arbitrary labels, not an ordinal ranking) ===')
print(profiles[['segment_id','segment_name','size','mean_recency','mean_frequency','mean_monetary','recommended_action']].to_string(index=False, formatters={'mean_recency':'{:.2f}'.format,'mean_frequency':'{:.2f}'.format,'mean_monetary':'{:.2f}'.format}))
print(f'Premium segment_id={premium_id}; PCA explained_variance_ratio_={np.round(pca.explained_variance_ratio_, 3).tolist()}, total={pca.explained_variance_ratio_.sum():.3f}')

# Task 4: cutoff cohort and labels. Features use only orders strictly before cutoff.
pre = orders[orders.order_ts < CUTOFF].copy()
pre_completed = pre[pre.status.eq('completed')].copy()
cohort_ids = pre_completed.customer_id.unique()
cohort = customers[customers.customer_id.isin(cohort_ids)].copy()
label_window = completed[(completed.order_ts >= CUTOFF) & (completed.order_ts < pd.Timestamp('2025-10-01'))]
post_counts = label_window.groupby('customer_id').order_id.nunique()
cohort['lapsed'] = (~cohort.customer_id.isin(post_counts.index)).astype(int)

# Behaviour features strictly from pre-cutoff orders.
pre_comp_agg = pre_completed.groupby('customer_id').agg(
    last_order_ts=('order_ts','max'), frequency=('order_id','nunique'),
    monetary=('order_amount','sum'), avg_discount_pct=('discount_pct','mean'),
    avg_delivery_minutes=('delivery_minutes','mean')
).reset_index()
pre_final = pre[pre.status.isin(['completed','cancelled'])]
final_counts = pre_final.groupby('customer_id').status.agg(total_final='size', cancelled=lambda x: int((x == 'cancelled').sum())).reset_index()
final_counts['cancellation_rate'] = final_counts['cancelled'] / final_counts['total_final'].replace(0, np.nan)
# Most frequently ordered category among completed pre-cutoff orders; ties resolved alphabetically.
cat_orders = pre_completed.merge(products[['product_id','category']], on='product_id', how='left')
cat_counts = cat_orders.groupby(['customer_id','category']).size().reset_index(name='n')
cat_counts = cat_counts.sort_values(['customer_id','n','category'], ascending=[True,False,True]).drop_duplicates('customer_id')
cat_counts = cat_counts[['customer_id','category']].rename(columns={'category':'favourite_category'})
features = cohort[['customer_id','city','age','gender','signup_date','lapsed']].merge(pre_comp_agg, on='customer_id', how='left').merge(final_counts[['customer_id','cancellation_rate']], on='customer_id', how='left').merge(cat_counts, on='customer_id', how='left')
features['recency'] = (CUTOFF - features['last_order_ts']).dt.days
features['tenure_days'] = (CUTOFF - features['signup_date']).dt.days
features['avg_delivery_minutes'] = features['avg_delivery_minutes'].fillna(0)
features['avg_discount_pct'] = features['avg_discount_pct'].fillna(0)
features['cancellation_rate'] = features['cancellation_rate'].fillna(0)
features['favourite_category'] = features['favourite_category'].fillna('Unknown').astype(str)
features['gender'] = features['gender'].fillna('Unknown').astype(str)
features['city'] = features['city'].fillna('Unknown').astype(str)
features['age'] = pd.to_numeric(features['age'], errors='coerce').fillna(features['age'].median())
num_cols = ['recency','frequency','monetary','avg_discount_pct','cancellation_rate','avg_delivery_minutes','age','tenure_days']
cat_cols = ['city','gender','favourite_category']
feature_cols = num_cols + cat_cols
assert not features[feature_cols].isna().any().any(), 'Feature columns contain missing values'
y = features['lapsed'].astype(int)
X = features[feature_cols].copy()
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, features.index, test_size=0.2, stratify=y, random_state=42)
print('\n=== TASK 4: COHORT ===')
print(f'Cohort={len(features)}; lapsed={int(y.sum())} ({y.mean()*100:.2f}%); train={len(X_train)}; test={len(X_test)}')
print(f'No missing values in engineered feature columns: {not X.isna().any().any()}')

# Task 5: training-only preprocessing in each Pipeline.
preprocessor = ColumnTransformer([
    ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                              ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))]), cat_cols)
], remainder='drop')
# For the requested frequency scaler mean, inspect the scaler fitted only on X_train.
scaler_probe = StandardScaler().fit(X_train[['frequency']])
print(f"Training-only fitted scaler mean(frequency)={scaler_probe.mean_[0]:.4f}; full-cohort mean(frequency)={X['frequency'].mean():.4f}")
print('These differ because the scaler is fitted on the training subset only, while the cohort mean includes the held-out test customers.')

# Task 6: baseline, grid searches, then one final untouched-test evaluation.
y_pred_base = np.zeros(len(y_test), dtype=int)
print('\n=== TASK 6: BASELINE (always not lapsed) ===')
print(f'Accuracy={accuracy_score(y_test,y_pred_base):.3f}; lapsed precision={precision_score(y_test,y_pred_base,zero_division=0):.3f}; '
      f'recall={recall_score(y_test,y_pred_base,zero_division=0):.3f}; F1={f1_score(y_test,y_pred_base,zero_division=0):.3f}')
log_pipe = Pipeline([('preprocess', preprocessor), ('model', LogisticRegression(max_iter=1000, random_state=42))])
log_grid = GridSearchCV(log_pipe, {'model__C':[0.01,0.1,1,10,100]}, cv=5, scoring='f1', n_jobs=-1, refit=True)
log_grid.fit(X_train, y_train)
rf_pipe = Pipeline([('preprocess', preprocessor), ('model', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))])
rf_grid = GridSearchCV(rf_pipe, {'model__max_depth':[3,5,8], 'model__min_samples_leaf':[5,10,20]}, cv=5, scoring='f1', n_jobs=-1, refit=True)
rf_grid.fit(X_train, y_train)
print('\n=== GRID SEARCH RESULTS (training CV only) ===')
print(f'Logistic best_params={log_grid.best_params_}; best_cv_f1={log_grid.best_score_:.3f}')
print(f'Random forest best_params={rf_grid.best_params_}; best_cv_f1={rf_grid.best_score_:.3f}')

def report_model(label, model):
    pred = model.predict(X_test)
    print(f'\n{label}: confusion matrix [[TN, FP], [FN, TP]] =\n{confusion_matrix(y_test,pred,labels=[0,1])}')
    print(f'lapsed precision={precision_score(y_test,pred,zero_division=0):.3f}; recall={recall_score(y_test,pred,zero_division=0):.3f}; F1={f1_score(y_test,pred,zero_division=0):.3f}')
    return pred
log_pred = report_model('TUNED LOGISTIC REGRESSION', log_grid.best_estimator_)
rf_pred = report_model('TUNED RANDOM FOREST', rf_grid.best_estimator_)
# Select RF if it has the stronger lapsed F1; persist this selected pipeline.
selected_name, selected_model, selected_pred = ('RandomForestClassifier', rf_grid.best_estimator_, rf_pred) if f1_score(y_test,rf_pred,zero_division=0) >= f1_score(y_test,log_pred,zero_division=0) else ('LogisticRegression', log_grid.best_estimator_, log_pred)
selected_f1 = f1_score(y_test, selected_pred, zero_division=0)
if selected_f1 < 0.55:
    warnings.warn(f'Selected model test F1 {selected_f1:.3f} is below the assignment target 0.55; review feature/seed differences.')
print(f'\nSelected model: {selected_name}, test lapsed F1={selected_f1:.3f}.')

# Task 7: duplicate lapsed training rows with replacement until classes are balanced.
train_df = X_train.copy()
train_df['_target'] = y_train.to_numpy()
class_counts = train_df['_target'].value_counts()
majority_n = int(class_counts.max())
minority_rows = train_df[train_df['_target'] == int(class_counts.idxmin())]
needed = majority_n - len(minority_rows)
extra = minority_rows.sample(n=needed, replace=True, random_state=42) if needed else minority_rows.iloc[0:0]
balanced = pd.concat([train_df, extra], ignore_index=True)
X_bal, y_bal = balanced[feature_cols], balanced['_target'].astype(int)
balanced_log = log_grid.best_estimator_
# Clone best estimator so the unbalanced fitted model remains unchanged.
from sklearn.base import clone
balanced_log = clone(log_grid.best_estimator_)
balanced_log.fit(X_bal, y_bal)
bal_pred = balanced_log.predict(X_test)
print('\n=== TASK 7: TRAIN-ONLY RANDOM OVERSAMPLING ===')
print(f'Balanced training class counts={y_bal.value_counts().sort_index().to_dict()}')
for nm, pred in [('Unbalanced logistic', log_pred), ('Balanced logistic', bal_pred)]:
    print(f'{nm}: precision={precision_score(y_test,pred,zero_division=0):.3f}; recall={recall_score(y_test,pred,zero_division=0):.3f}; F1={f1_score(y_test,pred,zero_division=0):.3f}')

# Task 8a: interpretable features for best RF and signs for best logistic.
rf_best = rf_grid.best_estimator_
rf_names = rf_best.named_steps['preprocess'].get_feature_names_out()
rf_importance = rf_best.named_steps['model'].feature_importances_
imp = pd.DataFrame({'feature':rf_names,'importance':rf_importance}).sort_values('importance',ascending=False).head(5)
imp.to_csv(OUT / 'random_forest_top_features.csv', index=False)
print('\n=== TASK 8: RANDOM FOREST TOP FIVE FEATURES ===')
print(imp.to_string(index=False, formatters={'importance':'{:.4f}'.format}))
log_best = log_grid.best_estimator_
log_names = log_best.named_steps['preprocess'].get_feature_names_out()
log_coef = log_best.named_steps['model'].coef_[0]
coef_df = pd.DataFrame({'feature':log_names,'coefficient':log_coef})
# Prefer three numeric features for a simple business interpretation.
coef_show = coef_df[coef_df.feature.isin(['numeric__recency','numeric__frequency','numeric__monetary','numeric__cancellation_rate','numeric__tenure_days'])].copy()
if len(coef_show) < 3:
    coef_show = coef_df.copy()
coef_show['abs_coef'] = coef_show.coefficient.abs()
coef_show = coef_show.sort_values('abs_coef',ascending=False).head(3).drop(columns='abs_coef')
coef_show.to_csv(OUT / 'logistic_coefficient_signs.csv', index=False)
print('\nLogistic regression three coefficients (positive means higher log-odds of lapse; negative means lower):')
print(coef_show.to_string(index=False, formatters={'coefficient':'{:+.4f}'.format}))

# Persist selected model and verify exact test prediction reproducibility.
model_path = OUT / 'lapse_model.joblib'
joblib.dump(selected_model, model_path)
reloaded = joblib.load(model_path)
reloaded_pred = reloaded.predict(X_test)
exact_match = bool(np.array_equal(selected_pred, reloaded_pred))
print(f'\nSaved and reloaded {selected_name}; exact test prediction match={exact_match}')
assert exact_match

# Premium list uses the task-3 whole-window segment flag post hoc, not as a model feature.
segment_flags = rfm[['customer_id','is_premium']]
test_customers = features.loc[idx_test, ['customer_id','lapsed']].copy()
test_customers = test_customers.merge(segment_flags, on='customer_id', how='left')
# Predicted lapse probability from selected model, column for positive class label 1.
positive_idx = list(reloaded.named_steps['model'].classes_).index(1)
proba = reloaded.predict_proba(X_test)[:, positive_idx]
test_customers['lapse_probability'] = proba
test_customers['actually_lapsed'] = test_customers['lapsed'].astype(int)
retention = test_customers[test_customers.is_premium.fillna(False)].sort_values('lapse_probability',ascending=False).head(10)
retention[['customer_id','lapse_probability','actually_lapsed']].to_csv(OUT / 'retention_list.csv', index=False)
print(f'\nRetention list rows={len(retention)}; truly lapsed among listed={int(retention.actually_lapsed.sum())}')
print(retention[['customer_id','lapse_probability','actually_lapsed']].to_string(index=False, formatters={'lapse_probability':'{:.4f}'.format}))

# Machine-readable summary for reproducibility.
summary = {
 'snapshot': str(SNAPSHOT), 'cutoff': str(CUTOFF), 'rfm_customers': int(len(rfm)),
 'cohort_customers': int(len(features)), 'lapsed_customers': int(y.sum()),
 'train_rows': int(len(X_train)), 'test_rows': int(len(X_test)), 'chosen_k': chosen_k,
 'premium_segment_id': premium_id, 'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
 'logistic_best_params': log_grid.best_params_, 'logistic_best_cv_f1': float(log_grid.best_score_),
 'rf_best_params': rf_grid.best_params_, 'rf_best_cv_f1': float(rf_grid.best_score_),
 'selected_model': selected_name, 'selected_test_f1': float(selected_f1),
 'oversampled_counts': {str(k):int(v) for k,v in y_bal.value_counts().sort_index().items()},
 'reloaded_predictions_exact_match': exact_match,
 'retention_list_rows': int(len(retention)), 'retention_list_true_lapsed': int(retention.actually_lapsed.sum())
}
(OUT / 'part_b_summary.json').write_text(json.dumps(summary, indent=2, default=str))
print(f'\nWrote outputs to {OUT}')
