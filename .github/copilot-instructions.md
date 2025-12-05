# Copilot Instructions

- Goal: Kaggle-style regression predicting `popularity` from Spotify track metadata; main workflow lives in `notebooks/02_modelos_regresion.ipynb` (most up to date).
- Data: use `data/train.csv` and `data/test.csv` relative to notebooks; keep target column `popularity` and id columns (`id`, `track_id`) untouched when engineering features.
- Seed: set `RANDOM_STATE = 4` and `np.random.seed(RANDOM_STATE)` to match current experiments (older baseline notebook uses 42).
- Feature engineering (current best): add `duration_min` and `energy_danceability`; cap numeric columns (except target) with IQR bounds computed on train and applied to test.
- Encodings to avoid leakage: `add_pop_mean_feature_oof` builds smoothed mean encodings for `artists` and `album_name` (5-fold, smoothing≈20, global fallback); `target_encode_oof` adds OOF target encodings for the same columns (5-fold, smoothing=10).
- Genre stats: map `track_genre` mean popularity with global fallback, and `genre_pop_std`; derived diffs `artist_vs_genre`, `album_vs_artist` capture relative popularity gaps.
- Counts: compute `artist_track_count` and `album_track_count` from train, fillna 0 for test, and include log-transformed versions (`log1p`).
- Feature set: drop `high_card_cols = ["id", "track_id", "artists", "album_name", "track_name"]` and the target; recompute `numeric_features`/`categorical_features` from the remaining `feature_cols` before building pipelines.
- Preprocessing: `ColumnTransformer` with numeric pipeline (median imputer → `StandardScaler`) and categorical pipeline (most_frequent imputer → `OneHotEncoder(handle_unknown="ignore")`). Keep the same preprocessor for train/val/test to avoid drift.
- Modeling pattern: wrap estimator in `Pipeline(preprocess, model)`; holdout split 80/20 with shared seed; primary metric RMSE (use `scoring="neg_root_mean_squared_error"` in `GridSearchCV`).
- Model grids: DecisionTree (depth/min_samples_leaf), RandomForest (trees, depth, min_samples_leaf, max_features="sqrt"), GradientBoosting (n_estimators, learning_rate, max_depth, min_samples_leaf), MLP (hidden_layer_sizes, alpha, learning_rate_init); LinearRegression trains without grid.
- Results handling: accumulate dicts in `resultados`, build `resultados_df`, and pick the model with lowest `rmse_val`; keep `best_estimator` for reuse.
- Final training: clone the best estimator, fit on full train (`X_full`, `y_full`), predict on test, and save submission CSV with columns `id`, `popularity` (e.g., `submission.csv`).
- Extra RF sweep: helper `probar_rf` compares 200/300/400 trees; final tuned model saved to `submission_rf_400_oof.csv` for reference.
- Conventions: section headers and comments are in Spanish; keep them concise; place new feature creation before defining `feature_cols`; avoid leakage by ensuring any group-based stats are OOF and computed on train only.
- Older baseline: `notebooks/preprocesamiento + modelos de regresión.ipynb` mirrors the flow without the OOF encodings; prefer the newer notebook for changes.
- No tests/build scripts: all work runs inside notebooks; prefer incremental notebook edits over new CLI tools; keep outputs reproducible and lightweight.
- If adding models: extend the `modelos` list with `{nombre, estimator, param_grid}` using `model__` prefixes so grids reach the estimator through the pipeline; reuse `evaluar_modelo` to stay consistent.
