import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import joblib

# ==============================
# LOAD DATA
# ==============================
train_df = pd.read_csv("data/train.csv")
valid_df = pd.read_csv("data/valid.csv")

# ==============================
# FEATURES
# ==============================
features = [
    "is_available",
    "has_skill_match",
    "provider_rating",
    "distance_km",
    "acceptance_rate_30d",
    "avg_response_time_min",
    "completed_jobs_90d"
]

target = "selected_label"

X_train = train_df[features]
y_train = train_df[target]

X_valid = valid_df[features]
y_valid = valid_df[target]

# ==============================
# CREATE DATASETS
# ==============================
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid)

# ==============================
# PARAMETERS
# ==============================
params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42,
    "verbose": -1
}

# ==============================
# TRAIN MODEL
# ==============================
print("\nTraining model...\n")

model = lgb.train(
    params,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=500,
    callbacks=[lgb.early_stopping(50)]
)

# ==============================
# EVALUATION
# ==============================
y_pred = model.predict(X_valid)
auc = roc_auc_score(y_valid, y_pred)

print(f"\nValidation AUC: {auc:.4f}")

# ==============================
# SAVE MODEL
# ==============================
joblib.dump(model, "model.pkl")

print("\nModel saved as model.pkl")