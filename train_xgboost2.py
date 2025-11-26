import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from dataset import TimeSeriesNH4Dataset
from utils import plot_nh4_predictions_xgboost

# --- Hyperparameters ---
train_csv_path = "data/raw_normalized/train.csv"
test_csv_path  = "data/raw_normalized/test.csv"

seq_len = 1
test_size = 0.1       # for val split inside the training portion
learning_rate = 0.05
max_depth = 6
n_estimators = 100

# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------
train_dataset = TimeSeriesNH4Dataset(train_csv_path, seq_len=seq_len)
test_dataset  = TimeSeriesNH4Dataset(test_csv_path,  seq_len=seq_len)

X_train_full = train_dataset.features
y_train_full = train_dataset.targets

X_test = test_dataset.features
y_test = test_dataset.targets

# Flatten sequences if seq_len > 1
if seq_len > 1:
    X_train_full = X_train_full.reshape(X_train_full.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

# ---------------------------------------------------------
# Train/validation split (inside the training set)
# ---------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=test_size, shuffle=True, random_state=42
)

# ---------------------------------------------------------
# Initialize model
# ---------------------------------------------------------
xgb_params = dict(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
)
model = xgb.XGBRegressor(**xgb_params)

# ---------------------------------------------------------
# Train
# ---------------------------------------------------------
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True,
)

# ---------------------------------------------------------
# Evaluation on train + val
# ---------------------------------------------------------
y_train_pred = model.predict(X_train)
y_val_pred   = model.predict(X_val)

train_loss = mean_squared_error(y_train, y_train_pred)
val_loss   = mean_squared_error(y_val,   y_val_pred)

print(f"Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

# ---------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------
y_test_pred = model.predict(X_test)
test_loss = mean_squared_error(y_test, y_test_pred)

print(f"TEST MSE: {test_loss:.6f}")

# ---------------------------------------------------------
# Plot predictions on test set
# ---------------------------------------------------------
class DummyLoader:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __iter__(self):
        for xi, yi in zip(self.X, self.y):
            yield xi, yi

test_loader_dummy = DummyLoader(X_test, y_test)
plot_nh4_predictions_xgboost(model, test_loader_dummy)
