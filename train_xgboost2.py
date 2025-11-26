import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from dataset import TimeSeriesNH4Dataset
from utils import plot_nh4_predictions_xgboost

# --- Hyperparameters ---
csv_path = "data/raw_normalized/normalized.csv"
seq_len = 1
num_epochs = 50  # XGBoost can use `num_boost_round`
test_size = 0.2
learning_rate = 0.05
max_depth = 6
n_estimators = 100

# --- Load dataset ---
dataset = TimeSeriesNH4Dataset(csv_path, seq_len=seq_len)
X = dataset.features  # shape: (num_samples, num_features)
y = dataset.targets   # shape: (num_samples, 1) or (num_samples,)

# Flatten sequences if seq_len > 1
if seq_len > 1:
    X = X.reshape(X.shape[0], -1)

# --- Train/validation split ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=42)

# --- Initialize XGBoost regressor ---
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

# --- Train model ---
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True,
)

# --- Evaluation ---
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

train_loss = mean_squared_error(y_train, y_train_pred)
val_loss = mean_squared_error(y_val, y_val_pred)

print(f"Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

# --- Plot predictions ---
# For plotting, we need a DataLoader-like interface for the XGBoost predictions
class DummyLoader:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __iter__(self):
        for xi, yi in zip(self.X, self.y):
            yield xi, yi

val_loader_dummy = DummyLoader(X_val, y_val)
plot_nh4_predictions_xgboost(model, val_loader_dummy)
