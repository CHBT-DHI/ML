import torch
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_nh4_predictions_xgboost(model, val_loader, save_path="nh4_prediction.png"):
    """
    Plot NH4 predicted vs measured for an XGBoost model + iterable loader.
    Works even if loader yields scalars or 1D arrays.
    """

    y_meas_all = []
    y_pred_all = []

    for idx, (x_batch, y_batch) in enumerate(val_loader):
        # Convert to numpy
        x_np = np.array(x_batch)
        y_np = np.array(y_batch)

        # Ensure 2D for XGBoost
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)

        # Make sure y is 1D
        y_np = y_np.reshape(-1)

        y_pred = model.predict(x_np)
        y_pred = np.array(y_pred).reshape(-1)

        y_meas_all.append(y_np)
        y_pred_all.append(y_pred)

        if idx > 1:  # only plot first few samples
            break

    # Convert lists to flat arrays safely
    y_meas_all = np.concatenate([y.reshape(-1) for y in y_meas_all])
    y_pred_all = np.concatenate([p.reshape(-1) for p in y_pred_all])

    residuals = y_pred_all - y_meas_all
    t = np.arange(len(y_meas_all))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6),
        gridspec_kw={"height_ratios": [0.7, 0.3]}, sharex=True
    )

    ax1.plot(t, y_meas_all, label="NH4 measured", color="limegreen")
    ax1.plot(t, y_pred_all, label="NH4 predicted", color="royalblue")
    ax1.set_ylabel("NH4 [mgN/L]")
    ax1.legend()
    ax1.set_title("NH4 Measured vs Predicted with Residuals")

    ax2.plot(t, residuals, label="Residual (pred - meas)", color="black")
    ax2.set_xlabel("Time (index)")
    ax2.set_ylabel("Residual NH4 [mgN/L]")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Figure saved to {save_path}")


# Context manager to replace torch.no_grad() for non-torch models
class dummy_context:
    def __enter__(self): return None
    def __exit__(self, *args): pass

def plot_nh4_predictions(model, val_loader, device="cuda", save_path="nh4_prediction.png"):
    """
    Plot NH4 predicted vs measured from a validation DataLoader, with residuals (matplotlib version).
    """
    model.eval()
    model.to(device)

    y_meas_all = []
    y_pred_all = []

    with torch.no_grad():
        for idx, (x_batch, y_batch) in enumerate(val_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            if idx>1:
                break

            y_pred = model(x_batch)

            y_meas_all.append(y_batch.cpu().numpy())
            y_pred_all.append(y_pred.cpu().numpy())

    y_meas_all = np.concatenate(y_meas_all)
    y_pred_all = np.concatenate(y_pred_all)
    residuals = y_pred_all - y_meas_all
    t = np.arange(len(y_meas_all))  # simple time index

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [0.7, 0.3]}, sharex=True)

    # Top: measured vs predicted
    ax1.plot(t, y_meas_all, label="NH4 measured", color="limegreen")
    ax1.plot(t, y_pred_all, label="NH4 predicted", color="royalblue")
    ax1.set_ylabel("NH4 [mgN/L]")
    ax1.legend()
    ax1.set_title("NH4 Measured vs Predicted with Residuals")

    # Bottom: residuals
    ax2.plot(t, residuals, label="Residual (pred - meas)", color="black")
    ax2.set_xlabel("Time (index)")
    ax2.set_ylabel("Residual NH4 [mgN/L]")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Figure saved to {save_path}")
