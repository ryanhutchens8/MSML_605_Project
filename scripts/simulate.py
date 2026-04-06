import pandas as pd
import os
from collections import deque
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

INPUT_FILE = "data/load_weather_full.csv"
LOG_FILE = "results/simulation_log.csv"
PLOT_FILE = "results/drift_simulation.png"
MODEL_DIR = "models"

TRAIN_START = "2018-01-01"
TRAIN_END = "2018-12-31"
DEPLOY_START = "2019-01-01"
DRIFT_THRESHOLD = 3000
DRIFT_WINDOW = 3       # rolling avg over this many days before triggering retrain
RETRAIN_YEARS = 2
COMFORT_TEMP = 65
SEQ_LEN = 14       # days of history fed to LSTM
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001
PATIENCE = 8

FEATURES = [
    "hdd", "cdd", "temp_f_max", "temp_f_min",
    "feels_like_f_mean", "humidity_pct_mean",
    "dayofweek", "month", "dayofyear", "is_weekend",
    "lag_1day"
]
TARGET = "mw_mean"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def aggregate_daily(df):
    df = df.copy()
    df["date"] = df["datetime"].dt.date

    daily = df.groupby("date").agg(
        mw_mean=("mw", "mean"),
        temp_f_mean=("temp_f", "mean"),
        temp_f_max=("temp_f", "max"),
        temp_f_min=("temp_f", "min"),
        feels_like_f_mean=("feels_like_f", "mean"),
        humidity_pct_mean=("humidity_pct", "mean"),
    ).reset_index()

    daily["date"] = pd.to_datetime(daily["date"])
    daily["dayofweek"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["dayofyear"] = daily["date"].dt.dayofyear
    daily["is_weekend"] = (daily["dayofweek"] >= 5).astype(int)
    daily["hdd"] = (COMFORT_TEMP - daily["temp_f_mean"]).clip(lower=0)
    daily["cdd"] = (daily["temp_f_mean"] - COMFORT_TEMP).clip(lower=0)

    daily = daily.sort_values("date").reset_index(drop=True)
    daily["lag_1day"] = daily["mw_mean"].shift(1)
    daily.dropna(inplace=True)

    return daily


class LoadDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.y) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.seq_len]
        y_val = self.y[idx + self.seq_len]
        return x_seq, y_val


class LoadLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)


def train_model(daily, start, end, model_name="lstm"):
    subset = daily[(daily["date"] >= start) & (daily["date"] <= end)].copy()

    if len(subset) < SEQ_LEN + 10:
        print(f"  not enough data ({len(subset)} days), skipping.")
        return None, None

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X = scaler_X.fit_transform(subset[FEATURES].values)
    y = scaler_y.fit_transform(subset[[TARGET]].values).flatten()

    # hold out last 20% for early stopping
    split = int(len(X) * 0.8)
    use_early_stop = (len(X) - split) > SEQ_LEN

    train_loader = DataLoader(LoadDataset(X[:split], y[:split], SEQ_LEN),
                              batch_size=BATCH_SIZE, shuffle=True)
    if use_early_stop:
        val_loader = DataLoader(LoadDataset(X[split:], y[split:], SEQ_LEN),
                                batch_size=BATCH_SIZE)

    model = LoadLSTM(input_size=len(FEATURES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights = None
    wait = 0

    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if not use_early_stop:
            continue

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += loss_fn(model(X_batch), y_batch).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  early stop at epoch {epoch+1}")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{model_name}.pt"))
    joblib.dump(scaler_X, os.path.join(MODEL_DIR, f"{model_name}_scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(MODEL_DIR, f"{model_name}_scaler_y.pkl"))

    print(f"  trained on {start} to {end} ({len(subset)} days)")
    return model, (scaler_X, scaler_y)


def predict_day(model, scalers, history_df):
    scaler_X, scaler_y = scalers
    if len(history_df) < SEQ_LEN:
        return None

    X_raw = history_df[FEATURES].values[-SEQ_LEN:]
    X_scaled = scaler_X.transform(X_raw)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_tensor).item()

    pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]
    return pred


def main():
    os.makedirs("results", exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    daily = aggregate_daily(df)

    print(f"{len(daily)} days loaded ({daily['date'].min().date()} to {daily['date'].max().date()})")

    print(f"training on {TRAIN_START} to {TRAIN_END}...")
    model, scalers = train_model(daily, TRAIN_START, TRAIN_END, model_name="lstm_2018")

    if model is None:
        print("training failed")
        return

    deploy_data = daily[daily["date"] >= DEPLOY_START].reset_index(drop=True)

    log = []
    retrain_events = []
    retrain_count = 0
    recent_errors = deque(maxlen=DRIFT_WINDOW)

    print(f"running simulation from {DEPLOY_START}, drift threshold = {DRIFT_THRESHOLD} MW ({DRIFT_WINDOW}-day avg)")

    for _, row in deploy_data.iterrows():
        date = row["date"]
        history = daily[daily["date"] < date]
        pred = predict_day(model, scalers, history)

        if pred is None:
            continue

        actual = row[TARGET]
        error = abs(actual - pred)
        recent_errors.append(error)

        rolling_avg = sum(recent_errors) / len(recent_errors)
        drift_detected = len(recent_errors) == DRIFT_WINDOW and rolling_avg > DRIFT_THRESHOLD
        status = "OK"

        date_str = date.strftime("%Y-%m-%d")

        if drift_detected:
            retrain_start = date - pd.DateOffset(years=RETRAIN_YEARS)
            print(f"  [{date_str}] DRIFT: {DRIFT_WINDOW}-day avg {rolling_avg:.0f} MW, retraining...")

            new_model, new_scalers = train_model(
                daily,
                str(retrain_start.date()),
                str(date.date()),
                model_name=f"lstm_retrained_{date_str}"
            )

            if new_model is not None:
                model = new_model
                scalers = new_scalers
                retrain_events.append(date)
                retrain_count += 1
                status = "RETRAINED"

        else:
            print(f"  [{date_str}] ok, error {error:.0f} MW")

        log.append({
            "date": date_str,
            "actual": round(actual, 1),
            "predicted": round(pred, 1),
            "error": round(error, 1),
            "drift_detected": drift_detected,
            "status": status,
        })

    log_df = pd.DataFrame(log)
    log_df.to_csv(LOG_FILE, index=False)
    print(f"\n{retrain_count} retrains. log saved to {LOG_FILE}")

    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))
    dates = pd.to_datetime(log_df["date"])
    rolling_error = log_df["error"].rolling(14, center=True).mean()

    ax.plot(dates, log_df["error"], color="steelblue", linewidth=0.4, alpha=0.3, label="Daily error")
    ax.plot(dates, rolling_error, color="steelblue", linewidth=1.5, label="14-day rolling avg")
    ax.axhline(DRIFT_THRESHOLD, color="red", linewidth=1, linestyle="--",
               label=f"threshold ({DRIFT_THRESHOLD} MW)")

    for event in retrain_events:
        ax.axvline(event, color="green", alpha=0.5, linewidth=1, linestyle="--")

    # mark a couple known events for context
    known_events = {
        "2020-03-01": "COVID lockdowns",
        "2022-11-01": "ChatGPT launch",
    }
    for ds, label in known_events.items():
        dt = pd.to_datetime(ds)
        if dt >= pd.to_datetime(DEPLOY_START):
            ax.axvline(dt, color="purple", alpha=0.5, linewidth=1.5)
            ax.text(dt, ax.get_ylim()[1] * 0.92, label,
                    rotation=90, fontsize=8, color="purple", va="top")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="steelblue", alpha=0.4, label="Daily error"),
        Line2D([0], [0], color="steelblue", linewidth=1.5, label="14-day rolling avg"),
        Line2D([0], [0], color="red", linestyle="--", label=f"threshold ({DRIFT_THRESHOLD} MW)"),
        Line2D([0], [0], color="green", linestyle="--", label="retrain triggered"),
        Line2D([0], [0], color="purple", alpha=0.6, label="known event"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    ax.set_title("Drift-Aware Load Forecasting (LSTM) - Daily Error with Retraining Events")
    ax.set_xlabel("Date")
    ax.set_ylabel("Absolute Error (MW)")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    print(f"plot saved to {PLOT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()