# ============================================================
# Wind‑Power Forecasting   |   unified split & preprocessing
# ============================================================

import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pysr import PySRRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

sns.set_theme(style="whitegrid", palette="colorblind", font_scale=0.9)

# ------------------------------------------------------------------
# USER PARAMS
# ------------------------------------------------------------------
MONTHS_TO_USE = 1         # 1, 3, 6, 12, 24 …
LOOK_BACK     = 52        # LSTM window (h)
TRAIN_FRAC    = 0.8       # 80 % chronological split

CSV_PATH   = "Wind_data/Location1.csv"
TARGET_COL = "Power"

RAW_FEATURES = [
    "temperature_2m", "relativehumidity_2m", "dewpoint_2m",
    "windspeed_10m",  "windspeed_100m",      "windgusts_10m",
    "winddirection_10m", "winddirection_100m"
]

# ------------------------------------------------------------------
# 1. LOAD LATEST N MONTHS
# ------------------------------------------------------------------
df = (pd.read_csv(CSV_PATH, parse_dates=["Time"])
        .set_index("Time")
        .tail(24*30*MONTHS_TO_USE)        # crude 30‑day month
        .dropna())                        # ① remove NaNs

# ------------------------------------------------------------------
# 2. FEATURE ENGINEERING & NORMALISATION
# ------------------------------------------------------------------
# ② encode circular directions
for col in ("winddirection_10m", "winddirection_100m"):
    radians = np.deg2rad(df[col])
    df[f"sin_{col}"], df[f"cos_{col}"] = np.sin(radians), np.cos(radians)
df = df.drop(columns=["winddirection_10m", "winddirection_100m"])

# continuous columns to scale
cont_cols = [c for c in df.columns if c != TARGET_COL]

scaler = MinMaxScaler()                   # ③ min–max → [0,1]
df[cont_cols] = scaler.fit_transform(df[cont_cols])

# ④ Z‑score outlier removal (|z|>3)
z = (df[cont_cols] - df[cont_cols].mean()) / df[cont_cols].std(ddof=0)
df = df[(np.abs(z) <= 3).all(axis=1)].copy()

# ------------------------------------------------------------------
# 3. CHRONOLOGICAL TRAIN / TEST SPLIT
# ------------------------------------------------------------------
split = int(len(df) * TRAIN_FRAC)
train_df, test_df = df.iloc[:split], df.iloc[split:]

X_train = train_df.drop(columns=[TARGET_COL]).values
y_train = train_df[TARGET_COL].values
X_test  = test_df.drop(columns=[TARGET_COL]).values
y_test  = test_df[TARGET_COL].values

# ------------------------------------------------------------------
# 4. SYMBOLIC REGRESSION (PySR)
# ------------------------------------------------------------------
sr = PySRRegressor(
    niterations=100, populations=50,
    binary_operators=["+","-","*","/"],
    unary_operators=["sin","cos","cube"],
    extra_sympy_mappings={"cube": lambda x: x**3},
    loss="loss(x, y) = (x - y)^2", verbosity=0)
sr.fit(X_train, y_train)
test_df["pred_SR"] = sr.predict(X_test)

# ------------------------------------------------------------------
# 5. LSTM (univariate, 1‑step)
# ------------------------------------------------------------------
def make_seq(vals, n=LOOK_BACK):
    X, y = [], []
    for i in range(n, len(vals)):
        X.append(vals[i-n:i, 0])
        y.append(vals[i, 0])
    return np.array(X), np.array(y)

# use already‑scaled power column
power_scaled = scaler.fit_transform(df[[TARGET_COL]].values)
train_seq = power_scaled[:split]
test_seq  = power_scaled[split-LOOK_BACK:]

X_tr, y_tr = make_seq(train_seq)
X_te, y_te = make_seq(test_seq)
X_tr = X_tr.reshape(-1, LOOK_BACK, 1)
X_te = X_te.reshape(-1, LOOK_BACK, 1)

lstm = Sequential([
    LSTM(128, return_sequences=True, input_shape=(LOOK_BACK,1)),
    LSTM(64), Dense(25), Dense(1)])
lstm.compile(optimizer="adam", loss="mse")
lstm.fit(X_tr, y_tr, epochs=5, batch_size=32, verbose=0)

pred_scaled = lstm.predict(X_te, verbose=0)
test_df = test_df.iloc[LOOK_BACK:].copy()           # align
test_df["pred_LSTM"] = scaler.inverse_transform(pred_scaled)

# ------------------------------------------------------------------
# 6. METRICS
# ------------------------------------------------------------------
rmse = lambda a,b: mean_squared_error(a,b,squared=False)
mae  = lambda a,b: mean_absolute_error(a,b)

metrics = pd.DataFrame({
    "Model":["Symbolic","LSTM"],
    "RMSE":[rmse(y_test, test_df["pred_SR"]),
            rmse(y_test[LOOK_BACK:], test_df["pred_LSTM"])],
    "MAE" :[mae (y_test, test_df["pred_SR"]),
            mae (y_test[LOOK_BACK:], test_df["pred_LSTM"])]})
print(f"\n=== Using last {MONTHS_TO_USE} month(s) ===")
print(metrics.to_string(index=False))

# ------------------------------------------------------------------
# 7. PLOT
# ------------------------------------------------------------------
plt.figure(figsize=(10,4.5))
sns.lineplot(x=test_df.index, y=test_df[TARGET_COL],
             label="Actual",  color="tab:blue")
sns.lineplot(x=test_df.index, y=test_df["pred_SR"],
             label="Symbolic", color="tab:green")
sns.lineplot(x=test_df.index, y=test_df["pred_LSTM"],
             label="LSTM",    color="tab:orange")
plt.title(f"Predictions vs. actual power — last {MONTHS_TO_USE} month(s)")
plt.xlabel("Date"); plt.ylabel("Power")
plt.xticks(rotation=45); plt.legend(); plt.tight_layout()
plt.savefig(f"predictions_{MONTHS_TO_USE}mo.png", dpi=300)
plt.show()