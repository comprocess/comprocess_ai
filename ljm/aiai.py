import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# ===========================================================
# 1) CSV 로드 및 원본 정리
# ===========================================================
csv_path = Path.home() / "Downloads" / "AI.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

raw = pd.read_csv(csv_path, header=None, dtype=str)

# 월 문자열 -> datetime
months = raw.iloc[0, 2:].astype(str).str.replace("월", "", regex=False)
dates = pd.to_datetime(months, format="%Y%m")

# 숫자 변환 함수
def to_float_series(s):
    s2 = s.astype(str).str.replace(",", "", regex=False).str.strip()
    s2 = s2.replace({"": np.nan, "nan": np.nan})
    return pd.to_numeric(s2, errors="coerce")

# 원/달러, 원/100엔
usd_vals = to_float_series(raw.iloc[1, 2:])
jpy_vals = to_float_series(raw.iloc[2, 2:])

df = pd.DataFrame({"USD": usd_vals.values, "JPY100": jpy_vals.values}, index=dates)

# 결측치 처리
df = df.sort_index()
df = df.asfreq("MS")
df = df.interpolate(method="linear", limit_direction="both")

# ===========================================================
# 2) 수익률 변환
# ===========================================================
df["USD_ret"] = df["USD"].pct_change()
df["JPY_ret"] = df["JPY100"].pct_change()
df = df.dropna()

# ===========================================================
# 3) 데이터 Split 기간 정의 (요청한 기간 반영)
# ===========================================================
train_period = ("2000-01-01", "2023-12-01")
val_period   = ("2024-01-01", "2024-12-01")
test_period  = ("2025-01-01", "2025-10-01")

# 스케일러는 train 영역만 사용해 fit
train_df_for_scale = df.loc[train_period[0]:train_period[1]]

# ===========================================================
# 4) 스케일링
# ===========================================================
scaler_X = MinMaxScaler((0,1))
scaler_y = MinMaxScaler((0,1))

scaler_X.fit(train_df_for_scale[["USD_ret", "JPY_ret"]])
scaler_y.fit(train_df_for_scale[["USD_ret"]])

scaled_X = scaler_X.transform(df[["USD_ret", "JPY_ret"]])
scaled_y = scaler_y.transform(df[["USD_ret"]])

# ===========================================================
# 5) 시퀀스 생성
# ===========================================================
lookback = 24
X_all, y_all = [], []

for i in range(lookback, len(df)):
    X_all.append(scaled_X[i-lookback:i])
    y_all.append(scaled_y[i])

X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all, dtype=np.float32)

seq_index = df.index[lookback:]

# ===========================================================
# 6) Train / Val / Test Split
# ===========================================================
train_end = seq_index.get_loc(train_period[1])
val_end   = seq_index.get_loc(val_period[1])

X_train = X_all[:train_end+1]
y_train = y_all[:train_end+1]

X_val   = X_all[train_end+1:val_end+1]
y_val   = y_all[train_end+1:val_end+1]

X_test  = X_all[val_end+1:]
y_test  = y_all[val_end+1:]

print("Train :", X_train.shape)
print("Val   :", X_val.shape)
print("Test  :", X_test.shape)

# ===========================================================
# 7) 모델 구성
# ===========================================================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

n_features = X_train.shape[2]

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=2000,
    decay_rate=0.9
)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(lookback, n_features)),
    Dropout(0.2),

    Bidirectional(LSTM(32, return_sequences=False)),
    Dropout(0.2),

    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss="mse", metrics=["mae"])

model_dir = Path.cwd() / "models"
model_dir.mkdir(exist_ok=True)
model_path = model_dir / "lstm_usd_model.h5"

es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
mc = ModelCheckpoint(str(model_path), monitor="val_loss", save_best_only=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=16,
    callbacks=[es, mc],
    verbose=1
)

# 학습이 끝난 후 모델 저장 (ModelCheckpoint는 val_loss 기준 최적 모델 저장, 여기는 현재 상태 저장)
model.save(model_path)

# ===========================================================
# 8) 예측 및 역변환
# ===========================================================
y_train_pred = scaler_y.inverse_transform(model.predict(X_train))
y_val_pred   = scaler_y.inverse_transform(model.predict(X_val))
y_test_pred  = scaler_y.inverse_transform(model.predict(X_test))

y_train_true = scaler_y.inverse_transform(y_train)
y_val_true   = scaler_y.inverse_transform(y_val)
y_test_true  = scaler_y.inverse_transform(y_test)

# ===========================================================
# 9) 평가 지표 (예쁘게 정렬된 표 형태)
# ===========================================================
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

def metrics(y_t, y_p):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_t, y_p)),
        "MAE": mean_absolute_error(y_t, y_p),
        "MAPE": mean_absolute_percentage_error(y_t, y_p),
        "R2": r2_score(y_t, y_p)
    }

def pretty_print_metrics(name, m):
    print(f"\n===== {name} Metrics =====")
    print(f"{'Metric':<10} | {'Value':>12}")
    print("-"*26)
    for k, v in m.items():
        print(f"{k:<10} | {v:>12.6f}")

pretty_print_metrics("Train", metrics(y_train_true, y_train_pred))
pretty_print_metrics("Validation", metrics(y_val_true, y_val_pred))
pretty_print_metrics("Test", metrics(y_test_true, y_test_pred))

# ===========================================================
# 10) plot 저장
# ===========================================================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].plot(history.history["loss"], label="train_loss")
axes[0,0].plot(history.history["val_loss"], label="val_loss")
axes[0,0].set_title("Training / Validation Loss")
axes[0,0].legend()

axes[0,1].plot(y_val_true, label="val_true")
axes[0,1].plot(y_val_pred, label="val_pred")
axes[0,1].set_title("Validation Actual vs Pred")
axes[0,1].legend()

axes[1,0].plot(y_test_true, label="test_true")
axes[1,0].plot(y_test_pred, label="test_pred")
axes[1,0].set_title("Test Actual vs Pred")
axes[1,0].legend()

resid = (y_test_true - y_test_pred).ravel()
axes[1,1].hist(resid, bins=15)
axes[1,1].set_title("Test Residuals")

plt.tight_layout()
plot_dir = Path.cwd() / "plots"
plot_dir.mkdir(exist_ok=True)
plot_file = plot_dir / "lstm_evaluation.png"
plt.savefig(plot_file)
plt.close(fig)

print(f"\nPlot saved to: {plot_file}")

# ===========================================================
# 11) 스케일러 저장
# ===========================================================
import joblib
joblib.dump(scaler_X, model_dir / "scaler_X.joblib")
joblib.dump(scaler_y, model_dir / "scaler_y.joblib")

print("Model and scalers saved.")
