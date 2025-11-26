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

months = raw.iloc[0, 2:].astype(str).str.replace("월", "", regex=False)
dates = pd.to_datetime(months, format="%Y%m")  # 예: 200601 → 2006-01-01

# 숫자 변환 함수
def to_float_series(s):
    s2 = s.astype(str).str.replace(",", "", regex=False).str.strip()
    s2 = s2.replace({"": np.nan, "nan": np.nan})
    return pd.to_numeric(s2, errors="coerce")

usd_vals = to_float_series(raw.iloc[1, 2:])
jpy_vals = to_float_series(raw.iloc[2, 2:])

df = pd.DataFrame({"USD": usd_vals.values, "JPY100": jpy_vals.values}, index=dates)

print("원본 df shape:", df.shape)
print(df.head(3))

# ===========================================================
# 2) 인덱스 정렬 + 결측치 처리
# ===========================================================
df = df.sort_index()
df = df.asfreq("MS")
print("asfreq 후 누락값 개수:\n", df.isna().sum())

df = df.interpolate(method="linear", limit_direction="both")
print("보간 후 누락값 개수:\n", df.isna().sum())

print("\n최종 df shape:", df.shape)


# ===========================================================
# 3) train/val/test 기간 지정 (날짜만 지정)
# ===========================================================
train_period = ("2006-01-01", "2023-12-01")
val_period   = ("2024-01-01", "2024-12-01")
test_period  = ("2025-01-01", "2025-10-01")

# train만 따로 임시로 만들어서 scaler fit 용도로 사용
train_df_for_scale = df.loc[train_period[0]:train_period[1]]

# ===========================================================
# 4) 스케일링 (train 구간으로 fit)
# ===========================================================
scaler_X = MinMaxScaler((0,1))
scaler_y = MinMaxScaler((0,1))

scaler_X.fit(train_df_for_scale.values)
scaler_y.fit(train_df_for_scale[["USD"]].values)

scaled = scaler_X.transform(df.values)
scaled_y = scaler_y.transform(df[["USD"]].values)

# ===========================================================
# 5) 시퀀스 전체 생성 (이 시점에서는 split 안 함)
# ===========================================================
lookback = 12
X_all, y_all = [], []

for i in range(lookback, len(scaled)):
    X_all.append(scaled[i-lookback:i])
    y_all.append(scaled_y[i])

X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all, dtype=np.float32)

# X_all, y_all의 인덱스는 df.index[12:]와 동일
seq_index = df.index[lookback:]

print("전체 시퀀스 길이:", X_all.shape, y_all.shape)

# ===========================================================
# 6) 시퀀스 기준으로 train/val/test split
# ===========================================================
train_end = seq_index.get_loc(train_period[1])
val_end   = seq_index.get_loc(val_period[1])

X_train_seq = X_all[:train_end+1]
y_train_seq = y_all[:train_end+1]

X_val_seq   = X_all[train_end+1:val_end+1]
y_val_seq   = y_all[train_end+1:val_end+1]

X_test_seq  = X_all[val_end+1:]
y_test_seq  = y_all[val_end+1:]

print("\n최종 SPLIT 결과:")
print("Train :", X_train_seq.shape, y_train_seq.shape)
print("Val   :", X_val_seq.shape, y_val_seq.shape)
print("Test  :", X_test_seq.shape, y_test_seq.shape)


# ===========================================================
# 7) LSTM 모델 구축 및 학습
# ===========================================================
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

np.random.seed(42)
tf.random.set_seed(42)

model_dir = Path.cwd() / "models"
model_dir.mkdir(exist_ok=True)
model_path = model_dir / "lstm_usd_model.h5"

n_features = X_train_seq.shape[2]

model = Sequential([
    LSTM(64, input_shape=(lookback, n_features), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

es = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
mc = ModelCheckpoint(str(model_path), monitor="val_loss", save_best_only=True)

history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=200,
    batch_size=16,
    callbacks=[es, mc],
    verbose=1
)

# ===========================================================
# 8) 예측 + 역변환
# ===========================================================
y_train_pred_s = model.predict(X_train_seq)
y_val_pred_s   = model.predict(X_val_seq)
y_test_pred_s  = model.predict(X_test_seq)

y_train_true = scaler_y.inverse_transform(y_train_seq)
y_train_pred = scaler_y.inverse_transform(y_train_pred_s)

y_val_true = scaler_y.inverse_transform(y_val_seq)
y_val_pred = scaler_y.inverse_transform(y_val_pred_s)

y_test_true = scaler_y.inverse_transform(y_test_seq)
y_test_pred = scaler_y.inverse_transform(y_test_pred_s)

# ===========================================================
# 9) 성능 평가
# ===========================================================
def compute_metrics(y_t, y_p):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_t, y_p)),
        "MAE": mean_absolute_error(y_t, y_p),
        "MAPE": mean_absolute_percentage_error(y_t, y_p),
        "R2": r2_score(y_t, y_p)
    }

metrics = {
    "Train": compute_metrics(y_train_true, y_train_pred),
    "Val": compute_metrics(y_val_true, y_val_pred),
    "Test": compute_metrics(y_test_true, y_test_pred),
}

metrics_df = pd.DataFrame(metrics).T
print("\n평가 지표:")
print(metrics_df)

# ===========================================================
# 10) 시각화
# ===========================================================
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

print(f"Plot saved to: {plot_file}")

# ===========================================================
# 11) 스케일러 저장
# ===========================================================
import joblib
joblib.dump(scaler_X, model_dir / "scaler_X.joblib")
joblib.dump(scaler_y, model_dir / "scaler_y.joblib")

print(f"Model saved to {model_path}")
