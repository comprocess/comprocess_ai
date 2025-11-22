import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# 1) 파일 로드 (wide -> DataFrame)
csv_path = Path.home() / "Downloads" / "AI.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

raw = pd.read_csv(csv_path, header=None, dtype=str)
months = raw.iloc[0, 2:].astype(str).str.replace("월", "", regex=False)
dates = pd.to_datetime(months, format="%Y%m")  # 월 단위 날짜 생성

# 숫자 문자열에 쉼표 제거, 빈값은 NaN 처리
def to_float_series(s):
    s2 = s.astype(str).str.replace(",", "", regex=False).str.strip()
    s2 = s2.replace({"": np.nan, "nan": np.nan})
    return pd.to_numeric(s2, errors="coerce")

usd_vals = to_float_series(raw.iloc[1, 2:])
jpy_vals = to_float_series(raw.iloc[2, 2:])

df = pd.DataFrame({"USD": usd_vals.values, "JPY100": jpy_vals.values}, index=dates)
df.index.name = None

# 간단 확인
print("원본 df shape:", df.shape)
print(df.head(3))

# 2) 전처리: 정렬, 월단위 인덱스 강제, 보간
df = df.sort_index()
df = df.asfreq("MS")  # 월 시작일 기준 인덱스 강제
print("asfreq 후 누락값 개수:\n", df.isna().sum())

df = df.interpolate(method="linear", limit_direction="both")
print("보간 후 누락값 개수:\n", df.isna().sum())

# 3) 분할: train / val / test
train = df.loc["2006-01-01":"2023-12-01"]
val   = df.loc["2024-01-01":"2024-12-01"]
test  = df.loc["2025-01-01":"2025-10-01"]

print("train/val/test shapes:", train.shape, val.shape, test.shape)

# 4) 스케일링: train만으로 fit
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

scaler_X.fit(train.values)            # 입력(USD, JPY100) 열별로 fit
scaler_y.fit(train[["USD"]].values)   # 타깃(USD)만 fit

X_train_scaled = scaler_X.transform(train.values)
y_train_scaled = scaler_y.transform(train[["USD"]].values)

X_val_scaled = scaler_X.transform(val.values)
y_val_scaled = scaler_y.transform(val[["USD"]].values)

X_test_scaled = scaler_X.transform(test.values)
y_test_scaled = scaler_y.transform(test[["USD"]].values)

# 5) 시퀀스 생성 (LSTM 입력용: samples, lookback, features)
lookback = 12

def create_sequences(X, y, lb):
    xs, ys = [], []
    for i in range(lb, len(X)):
        xs.append(X[i - lb:i])
        ys.append(y[i])  # 1-step ahead
    return np.array(xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, lookback)
X_val_seq,   y_val_seq   = create_sequences(X_val_scaled,   y_val_scaled,   lookback)
X_test_seq,  y_test_seq  = create_sequences(X_test_scaled,  y_test_scaled,  lookback)

# -- 입력/타깃 배열 강제 정리(모든 시퀀스가 float32, y는 (n,1) 모양) --
# X_*_seq가 object dtype이면 np.stack 사용
if X_train_seq.dtype == object:
    X_train_seq = np.stack(X_train_seq)
if X_val_seq.dtype == object:
    X_val_seq = np.stack(X_val_seq)
if X_test_seq.dtype == object:
    X_test_seq = np.stack(X_test_seq)

# 강제 float32
X_train_seq = np.asarray(X_train_seq, dtype=np.float32)
X_val_seq   = np.asarray(X_val_seq,   dtype=np.float32)
X_test_seq  = np.asarray(X_test_seq,  dtype=np.float32)

y_train_seq = np.asarray(y_train_seq, dtype=np.float32)
y_val_seq   = np.asarray(y_val_seq,   dtype=np.float32)
y_test_seq  = np.asarray(y_test_seq,  dtype=np.float32)

# y가 (n,)이면 (n,1)로 reshape
if y_train_seq.ndim == 1:
    y_train_seq = y_train_seq.reshape(-1, 1)
if y_val_seq.ndim == 1:
    y_val_seq = y_val_seq.reshape(-1, 1)
if y_test_seq.ndim == 1:
    y_test_seq = y_test_seq.reshape(-1, 1)

print("DEBUG shapes/dtypes:",
      X_train_seq.shape, X_train_seq.dtype,
      y_train_seq.shape, y_train_seq.dtype,
      X_val_seq.shape, X_val_seq.dtype,
      X_test_seq.shape, X_test_seq.dtype)

# 6) 상태 출력
print("df shape:", df.shape)
print("X_train_seq shape:", X_train_seq.shape)
print("y_train_seq shape:", y_train_seq.shape)
print("X_val_seq shape:", X_val_seq.shape)
print("X_test_seq shape:", X_test_seq.shape)

# 필요한 객체들을 이후 모델 학습에 사용할 수 있도록 노출
# df, train, val, test, scaler_X, scaler_y, X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq

# ------------------ LSTM 모델: 학습, 평가, 시각화 추가 코드 ------------------
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error

# 재현성(완전한 재현은 환경에 따라 다름)
import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)

model_dir = Path.cwd() / "models"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "lstm_usd_model.h5"

# 모델 설계 (간단하고 안정적인 기본 구조)
n_features = X_train_seq.shape[2]  # 2
model = Sequential([
    LSTM(64, input_shape=(lookback, n_features), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1, activation="linear")
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# 콜백
es = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
mc = ModelCheckpoint(str(model_path), monitor="val_loss", save_best_only=True, verbose=0)

# 학습
history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=200,
    batch_size=16,
    callbacks=[es, mc],
    verbose=1
)

# 예측 (스케일된 값)
y_train_pred_s = model.predict(X_train_seq)
y_val_pred_s   = model.predict(X_val_seq)
y_test_pred_s  = model.predict(X_test_seq)

# 역변환 (원 단위 USD)
y_train_true = scaler_y.inverse_transform(y_train_seq.reshape(-1, 1))
y_train_pred = scaler_y.inverse_transform(y_train_pred_s)

y_val_true = scaler_y.inverse_transform(y_val_seq.reshape(-1, 1))
y_val_pred = scaler_y.inverse_transform(y_val_pred_s)

y_test_true = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1))
y_test_pred = scaler_y.inverse_transform(y_test_pred_s)

# 평가 지표 계산
def compute_metrics(y_t, y_p):
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae = mean_absolute_error(y_t, y_p)
    mape = mean_absolute_percentage_error(y_t, y_p)
    r2 = r2_score(y_t, y_p)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}

metrics = {
    "Train": compute_metrics(y_train_true, y_train_pred),
    "Val": compute_metrics(y_val_true, y_val_pred),
    "Test": compute_metrics(y_test_true, y_test_pred)
}

metrics_df = pd.DataFrame(metrics).T
print("\nEvaluation metrics:")
print(metrics_df)

# 시각화: 학습곡선, 검증/테스트 예측 비교
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# 학습곡선
axes[0,0].plot(history.history["loss"], label="train_loss")
axes[0,0].plot(history.history["val_loss"], label="val_loss")
axes[0,0].set_title("Training / Validation Loss")
axes[0,0].legend()

# validation: 실제 vs 예측 (최근 일부만 보기)
axes[0,1].plot(y_val_true, label="val_true")
axes[0,1].plot(y_val_pred, label="val_pred")
axes[0,1].set_title("Validation: Actual vs Predicted (USD)")
axes[0,1].legend()

# test: 실제 vs 예측
axes[1,0].plot(y_test_true, label="test_true")
axes[1,0].plot(y_test_pred, label="test_pred")
axes[1,0].set_title("Test: Actual vs Predicted (USD)")
axes[1,0].legend()

# 잔차 히스토그램 (test)
resid = (y_test_true - y_test_pred).ravel()
axes[1,1].hist(resid, bins=15)
axes[1,1].set_title("Test Residuals Histogram")

plt.tight_layout()
plot_dir = Path.cwd() / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)
plot_file = plot_dir / "lstm_evaluation.png"
plt.savefig(plot_file)
print(f"Saved plot to {plot_file}")
plt.close(fig)

# 간단한 시각적 테이블 출력(터미널용)
print("\nMetrics table:")
print(metrics_df.round(4))

# 끝: 모델과 스케일러는 파일로 저장(선택)
import joblib
joblib.dump(scaler_X, model_dir / "scaler_X.joblib")
joblib.dump(scaler_y, model_dir / "scaler_y.joblib")
print(f"Saved model to {model_path}, scalers to {model_dir}")