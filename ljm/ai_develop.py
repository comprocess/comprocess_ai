import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score


# ===========================================================
# 0) CSV 경로 (요청한 경로 그대로 적용)
# ===========================================================
csv_path = Path.home() / "Downloads" / "AI.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"❌ CSV 파일이 없습니다: {csv_path}")


# ===========================================================
# 1) CSV 로딩
# ===========================================================
raw = pd.read_csv(csv_path, header=None, dtype=str)

months = raw.iloc[0, 2:].astype(str).str.replace("월", "", regex=False)
dates = pd.to_datetime(months, format="%Y%m")

def to_float_series(s):
    s2 = s.astype(str).str.replace(",", "", regex=False).str.strip()
    s2 = s2.replace({"": np.nan, "nan": np.nan})
    return pd.to_numeric(s2, errors="coerce")

usd_vals = to_float_series(raw.iloc[1, 2:])
jpy_vals = to_float_series(raw.iloc[2, 2:])

df = pd.DataFrame({"USD": usd_vals.values, "JPY100": jpy_vals.values}, index=dates)

df = df.sort_index()
df = df.asfreq("MS")
df = df.interpolate(method="linear", limit_direction="both")


# ===========================================================
# 2) 수익률 계산
# ===========================================================
df["USD_ret"] = df["USD"].pct_change()
df["JPY_ret"] = df["JPY100"].pct_change()
df = df.dropna()


# ===========================================================
# 3) Train / Val / Test 기간 설정
# ===========================================================
train_period = ("2000-01-01", "2023-12-01")
val_period   = ("2024-01-01", "2024-12-01")
test_period  = ("2025-01-01", "2025-10-01")

train_df_for_scale = df.loc[train_period[0]:train_period[1]]


# ===========================================================
# 4) 스케일링
# ===========================================================
scaler_X = MinMaxScaler((0, 1))
scaler_y = MinMaxScaler((0, 1))

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
# 6) Split
# ===========================================================
train_end = seq_index.get_loc(train_period[1])
val_end   = seq_index.get_loc(val_period[1])

X_train = X_all[:train_end+1]
y_train = y_all[:train_end+1]

X_val   = X_all[train_end+1:val_end+1]
y_val   = y_all[train_end+1:val_end+1]

X_test  = X_all[val_end+1:]
y_test  = y_all[val_end+1:]


# ===========================================================
# 7) Attention Layer 정의
# ===========================================================
class Attention(layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="normal"
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros"
        )
        super().build(input_shape)

    def call(self, inputs):
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        attention_weights = tf.nn.softmax(score, axis=1)
        context = inputs * attention_weights
        return tf.reduce_sum(context, axis=1)


# ===========================================================
# 8) 모델 구성 (LSTM + Attention)
# ===========================================================
n_features = X_train.shape[2]

input_layer = layers.Input(shape=(lookback, n_features))
x = layers.LSTM(64, return_sequences=True)(input_layer)
x = layers.Dropout(0.2)(x)

att = Attention()(x)

x = layers.Dense(32, activation="relu")(att)
x = layers.Dense(16, activation="relu")(x)

output = layers.Dense(1)(x)

model = Model(inputs=input_layer, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

model.summary()


# ===========================================================
# 9) 모델 저장 경로
# ===========================================================
model_dir = Path.cwd() / "models"
model_dir.mkdir(exist_ok=True)
model_path = model_dir / "lstm_usd_model.keras"

# Loss 증가 시 reduce LR
callbacks = [
    EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
    ModelCheckpoint(str(model_path), monitor="val_loss", save_best_only=True)
]


# ===========================================================
# 10) 학습
# ===========================================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)


# ===========================================================
# 11) 예측 및 역변환
# ===========================================================
y_train_pred = scaler_y.inverse_transform(model.predict(X_train))
y_val_pred   = scaler_y.inverse_transform(model.predict(X_val))
y_test_pred  = scaler_y.inverse_transform(model.predict(X_test))

y_train_true = scaler_y.inverse_transform(y_train)
y_val_true   = scaler_y.inverse_transform(y_val)
y_test_true  = scaler_y.inverse_transform(y_test)


# ===========================================================
# 12) 평가 지표 출력
# ===========================================================
def metrics(y_t, y_p):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_t, y_p)),
        "MAE": mean_absolute_error(y_t, y_p),
        "MAPE": mean_absolute_percentage_error(y_t, y_p),
        "R2": r2_score(y_t, y_p)
    }

def pretty_print(name, m):
    print(f"\n===== {name} Metrics =====")
    for k, v in m.items():
        print(f"{k:<6}: {v:.6f}")

pretty_print("Train", metrics(y_train_true, y_train_pred))
pretty_print("Validation", metrics(y_val_true, y_val_pred))
pretty_print("Test", metrics(y_test_true, y_test_pred))


# ===========================================================
# 13) 스케일러 저장
# ===========================================================
joblib.dump(scaler_X, model_dir / "scaler_X.joblib")
joblib.dump(scaler_y, model_dir / "scaler_y.joblib")

print("Model and scalers saved.")
