import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# ---------------------------------------------------------------------
# 0) Reproducibility & basic config
# ---------------------------------------------------------------------
np.random.seed(42)
tf.random.set_seed(42)

LOOKBACK = 24          # 더 긴 과거(24개월)를 보도록 변경
BATCH_SIZE = 16
EPOCHS = 300

DATA_PATH = Path.home() / "Downloads" / "AI.csv"
MODEL_DIR = Path.cwd() / "models"
PLOT_DIR = Path.cwd() / "plots"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "lstm_usd_model_improved.keras"
SCALER_X_PATH = MODEL_DIR / "scaler_X.joblib"
SCALER_Y_PATH = MODEL_DIR / "scaler_y.joblib"
PLOT_PATH = PLOT_DIR / "lstm_evaluation_improved.png"

# 기간 설정
TRAIN_PERIOD = ("2006-01-01", "2023-12-01")
VAL_PERIOD   = ("2024-01-01", "2024-12-01")
TEST_PERIOD  = ("2025-01-01", "2025-10-01")


# ---------------------------------------------------------------------
# 1) Data loading + preprocessing
# ---------------------------------------------------------------------
def to_float_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.replace(",", "", regex=False).str.strip()
    s2 = s2.replace({"": np.nan, "nan": np.nan})
    return pd.to_numeric(s2, errors="coerce")


def load_and_preprocess(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {path}")

    raw = pd.read_csv(path, header=None, dtype=str)

    # 첫 행: yyyyMM 또는 yyyyMM월 형식
    months = raw.iloc[0, 2:].astype(str).str.replace("월", "", regex=False)
    dates = pd.to_datetime(months, format="%Y%m")

    usd_vals = to_float_series(raw.iloc[1, 2:])
    jpy_vals = to_float_series(raw.iloc[2, 2:])

    df = pd.DataFrame({"USD": usd_vals.values, "JPY100": jpy_vals.values}, index=dates)
    df.index.name = None

    # 시계열 정렬 및 월별 주기 보정
    df = df.sort_index()
    df = df.asfreq("MS")
    df = df.interpolate(method="linear", limit_direction="both")

    # 이상값 완화(winsorize)
    df["USD_w"] = df["USD"].clip(df["USD"].quantile(0.01), df["USD"].quantile(0.99))
    df["JPY100_w"] = df["JPY100"].clip(df["JPY100"].quantile(0.01), df["JPY100"].quantile(0.99))

    # 기본 사용 컬럼
    df["USD_proc"] = df["USD_w"]
    df["JPY100_proc"] = df["JPY100_w"]

    # 1차 차분 / 퍼센트 변화 / 이동통계량
    df["USD_diff1"] = df["USD_proc"].diff(1).fillna(0.0)
    df["USD_pct_change"] = (
        df["USD_proc"].pct_change().replace([np.inf, -np.inf], 0).fillna(0.0)
    )
    df["USD_roll3_mean"] = df["USD_proc"].rolling(3, min_periods=1).mean()
    df["USD_roll6_std"] = df["USD_proc"].rolling(6, min_periods=1).std().fillna(0.0)

    # 월(계절성) 더미
    months = df.index.month
    month_dummies = pd.get_dummies(months, prefix="m", drop_first=True)
    month_dummies.index = df.index
    df = pd.concat([df, month_dummies], axis=1)

    # 결측/무한치 정리
    df = df.replace([np.inf, -np.inf], np.nan).bfill().ffill()

    return df


# ---------------------------------------------------------------------
# 2) Feature selection + scaling + sequence 만들기
# ---------------------------------------------------------------------
def create_scaled_sequences(df: pd.DataFrame, lookback: int):
    # feature 컬럼 정의 (타깃은 USD_proc)
    month_cols = [c for c in df.columns if c.startswith("m_")]
    feature_cols = [
        "USD_proc",
        "JPY100_proc",
        "USD_diff1",
        "USD_pct_change",
        "USD_roll3_mean",
        "USD_roll6_std",
    ] + month_cols

    # train 구간만 사용해서 scaler fit
    train_df = df.loc[TRAIN_PERIOD[0] : TRAIN_PERIOD[1], feature_cols]

    scaler_X = MinMaxScaler((0, 1))
    scaler_y = MinMaxScaler((0, 1))

    scaler_X.fit(train_df.values)
    scaler_y.fit(train_df[["USD_proc"]].values)

    # 전체 데이터에 스케일 적용
    X_scaled = scaler_X.transform(df[feature_cols].values)
    y_scaled = scaler_y.transform(df[["USD_proc"]].values)

    # 시퀀스 생성
    X_all, y_all = [], []
    for i in range(lookback, len(X_scaled)):
        X_all.append(X_scaled[i - lookback : i])
        y_all.append(y_scaled[i])

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.float32).reshape(-1, 1)

    seq_index = df.index[lookback:]
    return X_all, y_all, seq_index, scaler_X, scaler_y, feature_cols


# ---------------------------------------------------------------------
# 3) Train / Val / Test split
# ---------------------------------------------------------------------
def split_by_period(X_all, y_all, seq_index):
    train_end = seq_index.get_loc(TRAIN_PERIOD[1])
    val_end = seq_index.get_loc(VAL_PERIOD[1])

    X_train = X_all[: train_end + 1]
    y_train = y_all[: train_end + 1]

    X_val = X_all[train_end + 1 : val_end + 1]
    y_val = y_all[train_end + 1 : val_end + 1]

    X_test = X_all[val_end + 1 :]
    y_test = y_all[val_end + 1 :]

    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------
# 4) Model building
# ---------------------------------------------------------------------
def build_model(input_shape):
    model = Sequential([
        # 1층 LSTM: 시퀀스 전체를 다음 층으로 전달
        LSTM(
            64,
            return_sequences=True,
            input_shape=input_shape,
            dropout=0.2,
            recurrent_dropout=0.2,
            kernel_regularizer=l2(1e-4),
        ),
        # 2층 LSTM: 마지막 시점만 출력
        LSTM(
            32,
            dropout=0.2,
            recurrent_dropout=0.2,
            kernel_regularizer=l2(1e-4),
        ),
        Dropout(0.3),
        Dense(16, activation="relu", kernel_regularizer=l2(1e-4)),
        Dense(1, activation="linear"),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="mse",
                  metrics=["mae"])
    return model


# ---------------------------------------------------------------------
# 5) Metric 계산 + 시각화
# ---------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def plot_results(history, y_val_true, y_val_pred, y_test_true, y_test_pred, save_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history.history.get("loss", []), label="train_loss")
    axes[0, 0].plot(history.history.get("val_loss", []), label="val_loss")
    axes[0, 0].legend()
    axes[0, 0].set_title("Train/Val Loss")

    axes[0, 1].plot(y_val_true, label="val_true")
    axes[0, 1].plot(y_val_pred, label="val_pred")
    axes[0, 1].legend()
    axes[0, 1].set_title("Validation Actual vs Pred")

    axes[1, 0].plot(y_test_true, label="test_true")
    axes[1, 0].plot(y_test_pred, label="test_pred")
    axes[1, 0].legend()
    axes[1, 0].set_title("Test Actual vs Pred")

    resid = (y_test_true - y_test_pred).ravel()
    axes[1, 1].hist(resid, bins=15)
    axes[1, 1].set_title("Test Residuals")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------
# 6) Main train pipeline
# ---------------------------------------------------------------------
def main():
    # 데이터 로드 & 전처리
    df = load_and_preprocess(DATA_PATH)

    # 분포 확인(필요 없으면 주석 처리)
    train_slice = df.loc[TRAIN_PERIOD[0] : TRAIN_PERIOD[1], "USD_proc"]
    val_slice = df.loc[VAL_PERIOD[0] : VAL_PERIOD[1], "USD_proc"]
    test_slice = df.loc[TEST_PERIOD[0] : TEST_PERIOD[1], "USD_proc"]
    print("Distribution (USD_proc) — train mean/std:", train_slice.mean(), train_slice.std())
    print("Distribution (USD_proc) — val   mean/std:", val_slice.mean(), val_slice.std())
    print("Distribution (USD_proc) — test  mean/std:", test_slice.mean(), test_slice.std())

    # 스케일링 + 시퀀스 생성
    X_all, y_all, seq_index, scaler_X, scaler_y, feature_cols = create_scaled_sequences(df, LOOKBACK)
    print("전체 시퀀스 shape:", X_all.shape, y_all.shape)
    print("feature 수:", X_all.shape[2])
    print("사용 feature cols:", feature_cols)

    # Train/Val/Test 나누기
    X_train, y_train, X_val, y_val, X_test, y_test = split_by_period(X_all, y_all, seq_index)

    # 모델 생성
    input_shape = (LOOKBACK, X_train.shape[2])
    model = build_model(input_shape)
    model.summary()

    # 콜백 설정: EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
    es = EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5, verbose=1)
    mc = ModelCheckpoint(str(MODEL_PATH), monitor="val_loss", save_best_only=True, verbose=1)

    # 학습
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es, rlrop, mc],
        verbose=1,
    )

    # 예측 + inverse transform
    y_train_pred_s = model.predict(X_train)
    y_val_pred_s = model.predict(X_val)
    y_test_pred_s = model.predict(X_test)

    y_train_true = scaler_y.inverse_transform(y_train)
    y_train_pred = scaler_y.inverse_transform(y_train_pred_s)

    y_val_true = scaler_y.inverse_transform(y_val)
    y_val_pred = scaler_y.inverse_transform(y_val_pred_s)

    y_test_true = scaler_y.inverse_transform(y_test)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_s)

    # 지표 계산
    metrics = {
        "Train": compute_metrics(y_train_true, y_train_pred),
        "Val": compute_metrics(y_val_true, y_val_pred),
        "Test": compute_metrics(y_test_true, y_test_pred),
    }

    metrics_df = pd.DataFrame(metrics).T
    print("\nEvaluation metrics (improved model):")
    print(metrics_df)

    # 스케일러 저장
    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)

    # 시각화 저장
    plot_results(history, y_val_true, y_val_pred, y_test_true, y_test_pred, PLOT_PATH)
    print(f"Saved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
