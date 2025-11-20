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

# 6) 상태 출력
print("df shape:", df.shape)
print("X_train_seq shape:", X_train_seq.shape)
print("y_train_seq shape:", y_train_seq.shape)
print("X_val_seq shape:", X_val_seq.shape)
print("X_test_seq shape:", X_test_seq.shape)

# 필요한 객체들을 이후 모델 학습에 사용할 수 있도록 노출
# df, train, val, test, scaler_X, scaler_y, X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq