import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -----  ACF 함수 -----
def acf(x, nlags):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    acf_vals = []
    # lag 0부터 nlags까지 계산
    for lag in range(nlags + 1):
        acf_vals.append(np.corrcoef(x[:-lag] if lag>0 else x,
                                    x[lag:] if lag>0 else x)[0, 1])
    return np.array(acf_vals)

# ----- PACF 함수 -----
def pacf(x, nlags):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    pacf_vals = [1.0]  # lag 0
    # lag 1부터 nlags까지 계산
    for k in range(1, nlags + 1):
        y = x[k:]
        X = np.column_stack([x[k - i - 1: -i - 1 or None] for i in range(k)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pacf_vals.append(coef[-1])  # 마지막 계수가 partial corr
    return np.array(pacf_vals)

# ----- 시뮬레이션 유틸 -----
# AR(2): X_t = 1.5 X_{t-1} - 0.75 X_{t-2} + a_t
def simulate_ar2(phi1=1.5, phi2=-0.75, n=1200, burnin=200, sigma=1.0, seed=42):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, sigma, n + burnin)
    x = np.zeros(n + burnin)
    # 초기 두 값은 백색소음으로 시작
    x[0], x[1] = a[0], a[1]
    for t in range(2, n + burnin):
        x[t] = phi1 * x[t-1] + phi2 * x[t-2] + a[t]
    return x[burnin:]  

# MA(2): X_t = a_t + 0.7 a_{t-1} + 0.8 a_{t-2}
def simulate_ma2(theta1=0.7, theta2=0.8, n=1200, burnin=200, sigma=1.0, seed=123):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, sigma, n + burnin + 2)  # 초깃값 위한 여유분
    x = np.zeros(n + burnin)
    for t in range(n + burnin):
        x[t] = a[t+2] + theta1 * a[t+1] + theta2 * a[t]
    return x[burnin:] 

def plot_series(y, title="Time Series", nshow=400):
    plt.figure(figsize=(10, 3))
    plt.plot(y[:nshow])
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("x_t")
    plt.tight_layout()
    plt.show()

def plot_stem(values, title="ACF / PACF", xlabel="lag"):
    # 0-lag 포함 막대그래프
    lags = np.arange(len(values))
    plt.figure(figsize=(8, 3))
    markerline, stemlines, baseline = plt.stem(lags, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("value")
    # 95% 신뢰구간 +-1.96/sqrt(N)
    N = len(values)
    if N > 0:
        pass
    plt.axhline(0, linewidth=1)
    plt.tight_layout()
    plt.show()

# ----- 실행 -----
if __name__ == "__main__":
    # 1) AR(2) 생성
    ar2 = simulate_ar2(phi1=1.5, phi2=-0.75, n=2000, burnin=300, sigma=1.0, seed=7)
    # 2) MA(2) 생성
    ma2 = simulate_ma2(theta1=0.7, theta2=0.8, n=2000, burnin=300, sigma=1.0, seed=9)

    # ACF/PACF 계산
    NLAGS = 140
    ar2_acf = acf(ar2, NLAGS)
    ar2_pacf = pacf(ar2, NLAGS)
    ma2_acf = acf(ma2, NLAGS)
    ma2_pacf = pacf(ma2, NLAGS)

    # ----- AR(2) 시각화 -----
    plot_series(ar2, title="AR(2) time series (phi1=1.5, phi2=-0.75)", nshow=500)
    plot_stem(ar2_acf, title="AR(2) ACF", xlabel="lag")
    plot_stem(ar2_pacf, title="AR(2) PACF", xlabel="lag")

    # ----- MA(2) 시각화 -----
    plot_series(ma2, title="MA(2) time series (theta1=0.7, theta2=0.8)", nshow=500)
    plot_stem(ma2_acf, title="MA(2) ACF", xlabel="lag")
    plot_stem(ma2_pacf, title="MA(2) PACF", xlabel="lag")

# ----- 연령별 인구자료 ACF/PACF -----
df = pd.read_excel("./완성자료/통합 연령별 인구자료.xlsx", sheet_name="Sheet1")
series = df["PyeongChang"].astype(float)
x = series.values

# Lags = 15
NLAGS = 15
series_acf = acf(x, NLAGS)
series_pacf = pacf(x, NLAGS)
# 95% 신뢰구간
ci = 1.96 / np.sqrt(len(x))
# ACF/PACF 시각화
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].stem(range(NLAGS+1), series_acf)
ax[0].axhline(ci, ls="--"); ax[0].axhline(-ci, ls="--")
ax[0].set_title("ACF")

ax[1].stem(range(NLAGS+1), series_pacf)
ax[1].axhline(ci, ls="--"); ax[1].axhline(-ci, ls="--")
ax[1].set_title("PACF")

plt.tight_layout()
plt.show()