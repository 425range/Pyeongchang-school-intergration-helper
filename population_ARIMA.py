import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pmdarima import auto_arima

# 파일 로딩 및 전처리
df = pd.read_excel("./완성자료/통합 연령별 인구자료.xlsx", sheet_name="Sheet1")
df_ts = df.set_index("연도")
df_ts.columns = ["PyeongChang", "Yangyang", "Hongcheon", "Hoengseong"]

regions = ["PyeongChang", "Yangyang", "Hongcheon", "Hoengseong"]
years = list(range(2026, 2031))
forecast_results = {}

plt.figure(figsize=(12, 8))

for i, region in enumerate(regions, 1):
    data = df_ts[region].dropna()

    # ARIMA 차수 자동 추정 (AIC 기반)
    stepwise_model = auto_arima(data,
                                 start_p=0, start_q=0,
                                 max_p=3, max_q=3,
                                 seasonal=False,
                                 trace=False,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=True)

    # 예측
    forecast = stepwise_model.predict(n_periods=5)
    forecast = np.nan_to_num(forecast, nan=0.0, posinf=0.0, neginf=0.0)
    forecast = pd.Series(forecast, index=years).round().astype(int)

    # 인구 감소율 계산
    decrease_rate = 100 * (forecast.iloc[-1] - data.iloc[-1]) / data.iloc[-1]

    forecast_results[region] = {
        "years": years,
        "forecast": forecast.tolist(),
        "decrease_rate": decrease_rate,
        "order": stepwise_model.order
    }

    # 시각화
    plt.subplot(2, 2, i)
    plt.plot(data.index, data, label="Actual", marker='o')
    plt.plot(forecast.index, forecast, label="Forecast", marker='x', linestyle='--')
    plt.title(f"{region} (ARIMA{stepwise_model.order})\nDecrease Rate: {decrease_rate:.2f}%", fontsize=10)
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.legend()

plt.tight_layout()
plt.suptitle("Population Forecast and Decreasing Rate for 4 Counties in Gangwon-do", fontsize=16, y=1.03)
plt.show()
