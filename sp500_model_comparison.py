import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import warnings

# 忽略不必要的警告訊息，保持輸出畫面整潔
warnings.filterwarnings('ignore')

def main():
    # ==========================================
    # 1. 資料下載與前處理
    # ==========================================
    print("Downloading S&P 500 data (2021-2024 train, 2025 test)...")
    
    # 設定目標股票代號為 S&P 500 指數 (^GSPC)
    ticker = '^GSPC'
    
    # 使用 yfinance 下載 2021-01-01 到 2025-12-31 的歷史交易資料
    data = yf.download(ticker, start='2021-01-01', end='2025-12-31')

    # 處理新版 yfinance 可能回傳的多重索引 (MultiIndex) 欄位問題
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # 定義我們要使用的特徵 (開盤價、最高價、最低價、收盤價、交易量)
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # 檢查這些特徵是否都存在於下載的資料中，若無則報錯
    for f in features:
        if f not in data.columns:
            raise ValueError(f"Feature {f} not found in downloaded data.")

    # 設定預測目標：將收盤價往上移一天，表示我們要「用今天的特徵，預測明天的收盤價」
    data['Target'] = data['Close'].shift(-1)
    
    # 移除含有缺失值 (NaN) 的資料（最後一天因為沒有明天的收盤價，所以 Target 會是 NaN）
    data.dropna(inplace=True)

    # ==========================================
    # 2. 資料切分 (訓練集與測試集)
    # ==========================================
    # 使用時間區段進行資料切分（訓練：2021-2024，測試：2025）
    train_data = data.loc[:'2024-12-31']
    test_data = data.loc['2025-01-01':]

    # 分離特徵 (X) 與 目標變數 (y)
    X_train = train_data[features]
    y_train = train_data['Target']

    X_test = test_data[features]
    y_test = test_data['Target']

    # 印出訓練集與測試集的資料筆數
    print(f"Train data (2021-2024): {len(train_data)} samples")
    print(f"Test data (2025): {len(test_data)} samples")

    # 防呆機制：如果測試集是空的，代表下載資料可能有誤或日期未到，程式提早結束
    if len(test_data) == 0:
        print("Warning: Test data is empty. Cannot evaluate models.")
        return

    # ==========================================
    # 3. 訓練模型
    # ==========================================
    print("Training Random Forest...")
    # 建立隨機森林迴歸模型，設定 100 棵決策樹，並固定 random_state 確保每次執行結果一致
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    print("Training XGBoost...")
    # 建立 XGBoost 迴歸模型，設定 100 棵樹，並固定 random_state
    xgb_model = XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)

    # ==========================================
    # 4. 模型評估 (計算並比較 MSE)
    # ==========================================
    # 使用測試集進行預測
    rf_preds = rf_model.predict(X_test)
    xgb_preds = xgb_model.predict(X_test)

    # 計算均方誤差 (Mean Squared Error, MSE) 來衡量預測準確度
    rf_mse = mean_squared_error(y_test, rf_preds)
    xgb_mse = mean_squared_error(y_test, xgb_preds)

    print(f"--- Results ---")
    print(f"Random Forest MSE: {rf_mse:.2f}")
    print(f"XGBoost MSE:       {xgb_mse:.2f}")

    # ==========================================
    # 5. 結果視覺化與輸出成 PDF
    # ==========================================
    # 設定圖表大小
    plt.figure(figsize=(12, 6))
    
    # 繪製真實的隔日收盤價 (黑線)
    plt.plot(test_data.index, y_test, label='Actual Next-Day Close', color='black', linewidth=2)
    
    # 繪製隨機森林的預測結果 (藍線)
    plt.plot(test_data.index, rf_preds, label='Random Forest Predictions', color='blue', alpha=0.7)
    
    # 繪製 XGBoost 的預測結果 (紅線)
    plt.plot(test_data.index, xgb_preds, label='XGBoost Predictions', color='red', alpha=0.7)

    # 設定圖表標題、X軸/Y軸標籤
    plt.title('S&P 500 Price Prediction - Year 2025', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Next Day Close Price', fontsize=12)
    
    # 顯示圖例
    plt.legend(fontsize=12)
    
    # 加上網格線方便閱讀
    plt.grid(True, linestyle='--', alpha=0.6)

    # 在圖表左上方區塊加入文字方塊，顯示兩種模型的 MSE 評估結果
    info_text = f"Evaluation (MSE):\nRandom Forest: {rf_mse:.2f}\nXGBoost: {xgb_mse:.2f}"
    plt.text(0.02, 0.95, info_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', 
             bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.9})

    # 確保圖表排版不會互相擠壓重疊
    plt.tight_layout()
    
    # 決定 PDF 的儲存路徑（儲存在與此程式相同的資料夾下，檔名為 HW1_Report.pdf）
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HW1_Report.pdf')
    
    # 儲存圖表成 PDF
    plt.savefig(pdf_path)
    print(f"Saved prediction plot to {pdf_path}")

# 確保當這支檔案被直接執行時，才會呼叫 main 函數
if __name__ == "__main__":
    main()
