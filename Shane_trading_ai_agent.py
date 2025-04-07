# File: trading_ai_agent.py

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# 🧙‍ Shane - LSTM AI Trading Agent

# 1. Tải dữ liệu
df = yf.download("BTC-USD", start="2022-01-01", end="2023-01-01")
df["Return"] = df["Close"].pct_change()
df["SMA_10"] = df["Close"].rolling(10).mean()
df["SMA_50"] = df["Close"].rolling(50).mean()
df = df.dropna()

# 2. Tiến trình dữ liệu cho LSTM
features = ["SMA_10", "SMA_50", "Return"]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

sequence_length = 20
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    # Gắn nhãn 1 nếu giá tăng, 0 nếu giá giảm
    y.append(1 if df["Close"].iloc[i] > df["Close"].iloc[i-1] else 0)

X = np.array(X)
y = np.array(y)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# 3. Xây dựng mô hình LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

model = LSTMModel(input_size=3, hidden_size=64, num_layers=2)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Huấn luyện mô hình
for epoch in range(10):
    output = model(X_tensor)
    loss = loss_fn(output, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 2 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 5. Dự đoán và gán tín hiệu
with torch.no_grad():
    predictions = model(X_tensor)
    predicted_labels = torch.argmax(predictions, dim=1).numpy()

df = df.iloc[sequence_length:]
df["Predicted"] = predicted_labels

# 6. Quản lý rủi ro
risk_threshold = 0.02  # cắt lỗ 2%
df["Position"] = df["Predicted"].shift(1)
df["Entry_Price"] = df["Close"].shift(1)
df["Drawdown"] = (df["Close"] - df["Entry_Price"]) / df["Entry_Price"]
df["Position"] = np.where(df["Drawdown"] < -risk_threshold, 0, df["Position"])

# 7. Tính lợi nhuận
df["Strategy Return"] = df["Return"] * df["Position"]
df[["Return", "Strategy Return"]].cumsum().plot()
plt.title("Shane: So sánh lợi nhuận - Thị trường vs Agent")
plt.show()
