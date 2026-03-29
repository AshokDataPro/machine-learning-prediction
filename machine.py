import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- Load + Clean
df = pd.read_csv(r"C:\Users\dell\.vscode\NIFTY 50.csv")
df.columns = df.columns.str.strip()
df = df.set_index("Date")
df.index = pd.to_datetime(df.index, format="%d-%b-%y")
print(df.head(10))
print(df.tail(10))

# --- Feature Engineering
df["Close_next"] = df["Close"].shift(-1)
df["close_lag1"] = df["Close"].shift(1)
df["close_lag2"] = df["Close"].shift(2)
df["shared_value"] = df["Shares Traded"].shift(1)
df["15_ema"] = df["Close"].rolling(window=15).mean()
df["25_ema"] = df["Close"].rolling(window=25).mean()
df["50_ema"] = df["Close"].rolling(window=50).mean()
df["pct_close"] = df["Close"].pct_change() * 100
 



# RSI
delta = df["Close"].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

# Drop NaNs from feature columns
features = ["close_lag1", "close_lag2", "shared_value", "15_ema", 
            "25_ema", "50_ema", "pct_close", "RSI"]
target = "Close_next"

df.dropna(subset=features + [target], inplace=True)



# --- Split data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- Scaled Ridge Regression Pipeline

 

last_value = df.iloc[-1]['Close']  # adjust column name

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Next predicted value from {last_value} is {round(y_pred[0])}")
# --- Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
 

print("🔍 Model Evaluation Metrics")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("R²:", round(r2, 4))
showss=  last_value / y_pred[0] * 100
print(showss.round(2))

# --- Plot: Actual vs Predicted
plt.figure(figsize=(14, 6))
plt.plot(y_test.index, y_test, label='Actual', color='blue', linewidth=2)
plt.plot(y_test.index, y_pred, label='Predicted', color='orange', linestyle='--', linewidth=2)
plt.title("📈 Actual vs Predicted Close Prices")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot: Residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 4))
sns.histplot(residuals, bins=40, kde=True, color="purple")
plt.title("Residual Distribution (Error = Actual - Predicted)")
plt.xlabel("Residuals")
plt.tight_layout()
plt.show()


plt.figure(figsize=(12,8))
sns.scatterplot(x="Close", y="Open", data=df, hue='Shares Traded', palette="cool")
plt.xlabel("close price view")
plt.ylabel("open price view")
plt.title("this is open Vs close")
plt.show()
