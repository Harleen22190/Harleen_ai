import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import joblib

x = np.array([[100], [200], [300], [400], [500], [600]])
y = np.array([10000, 20000, 30000, 40000, 50000, 60000])

model = LinearRegression()
model.fit(x, y)
joblib.dump(model, "house_price.pkl")

x_pred = 800
y_pred = model.predict([[x_pred]])[0]

x_line = np.array([[x.min()], [x_pred]])
y_line = model.predict(x_line)

plt.scatter(x, y)
plt.plot(x_line, y_line)

plt.scatter(x_pred, y_pred)

plt.xlabel("Area")
plt.ylabel("Price")

plt.show()
print("Model trained and saved successfully")

