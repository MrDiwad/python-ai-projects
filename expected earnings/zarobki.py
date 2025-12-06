import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x=np.array([[1], [2], [4], [6], [9], [12]])
y=np.array([5500, 7000, 10000, 13500, 19000, 25000])

model = LinearRegression()

model.fit(X=x,y=y)

x15=model.predict([[15]])

print(x15)

plt.scatter(x,y)
plt.plot(x,model.predict(x))
plt.show()