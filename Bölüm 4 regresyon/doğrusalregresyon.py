import numpy as np
import matplotlib.pyplot as plt

# Örnek veri oluştur
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Veriyi çiz
plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Örnek Veri")
plt.show()

# Doğrusal regresyon modelini oluştur
X_b = np.c_[np.ones((100, 1)), X]  # Bias terimini ekler
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Modelin bulduğu katsayılar
print("Modelin katsayıları:", theta_best.ravel())

# Modeli çiz
plt.scatter(X, y)
plt.plot(X, X_b.dot(theta_best), color='red', label='Doğrusal Regresyon Modeli')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Doğrusal Regresyon Modeli")
plt.show()