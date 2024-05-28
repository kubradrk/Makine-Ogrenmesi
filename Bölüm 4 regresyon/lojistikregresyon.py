# Gerekli kütüphaneleri içe aktarın
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Örnek veri setini oluşturun
X = np.array([[2], [3], [4], [5], [6], [7], [8], [9]])  # Öğrencilerin çalışma saatleri
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # Sınavda başarılı olup olmadıkları (0: Başarısız, 1: Başarılı)

# Lojistik regresyon modelini oluşturun ve eğitin
model = LogisticRegression()
model.fit(X, y)

# Modelin doğrusal karar sınırını çizin
x_values = np.linspace(1, 10, 100)
y_values = model.predict_proba(x_values.reshape(-1, 1))[:, 1]  # Tahmin olasılıklarını alın

# Veriyi ve doğrusal karar sınırını çizin
plt.scatter(X, y, color='blue')
plt.plot(x_values, y_values, color='red')
plt.xlabel('Çalışma Saatleri')
plt.ylabel('Sınav Sonucu (0: Başarısız, 1: Başarılı)')
plt.title('Lojistik Regresyon')
plt.show()
