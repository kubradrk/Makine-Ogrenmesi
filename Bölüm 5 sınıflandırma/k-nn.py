from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Veri kümesini yükle
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# K-NN sınıflandırıcı modelini oluştur
k = 5
model = KNeighborsClassifier(n_neighbors=k)

# Modeli eğit
model.fit(X_train, y_train)

# Test verisiyle modelin performansını değerlendir
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Modelin doğruluk skoru:", accuracy)