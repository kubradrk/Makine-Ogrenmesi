import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sigmoid aktivasyon fonksiyonu
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid aktivasyon fonksiyonunun türevi
def sigmoid_derivative(x):
    return x * (1 - x)

# Veri kümesini yükle
iris = load_iris()
X = iris.data
y = iris.target

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendir
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model parametreleri
input_neurons = X_train_scaled.shape[1]
hidden_neurons = 10
output_neurons = len(np.unique(y_train))
learning_rate = 0.1
epochs = 1000

# Ağırlıkları ve bias'ı rastgele başlat
np.random.seed(0)
weights_input_hidden = np.random.rand(input_neurons, hidden_neurons)
bias_input_hidden = np.random.rand(1, hidden_neurons)
weights_hidden_output = np.random.rand(hidden_neurons, output_neurons)
bias_hidden_output = np.random.rand(1, output_neurons)

# Eğitim
for epoch in range(epochs):
    # İleri yayılım
    hidden_layer_input = np.dot(X_train_scaled, weights_input_hidden) + bias_input_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_hidden_output
    output_layer_output = sigmoid(output_layer_input)
    
    # Hata hesaplama
    error = y_train.reshape(-1,1) - output_layer_output
    
    # Geri yayılım
    d_output = error * sigmoid_derivative(output_layer_output)
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Ağırlık güncelleme
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    bias_hidden_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += X_train_scaled.T.dot(d_hidden_layer) * learning_rate
    bias_input_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# Test
hidden_layer_input = np.dot(X_test_scaled, weights_input_hidden) + bias_input_hidden
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_hidden_output
output_layer_output = sigmoid(output_layer_input)

predictions = np.argmax(output_layer_output, axis=1)
accuracy = np.mean(predictions == y_test)
print("Modelin doğruluk skoru:", accuracy)
