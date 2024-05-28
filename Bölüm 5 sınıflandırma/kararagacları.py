from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Iris veri setini yükleyin
iris = load_iris()
X = iris.data
y = iris.target

# Karar ağacı modelini oluşturun
model = DecisionTreeClassifier()

# Modeli eğitirken maliyeti ölçün
model.fit(X, y)

# Modelin derinliğini, yaprak sayısını ve dal sayısını alın
depth = model.get_depth()
leaves = model.get_n_leaves()
nodes = model.tree_.node_count

# Maliyeti hesapla
# Genellikle, karar ağaçlarının maliyeti, ağacın derinliği, yaprak sayısı ve dal sayısı gibi faktörlere bağlıdır.
# Burada basit bir maliyet örneği veriliyor: derinlik + yaprak sayısı + dal sayısı
cost = depth + leaves + nodes

print("Ağaç derinliği:", depth)
print("Yaprak sayısı:", leaves)
print("Dal sayısı:", nodes)
print("Toplam maliyet:", cost)
