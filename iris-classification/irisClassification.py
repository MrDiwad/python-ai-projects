import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

nowy_kwiat = np.array([[5.0, 3.6, 1.4, 0.2]])
iris=load_iris()
X=iris.data
y=iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = KNeighborsClassifier(3)

model.fit(X=X_train,y=y_train)

nowy_kwiat_pred=model.predict(nowy_kwiat)

print(f"\nModel przewidział dla Nowego Kwiatu kategorię: {nowy_kwiat_pred[0]}")

# Tłumaczymy to na nazwę gatunku (DODATKOWA LOGIKA)
print(f"Co oznacza gatunek: {iris.target_names[nowy_kwiat_pred[0]]}")

y_pred=model.predict(X_test)

acc = model.score(X_test,y_test)

print(f"\nDokładność modelu na danych testowych: {acc:.2f}")