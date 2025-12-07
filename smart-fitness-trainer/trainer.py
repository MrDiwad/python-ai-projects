import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

X = np.array([
    [2000, 30, 90],   # Spacer
    [3000, 45, 100],  # Szybki marsz
    [5000, 40, 135],  # Bieganie
    [1000, 15, 85],   # Krótkie wyjście
    [8000, 60, 150],  # Mocne cardio
    [6000, 50, 140],  # Jogging
    [2500, 30, 95],   # Spacer z psem
    [9000, 60, 155]   # Bardzo mocny trening
])

# CEL 1: REGRESJA (Ile kalorii spalono?)
y_kalorie = np.array([150, 220, 400, 70, 650, 480, 180, 700])

# CEL 2: KLASYFIKACJA (Jaki to typ treningu?)
# 0 = Spacer (Lekki), 1 = Cardio (Średni), 2 = Wyczyn (Ostry)
y_intensywnosc = np.array([0, 0, 1, 0, 2, 1, 0, 2])

model_reg=LinearRegression()
model_clas=KNeighborsClassifier(3)

model_reg.fit(X=X,y=y_kalorie)
model_clas.fit(X=X,y=y_intensywnosc)

new_training=[[6500,55,142]]

print(model_reg.predict(new_training))

typeOfTraining=model_clas.predict(new_training)

index=int(typeOfTraining[0])

cateogries = ["Light","Medium","Hard"]

print(cateogries[index])