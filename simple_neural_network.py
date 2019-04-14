#!/usr/bin/python3
#coding: utf-8

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Sert à cacher les warnings de tensorflow 2.0 alpha

assert hasattr(tf, "function") # Pour être sûr d'utiliser tensorflow 2.0

#On charge la base de donnée fashion MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
(images, targets),(_,_) = fashion_mnist.load_data()

#On ne garde qu'une partie de la base de donnée
images = images[:10000]
targets = targets[:10000]

#On note le type de vêtement associé à chaque classe
targets_name = ["Tshirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

images = images.reshape(-1,784)
images = images.astype(float)
scaler = StandardScaler()
images = scaler.fit_transform(images) #Afin de flatten les images

images_train, images_test, targets_train, targets_test = train_test_split(images, targets, test_size=0.2, random_state=1) #Permet de créer un train set et un validation set

#Affiche une image de la dataset
# plt.imshow(images[20], cmap="binary")
# plt.title(targets_name[targets[20]])
# plt.show()

#Création du modèle: 28*28 pixels -> 256 neurones -> 128 neurones -> 10 neurones
model = tf.keras.models.Sequential()

#print("Forme de l'image", images[0:1].shape)
#model_output = model.predict(images[0:1]) #Execute les opérations définies via le add. Ici cela applatit donc juste l'image

#On ajoute les couches de neurones
model.add(tf.keras.layers.Dense(256, activation="relu")) #Ajoute dans la séquence la création d'une couche de 256 neurones liés avec l'opération précédente, c'est à dire les pixels de l'image, avec l'activation relu
model.add(tf.keras.layers.Dense(128, activation="relu")) #Ajoute dans la séquence la création d'une couche de 128 neurones liés à l'opération précédente de la séquence, donc aux 256 neurones précédents
model.add(tf.keras.layers.Dense(10, activation="softmax"))

#model_output = model.predict(images[0:1]) #Prédit la sortie, en étant pour l'instant non entrainé
#print(model_output, targets[0:1])#affiche la prédiction et la valeur attendue

#model.summary() # Donne une vue d'ensemble du modèle

#Compilation du modèle
model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"]) #Définit entre autre la fonction d'erreur à utiliser. sgd = stochastic gradient descent. accuracy est le pourcentage de prédiction correcte.

history = model.fit(images_train,targets_train,epochs=50, validation_split=0.2) #entraine le modèle avec nos images, les targets et le nombre d'epochs voulu
model_output = model.predict(images[0:1])

#On affiche l'évolution de l'erreur et de la précision en fonction du nombre d'epoch
loss_curve = history.history["loss"] #Erreur sur les valeurs du jeu d'entrainement
acc_curve = history.history["accuracy"]#Précision sur les valeurs du jeu d'entrainement

loss_val_curve = history.history["val_loss"]#Erreur sur les valeurs du jeu de validation
acc_val_curve = history.history["val_accuracy"]#Précision sur les valeurs du jeu de validation

plt.plot(loss_curve, label="Train loss")
plt.plot(loss_val_curve, label="Validation loss")
plt.legend(loc="upper right")
plt.title("Train loss VS validation loss")
plt.show() #Affichage des courbes
