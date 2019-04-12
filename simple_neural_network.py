#!/usr/bin/python3
#coding: utf-8

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Sert à cacher les warnings de tensorflow

assert hasattr(tf, "function") # Pour être sûr d'utiliser tensorflow 2.0

#On charge la base de donnée fashion MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
(images, targets),(_,_) = fashion_mnist.load_data()

#On ne garde qu'une partie de la base de donnée
images = images[:10000]
targets = targets[:10000]

#On note le type de vêtement associé à chaque classe
targets_name = ["Tshirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

#Affiche une image de la dataset
# plt.imshow(images[20], cmap="binary")
# plt.title(targets_name[targets[20]])
# plt.show()

#Création du modèle: 28*28 pixels -> 256 neurones -> 128 neurones -> 10 neurones
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[28,28])) #Permet de transformer une image 28*28 en un vecteur de 28*28 éléments sur une ligne.

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

history = model.fit(images,targets,epochs=20) #entraine le modèle avec nos images, les targets et le nombre d'epochs voulu
model_output = model.predict(images[0:1])
print(model_output, targets[0:1])

#On affiche l'évolution de l'erreur et de la précision en fonction du nombre d'epoch
loss_curve = history.history["loss"]
acc_curve = history.history["accuracy"]

plt.plot(loss_curve)
plt.title("Loss")
plt.show()

plt.plot(acc_curve)
plt.title("Accuracy")
plt.show()
