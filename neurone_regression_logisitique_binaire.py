import matplotlib.pyplot as plt
import numpy as np

########## CLASSE #########
class Neurone():
    """Classe représentant un neurone de machine learning. L'entrainement repose sur
    l'algorithme de la descente de gradient et le modèle est celui de la regression
    logisitique. L'apprentissage est ici supervisé. Ce neurone peut donc classer un
    jeu de donnée en tracant une droite délimitant nos deux classes de sorties.
    Dans cette implémentation, nous n'avons que deux classes possibles, on parle
    de régression logistique binaire. Nous utiliserons comme fonction de préactivation
    la somme pondérée des entrée avec les poids à laquelle on rajoute le bias.
    La fonction d'activation est la sigmoïde
    """

    def __init__(self, features, targets):
        self.features = features #Jeu de donnée que l'on souhaite trier
        self.targets = targets #Valeurs de références pour l'apprentissage supervisé
        self.bias = 0. #Affine à l'origine de notre droite de délimitation. Se généralise en dimension supérieures à des hyperplans.
        self.weights = np.random.normal(size=2) #Poids associés aux données en entrées du neurone. Par défaut, ceux-ci sont aléatoires.
        self.preactivate = 0 #Valeur de préactivation
        self.activate = 0 #Valeur d'activation, correspond à une probabilité d'appartenir ou non à une classe
        self.prediction = 0 #Classe prédite. Il s'afit de la valeur d'activation arrondie dans le cas d'une regression logisitique binaire comme ici.

    def __str__(self):
        """Gère l'affichage de l'objet via la fonction print"""
        return "Neurone utilisant la régression logistique caractérisé par \nSes poids {} \nEt son bias {}".format(self.weights,self.bias)

    def update_dataset(self, features, targets):
        """Méthode permettant de modifier les valeurs de la dataset afin de faire des prédiction après l'entrainement"""
        self.features = features
        self.targets = targets

    def activation(self,preact):
        """Retourne la sigmoïde de la valeur d'entrée"""
        self.activate = 1 / (1 + np.exp(-preact))
        return self.activate

    def derivative_activation(self,preact):
        """Retourne la dérivé de la sigmoide de la valeur d'entrée"""
        return self.activation(preact) * (1 - self.activation(preact)) #Formule à calculer soi-même ou à trouver sur internet

    def pre_activation(self,features, weights, bias):
        """Calcule la valeur de la préactivation"""
        self.preactivate = np.dot(features, weights) + bias
        return self.preactivate

    def predict(self):
        """Effectue la prédiction en arrondissant la valeur de l'activation"""
        self.preactivate = self.pre_activation(self.features, self.weights, self.bias)
        self.activate = self.activation(self.preactivate)
        #On arrondi la valeur d'activation afin d'obtenir 0 ou 1, correspondant à nos deux classes.
        self.prediction = np.round(self.activate)
        return self.prediction

    def cost(self,predictions, targets):
        """Caclule l'erreur"""
        return np.mean((predictions - targets) ** 2)

    def train(self):
        """Entraine le modèle grace à l'algorithme de la descente de gradient"""
        features = self.features
        targets = self.targets
        weights = self.weights
        bias = self.bias

        epochs = 100 #Nombre de fois où notre base de donnée labelisée passera dans notre neurone pendant l'entrainement.
        learning_rate = 0.1 #Proportion du gradient à ajouter à nos poids et notre bias. Cette valeur influe directement la qualité de l'apprentissage.

        print("Début de l'entrainement ! \n")

        predictions = self.predict() #On effectue une première prédiction aléatoire
        print("Précision initiale = %s" % np.mean(predictions == targets)) #On calcule notre erreur.

        for epoch in range(epochs): #On itère sur chaque passage de la base de donnée dans le neurone
            if epoch % 10 == 0: #On affiche l'erreur tous les 10 passages pour voir la progression du neurone.
                predictions = self.activation(self.pre_activation(features, weights, bias))
                print("Erreur actuelle = %s" % self.cost(predictions, targets))

            weights_gradients = np.zeros(weights.shape) #On initialise les variables des gradients.
            bias_gradient = 0.

            for feature, target in zip(features, targets): #On parcours ligne par ligne nos deux tableaux.
                preact = self.pre_activation(feature, weights, bias)
                act = self.activation(preact)
                weights_gradients += (act - target)*self.derivative_activation(preact)*feature #Calcul des gradients.
                bias_gradient += (act-target)*self.derivative_activation(preact)

            weights = weights - (learning_rate * weights_gradients) #Descente de gradient pour ajuster les poids et le bias
            bias = bias - (learning_rate * bias_gradient)
            self.weights = weights
            self.bias = bias

        predictions = self.predict()
        print("Précision finale = %s" % np.mean(predictions == targets))
        print("Entrainement terminé ! \n")



#########FONCTIONS#########

def get_dataset():
    """Génère une base de donnée labelisée aléatoire"""
    # Nombre de lignes par classe
    row_per_class = 100
    # Génère les lignes de nos deux classes
    sick = np.random.randn(row_per_class, 2) + np.array([-2, -2]) #On sépare bien les deux classes pour aider le neurone dans son apprentissage.
    healthy = np.random.randn(row_per_class, 2) + np.array([2, 2]) #Etant en présence d'un seul neurone, celui-ci pourra seulement tracer une ligne de délimitation.

    features = np.vstack([sick, healthy]) #On empile nos deux tableaux pour obtenir notre base de donnée.
    targets = np.concatenate((np.zeros(row_per_class), np.zeros(row_per_class) + 1)) #On crée les labels.

    return features, targets

##########MAIN#########

if __name__ == "__main__":

#On va d'abord créer deux bases de données. On entraine notre neurone sur la première et on le teste sur la seconde. Programme à lancer plusieurs fois
#afin d'observer l'efficacité du neurone !

    features1, targets1 = get_dataset()
    features2, targets2 = get_dataset()

    print("Affichage de la première base de donnée :\n")
    plt.scatter(features1[:, 0], features1[:, 1], s=40, c=targets1, cmap=plt.cm.Spectral) #Sert à afficher la première base de donnée
    plt.show()

    n = Neurone(features1, targets1) #Création du neurone
    n.train()#Entrainement du neurone

    n.update_dataset(features2,targets2) #Changement de données pour tester son entrainement.
    print("Affichage de la deuxième base de donnée :\n")
    plt.scatter(features2[:, 0], features2[:, 1], s=40, c=targets2, cmap=plt.cm.Spectral)#Sert à afficher la deuxième base de donnée
    plt.show()
    print("Précision sur la nouvelle dataset = %s" % np.mean(n.predict() == targets2))
