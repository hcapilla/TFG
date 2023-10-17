import numpy as np
import os
import random
import cv2

# Sol·licitar la variable "mida" a l'usuari
mida = int(input("Introdueix la variable 'mida': "))

# Sol·licitar el número d'iteracions "nIteracions"
nIteracions = int(input("Introdueix el nombre d'iteracions: "))

# Crear un directori "banc" si no existeix
directori_banc = "banc"
if not os.path.exists(directori_banc):
    os.mkdir(directori_banc)

seed = np.random.randint(0, 1000)

# Generar i desar les imatges en el directori "banc"
for i in range(nIteracions):
    # Generar una nova matriu amb valors aleatoris
    nova_matriu = np.random.randint(0, 256, size=(mida, mida))

    # Desar la matriu com una imatge .png en el directori "banc"
    nom_arxiu = os.path.join(directori_banc, f"{mida}_{seed}_{i}.png")
    cv2.imwrite(nom_arxiu, nova_matriu)

print(f"S'han generat i desat {nIteracions} imatges al directori 'banc' com a arxius .png.")
