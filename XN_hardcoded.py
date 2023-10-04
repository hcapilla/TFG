import numpy as np
import cv2
import os
import random

def generate_pixel_value(probabilities):
    rand = np.random.rand()
    if rand < probabilities[0]:
        return np.random.randint(0, 100)
    elif rand < probabilities[1]:
        return np.random.randint(0, 50)
    elif rand < probabilities[2]:
        return np.random.randint(226, 256)
    elif rand < probabilities[3]:
        return np.random.randint(0, 50)
    elif rand < probabilities[4]:
        return np.random.randint(0, 130)
    elif rand < probabilities[5]:
        return np.random.randint(201, 256)
    elif rand < probabilities[6]:
        return np.random.randint(0, 150)
    elif rand < probabilities[7]:
        return np.random.randint(201, 256)
    else:
        return np.random.randint(231, 256)

def generate_image():
    probabilities = [0.8, 0.9, 0.9, 0.9, 0.8, 0.7, 0.9, 0.7, 0.9]
    image = np.zeros((3, 3), dtype=np.uint8)
    for i in range(3):
        for j in range(3):
            image[i, j] = generate_pixel_value(probabilities)
    return image

# 3x3 escala de grisos - 1
def features_0_1_1i():
    output_dir = 'XN_features_0_1'
    os.makedirs(output_dir, exist_ok=True)

    for i in range(300):
        image = generate_image()
        filename = os.path.join(output_dir, f'1_{i}.png')
        cv2.imwrite(filename, image)

# 3x3 colors - 32
def features_0_1_32i():
    output_dir = 'XN_features_0_1'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for i in range(300):
        # Crear una nueva imagen de 3x3 en blanco
        img = np.zeros((3, 3, 3), dtype=np.uint8)

        # Definir los colores para cada franja
        cyan = (0, 255, 255)
        magenta = (255, 0, 255)
        red = (0, 0, 255)

        # Generar valores de color aleatorios dentro de los rangos de cada franja
        cyan_random = np.random.randint(0, 256, size=(1, 3), dtype=np.uint8)
        magenta_random = np.random.randint(0, 256, size=(1, 3), dtype=np.uint8)
        red_random = np.random.randint(0, 256, size=(1, 3), dtype=np.uint8)

        # Rellenar la franja superior con color cian aleatorio
        img[0:1, :, :] = cyan_random

        # Rellenar la franja central con color magenta aleatorio
        img[1:2, :, :] = magenta_random

        # Rellenar la franja inferior con color rojo aleatorio
        img[2:3, :, :] = red_random

        # Guardar la imagen en la carpeta con el nombre adecuado
        filename = os.path.join(output_dir, f'32_{i}.png')
        cv2.imwrite(filename, img)

if __name__ == '__main__':
    features_0_1_1i()
    features_0_1_32i()
