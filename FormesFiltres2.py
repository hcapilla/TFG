import cv2
import numpy as np
import matplotlib.pyplot as plt

def crea_Imatge():
    # Inicialitzar l'array dinàmic
    imatges = []

    while True:
        # Demanar a l'usuari una mida o "f" per finalitzar
        entrada = input("Introdueix una mida per a la imatge (o 'f' per finalitzar): ")

        if entrada.lower() == 'f':
            break  # Sortir del bucle si s'entra "f"

        try:
            # Convertir l'entrada a un enter
            mida = int(entrada)

            # Crear una matriu numpy de zeros amb la mida especificada
            imatge = np.zeros((mida, mida), dtype=np.uint8)

            # Afegir la matriu numpy a la llista
            imatges.append(imatge)

        except ValueError:
            print("Si us plau, introduïu una mida adequada.")

    return imatges

def crea_Forma(imatges):
    # Definir un diccionari per mapejar les opcions de forma a funcions
    opcions_forma = {'l': crea_linia, 'r': crea_rectangle, 'c': crea_cercle}

    for i, imatge in enumerate(imatges):
        print(f"\033[93mTreballant amb imatge de mida {imatge.shape[0]}x{imatge.shape[1]}\033[m")

        # Demanar a l'usuari la forma desitjada o "f" per finalitzar
        forma = input(f"Introduïu la forma per a la imatge {i + 1} (línia 'l', rectangle 'r', cercle 'c', o 'f' per finalitzar): ")

        if forma.lower() == 'f':
            continue  # Continuar amb la pròxima imatge si s'entra "f"

        if forma in opcions_forma:
            # Cridar la funció corresponent per crear la forma i passar-li la imatge com a primer argument
            opcions_forma[forma](imatge)
        else:
            print("Forma no vàlida. Les opcions són 'l', 'r', 'c'.")

    print("Procés finalitzat.")

def crea_linia(imatge):
    # Demanar a l'usuari les coordenades inicial i final de la línia
    x1, y1, x2, y2 = map(int, input("Introduïu les coordenades inicial (x1 y1) i final (x2 y2) de la línia separades per espais: ").split())

    # Demanar a l'usuari el color de la línia (0 per negre, 255 per blanc)
    color = int(input("Introduïu el color de la línia (0 per negre, 255 per blanc): "))

    # Dibuixar la línia a la imatge
    cv2.line(imatge, (x1, y1), (x2, y2), color, 1)

def crea_rectangle(imatge):
    # Demanar a l'usuari les coordenades de la cantonada superior esquerra (x1, y1)
    x1, y1 = map(int, input("Introduïu les coordenades de la cantonada superior esquerra (x1 y1): ").split())

    # Demanar a l'usuari les coordenades de la cantonada inferior dreta (x2, y2)
    x2, y2 = map(int, input("Introduïu les coordenades de la cantonada inferior dreta (x2 y2): ").split())

    # Demanar a l'usuari el color del rectangle (0 per negre, 255 per blanc)
    color = int(input("Introduïu el color del rectangle (0 per negre, 255 per blanc): "))

    # Dibuixar el rectangle a la imatge
    cv2.rectangle(imatge, (x1, y1), (x2, y2), color, 1)

def crea_cercle(imatge):
    # Demanar a l'usuari les coordenades del centre del cercle (x, y)
    x, y = map(int, input("Introduïu les coordenades del centre del cercle (x y): ").split())

    # Demanar a l'usuari el radi del cercle
    radi = int(input("Introduïu el radi del cercle: "))

    # Demanar a l'usuari el color del cercle (0 per negre, 255 per blanc)
    color = int(input("Introduïu el color del cercle (0 per negre, 255 per blanc): "))

    # Dibuixar el cercle a la imatge
    cv2.circle(imatge, (x, y), radi, color, 1)

def aplica_Gauss(imatges):
    # Crear una llista per emmagatzemar les imatges gaussianes
    gauss_imatges = []
    gauss_sigmas = []  # Nova llista per emmagatzemar els sigmes

    for i, imatge in enumerate(imatges):
        print(f"\033[93mTreballant amb imatge de mida {imatge.shape[0]}x{imatge.shape[1]}\033[m")

        while True:
            try:
                sigma = float(input(f"Introduïu el valor de sigma per a la imatge {i + 1}: "))
                if sigma <= 0:
                    print("Sigma ha de ser un valor positiu.")
                else:
                    break
            except ValueError:
                print("Si us plau, introduïu un valor numèric vàlid per a sigma.")

        x, y = np.meshgrid(np.linspace(-1, 1, imatge.shape[1]), np.linspace(-1, 1, imatge.shape[0]))
        #A = 1 / (2 * np.pi * sigma**2)
        #gaussian_image = A * np.exp(-(((x**2 + y**2)) / (2.0 * sigma**2)))
        gaussian_image = 1 - np.abs(x)
        gaussian_image = gaussian_image / np.max(gaussian_image)
        gaussian_image = (gaussian_image * 255).astype(np.uint8)

        # Afegir la imatge gaussiana a la llista gauss_imatges
        gauss_imatges.append(gaussian_image)
        # Afegir el sigma corresponent a la llista gauss_sigmas
        gauss_sigmas.append(sigma)

    return gauss_imatges, gauss_sigmas

def aplica_Sobel(imatges):
    # Crear una llista per emmagatzemar les imatges de Sobel
    sobel_imatges = []
    ordenes_kernels = []  # Nova llista per emmagatzemar les tuples d'ordres i kernel

    for i, imatge in enumerate(imatges):
        print(f"\033[93mTreballant amb imatge de mida {imatge.shape[0]}x{imatge.shape[1]}\033[m")

        while True:
            try:
                orden_x = int(input(f"Introduïu l'ordre de derivació en X per a la imatge {i + 1}: "))
                orden_y = int(input(f"Introduïu l'ordre de derivació en Y per a la imatge {i + 1}: "))
                if orden_x < 0 or orden_y < 0:
                    print("L'ordre de derivació ha de ser un valor positiu o zero.")
                else:
                    break
            except ValueError:
                print("Si us plau, introduïu un valor numèric vàlid per a l'ordre de derivació.")

        while True:
            try:
                tamany_kernel = int(input(f"Introduïu la mida del kernel (mínim 3, màxim 31) per a la imatge {i + 1}: "))
                if tamany_kernel < 3 or tamany_kernel > 31:
                    print("La mida del kernel ha de ser entre 3 i 31.")
                else:
                    break
            except ValueError:
                print("Si us plau, introduïu un valor numèric vàlid per a la mida del kernel.")

        sobel_imatge = cv2.Sobel(imatge, cv2.CV_32F, orden_x, orden_y, ksize=tamany_kernel, scale=10, borderType=cv2.BORDER_REFLECT)
        sobel_imatge = ((sobel_imatge / (np.max(np.abs(sobel_imatge))+0.00000001) + 1) * 127.5).astype(np.uint8)

        # Afegir la tupla d'ordres i kernel a la llista
        ordenes_kernels.append((orden_x, orden_y, tamany_kernel))

        # Afegir la imatge de Sobel a la llista
        sobel_imatges.append(sobel_imatge)

    return sobel_imatges, ordenes_kernels

def mostra_Imatge(imatges, gauss_imatges, gauss_sigmas, sobel_imatges, ordenes_kernels):
    num_imatges = len(imatges)

    for i in range(num_imatges):
        mida = imatges[i].shape[0]  # Obtenir la mida de la imatge
        sigma = gauss_sigmas[i]
        orden_x, orden_y, kernel = ordenes_kernels[i]

        # Crear una figura per mostrar les imatges
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Mostrar la imatge després d'aplicar una forma
        axs[0].imshow(imatges[i], cmap='gray')
        axs[0].set_title(f'Original - {mida}x{mida}')
        axs[0].axis('off')

        # Mostrar la imatge Gauss
        axs[1].imshow(gauss_imatges[i], cmap='gray')
        axs[1].set_title(f'Gauss - Sigma: {sigma}')
        axs[1].axis('off')

        # Mostrar la imatge Sobel
        axs[2].imshow(sobel_imatges[i], cmap='gray')
        axs[2].set_title(f'Sobel - Ordre X: {orden_x}, Ordre Y: {orden_y}, Kernel: {kernel}')
        axs[2].axis('off')

        # Mostrar la figura
        plt.show()

def main():
    # Crear les imatges
    imatges = crea_Imatge()

    # Crear formes a les imatges
    crea_Forma(imatges)

    # Aplicar el filtre Gaussià
    gauss_imatges, gauss_sigmas = aplica_Gauss(imatges)

    # Aplicar el filtre de Sobel
    sobel_imatges, ordenes_kernels = aplica_Sobel(gauss_imatges)

    # Mostrar les imatges amb els títols adaptats
    mostra_Imatge(imatges, gauss_imatges, gauss_sigmas, sobel_imatges, ordenes_kernels)

    print("\033[92mProcés finalitzat correctament!\033[m")

if __name__ == "__main__":
    main()
