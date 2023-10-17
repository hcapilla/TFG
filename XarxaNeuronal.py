import os
import torch
from PIL import Image as im
from torchvision import transforms
import numpy as np
from torch import nn
import torchvision
import pickle
import cv2
import glob

BATCH_SIZE = 100

def preproces_VGG16(imgs_hr):
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tnsr = preprocess(imgs_hr)
    return tnsr

def preproces_VGG16_2(imgs_hr):
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    tnsr = preprocess(imgs_hr)
    return tnsr

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def main():
    with open('Hernan.pkl', 'rb') as fp:
        diccionari = pickle.load(fp)

    folder_dir = 'VGG16_normal'
    device = "cpu"
    model = torchvision.models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(4096, 200)
    model.load_state_dict(torch.load(folder_dir, map_location=device))

    layers = []
    for l, layer in enumerate(model.features):
        if hasattr(layer, "weight"):
            layers.append("features." + str(l))

    carpeta_output = "prova/"
    patron_archivo = carpeta_output + "*.png"
    archivos_imagen = glob.glob(patron_archivo)

    resultados = []

    max_activation_global = float('-inf')
    neurona_interes_max = None

    for archivo_imagen in archivos_imagen:
        patch = cv2.imread(archivo_imagen)
        neurona_interes = 45

        maxima_activacio_neurona_interes = diccionari['features.3']['Max Activations'][neurona_interes]

        posicio_im = diccionari['features.3']["position for the image"][0]
        tamany_im = diccionari['features.3']['Image size']

        # Modifica l'imatge d'entrada a (224, 224, 3)
        diff = patch.shape[0] - tamany_im
        start = diff // 2
        end = start + tamany_im
        patch_cropped = patch[start:end, start:end, :]
        imatge = np.zeros((224, 224, 3))
        imatge[posicio_im:posicio_im+tamany_im, posicio_im:posicio_im+tamany_im, :] = patch_cropped


        imatge = im.fromarray(imatge.astype(np.uint8))
        imatge.show()
        input = preproces_VGG16(imatge)
        input = torch.unsqueeze(input, 0)

        model.eval()
        model.features[3].register_forward_hook(get_activation('my_activation'))
        output = model(input)

        posicio_act = diccionari['features.3']["Activation position"][0]
        max_activation = activation['my_activation'][0, neurona_interes, posicio_act, posicio_act]

        print(max_activation)

        nombre_imagen = os.path.splitext(os.path.basename(archivo_imagen))[0]
        resultado = f"\n{nombre_imagen}: {max_activation:.6f}"

        resultados.append(resultado)

        if max_activation > max_activation_global:
            max_activation_global = max_activation
            neurona_interes_max = neurona_interes

    # Guardar todos los resultados en el archivo de texto
    with open('resultados.txt', 'w') as archivo_resultados:
        for resultado in resultados:
            archivo_resultados.write(resultado)

if __name__ == '__main__':
    main()
