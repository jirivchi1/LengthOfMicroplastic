import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize

def preprocess_image(image_path, save_path_prefix="output", pixels_to_microns=200/190):
    # Leer la imagenpython
    img = cv2.imread(image_path)
    img = img[::,500::]

    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mejora del contraste usando CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    # Aplicar un filtro Gaussiano para reducir el ruido
    blurred = cv2.GaussianBlur(clahe_img, (5, 5), 0)

    # Guardar la figura de preprocesamiento
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(gray, cmap='gray')
    plt.title('Original en Grises')

    plt.subplot(132)
    plt.imshow(clahe_img, cmap='gray')
    plt.title('Después de CLAHE')

    plt.subplot(133)
    plt.imshow(blurred, cmap='gray')
    plt.title('Después del Suavizado')
    plt.savefig(f"{save_path_prefix}_preprocessing.png")
    plt.show()

    # Detección de bordes con Canny
    edges = cv2.Canny(blurred, 20, 100)

    # Operaciones morfológicas
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Guardar la figura de bordes y morfología
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(edges, cmap='gray')
    plt.title('Bordes')

    plt.subplot(132)
    plt.imshow(closed, cmap='gray')
    plt.title('Después de Operaciones Morfológicas')

    plt.subplot(133)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original para comparación')
    plt.savefig(f"{save_path_prefix}_morphology.png")
    plt.show()

    # Skeletonización
    skeleton = skeletonize(closed > 0)

    # Calcular longitud en micras
    skeleton_pixels = np.sum(skeleton)
    length_microns = skeleton_pixels * pixels_to_microns

    # Mostrar y guardar los resultados de skeletonización
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('Original', fontsize=20)

    ax[1].imshow(skeleton, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title(f'Length: {length_microns:.1f} µm', fontsize=20)

    fig.tight_layout()
    plt.savefig(f"{save_path_prefix}_skeleton.png")
    plt.show()

    return length_microns

# Ejemplo de uso
def main():
    image_path = 'images/raw/Ai_F3C-2_1.tiff'  # Ruta de entrada
    save_path_prefix = 'reports/figures/result'  # Prefijo para guardar los resultados
    length_microns = preprocess_image(image_path, save_path_prefix)
    print(f"Longitud del esqueleto: {length_microns:.1f} µm")

if __name__ == "__main__":
    main()
