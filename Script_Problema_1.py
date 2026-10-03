import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ecualización local de histograma
def ecualizacion_local(img, M, N=None):
    if N is None:
        N = M
    if M % 2 == 0:
        M += 1
    if N % 2 == 0:
        N += 1
    pm, pn = M//2, N//2

    # Agrego bordes para poder centrar la ventana en los píxeles de los bordes de la imagen
    img_pad = cv2.copyMakeBorder(img, pm, pm, pn, pn, cv2.BORDER_REPLICATE)
    img_out = np.zeros_like(img)

    H, W = img.shape
    for i in range(H):
        for j in range(W):
            ventana = img_pad[i:i+M, j:j+N]                                  # Ventana centrada en (i,j)
            hist = cv2.calcHist([ventana], [0], None, [256], [0, 256])       # Histograma local
            cdf = hist.cumsum() / ventana.size                               # CDF normalizada
            img_out[i, j] = np.uint8(np.round(255 * cdf[img[i, j]]))         # Transformación del píxel central
    return img_out

# Análisis de la imagen con detalles ocultos y influencia del tamaño de ventana
img = cv2.imread("Imagen_con_detalles_escondidos.tif", cv2.IMREAD_GRAYSCALE)
tamanos = [3, 11, 25, 51, 101]
plt.figure()
ax1 = plt.subplot(231)
plt.imshow(img, cmap='gray', vmin=0, vmax=255), plt.title('Original')
for k, m in enumerate(tamanos):
    plt.subplot(2, 3, k+2, sharex=ax1, sharey=ax1)
    plt.imshow(ecualizacion_local(img, m), cmap='gray', vmin=0, vmax=255)
    plt.title(f'Ventana {m}x{m}')
plt.show()