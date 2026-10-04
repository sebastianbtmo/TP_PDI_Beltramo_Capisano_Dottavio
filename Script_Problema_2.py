import cv2
import numpy as np
import csv
import os

# Parámetros
TH_BIN        = 128     # Umbral de binarización: img < TH_BIN --> "tinta" (texto y líneas)
TH_LINEA      = 0.6     # Una fila/columna es una línea de la tabla si tiene más de TH_LINEA*max píxeles de tinta
TH_AREA       = 2       # Se descartan componentes conectadas con área menor (ruido de un solo píxel)
FACTOR_ESPACIO = 0.6    # Hay un espacio entre dos caracteres si la separación es > FACTOR_ESPACIO * alto típico de caracter

CAMPOS = ['Legajo', 'Nombre y apellido', 'Parcial 1', 'Parcial 2', 'Parcial 3', 'Condición Final']


# Detección de líneas de la tabla
def rangos_true(v):
    # Devuelve (inicio, fin) de cada tramo consecutivo de valores True del vector v
    d = np.diff(np.concatenate(([0], v.astype(int), [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def detectar_lineas(img):
    img_th = (img < TH_BIN).astype(np.uint8)
    img_cols = np.sum(img_th, 0)                    # Suma por columna --> líneas verticales
    img_rows = np.sum(img_th, 1)                    # Suma por fila    --> líneas horizontales
    lineas_v = rangos_true(img_cols > TH_LINEA * img_cols.max())
    lineas_h = rangos_true(img_rows > TH_LINEA * img_rows.max())
    return img_th, lineas_v, lineas_h


# Análisis de un campo (celda)
def analizar_celda(celda_th):
    # celda_th: sub-imagen binaria (1 = tinta) del interior de la celda
    # Devuelve la lista de bounding boxes (x, y, w, h) de los caracteres, ordenadas de izquierda a derecha
    n, labels, stats, _ = cv2.connectedComponentsWithStats(celda_th, connectivity=8)
    stats = stats[1:]                               # Elimino el fondo
    stats = stats[stats[:, -1] >= TH_AREA, :]       # Elimino componentes de área muy pequeña
    stats = stats[np.argsort(stats[:, 0]), :]       # Ordeno por posición horizontal
    return stats[:, :4]


def contar_palabras(cajas, alto_caracter):
    # Cantidad de palabras: se separan cuando el espacio entre caracteres consecutivos es grande
    if len(cajas) == 0:
        return 0
    separacion = cajas[1:, 0] - (cajas[:-1, 0] + cajas[:-1, 2])
    return 1 + int(np.sum(separacion > FACTOR_ESPACIO * alto_caracter))


# Validación de cada campo
def validar_legajo(n_car, n_pal):
    return n_car == 8 and n_pal == 1                # 8 caracteres, una única palabra

def validar_nombre(n_car, n_pal):
    return n_pal >= 2 and (n_car + n_pal - 1) <= 12 # Al menos 2 palabras, máx. 12 caracteres (contando espacios)

def validar_nota(n_car, n_pal):
    return n_pal == 1 and n_car in (1, 2)           # 1 o 2 caracteres consecutivos

def validar_condicion(n_car, n_pal):
    return n_car == 1                               # Un único caracter

VALIDADORES = [validar_legajo, validar_nombre, validar_nota, validar_nota, validar_nota, validar_condicion]


# Reconocimiento de la Condición Final (A / L / R)
def reconocer_condicion(celda_th, caja):
    # Se usa para clasificar un único caracter entre 'A', 'L' y 'R':
    #   - 'L' no tiene agujeros; 'A' y 'R' tienen uno.
    #   - 'A' es casi simétrica respecto de un eje vertical; 'R' no.
    x, y, w, h = caja
    glifo = celda_th[y:y+h, x:x+w]
    glifo_pad = cv2.copyMakeBorder(glifo, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    n_comp_fondo, _, _, _ = cv2.connectedComponentsWithStats(1 - glifo_pad, connectivity=4)
    n_agujeros = n_comp_fondo - 2                   # Se resta el fondo y la componente exterior
    if n_agujeros == 0:
        return 'L'
    espejo = glifo[:, ::-1]
    simetria = np.sum(glifo & espejo) / np.sum(glifo | espejo)
    return 'A' if simetria > 0.65 else 'R'


# Imagen de salida
def generar_imagen_salida(no_aprobados, titulo, alto_crop=40):
    # no_aprobados: lista de (crop del Nombre y Apellido, 'L' o 'R')
    # Cada crop se redimensiona a un alto fijo (las planillas tienen distinta resolución)
    crops = []
    for crop, cond in no_aprobados:
        factor = alto_crop / crop.shape[0]
        crops.append(cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC))

    ancho_txt, alto_fila, margen = 220, alto_crop + 20, 10
    ancho_crop = max([c.shape[1] for c in crops], default=200)
    ancho = margen*3 + ancho_crop + ancho_txt
    alto = 50 + max(len(crops), 1) * alto_fila + margen
    salida = np.full((alto, ancho, 3), 255, np.uint8)
    cv2.putText(salida, titulo, (margen, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    if len(crops) == 0:
        cv2.putText(salida, 'Sin registros', (margen, 50 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    colores = {'R': (0, 165, 255), 'L': (0, 0, 255)}        # BGR: naranja = recupera, rojo = libre
    textos = {'R': 'R - Recupera', 'L': 'L - Libre'}
    for i, (crop, (_, cond)) in enumerate(zip(crops, no_aprobados)):
        y0 = 50 + i * alto_fila
        h, w = crop.shape
        salida[y0+10:y0+10+h, margen:margen+w] = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        x_ind = margen*2 + ancho_crop
        cv2.rectangle(salida, (x_ind, y0 + 5), (x_ind + ancho_txt, y0 + alto_fila - 5), colores[cond], -1)
        cv2.putText(salida, textos[cond], (x_ind + 15, y0 + alto_fila//2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return salida


# Algoritmo principal
def validar_planilla(path_img, id_planilla):
    img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
    img_th, lineas_v, lineas_h = detectar_lineas(img)
    assert len(lineas_v) == 8 and len(lineas_h) == 22, f"No se pudo detectar la tabla en {path_img}"

    # Celdas: la fila i (1..20) está entre lineas_h[i] y lineas_h[i+1]; la columna j (1..6) entre lineas_v[j] y lineas_v[j+1]
    # (la columna 0 es "Nro." y la fila 0 es el encabezado)
    celdas = {}
    for i in range(1, 21):
        for j in range(1, 7):
            y0, y1 = lineas_h[i][1] + 1, lineas_h[i+1][0]
            x0, x1 = lineas_v[j][1] + 1, lineas_v[j+1][0]
            celdas[(i, j)] = (img_th[y0:y1, x0:x1], img[y0:y1, x0:x1])

    # Alto típico de un caracter (para decidir cuándo una separación es un espacio)
    alturas = [c[3] for (th, _) in celdas.values() for c in analizar_celda(th) if c[3] > 3]
    alto_caracter = np.median(alturas)

    resultados, no_aprobados = [], []
    print(f"\n===== {path_img} =====")
    for i in range(1, 21):
        ok = []
        print(f"Registro {i}:")
        for j, (campo, validar) in enumerate(zip(CAMPOS, VALIDADORES), start=1):
            cajas = analizar_celda(celdas[(i, j)][0])
            n_pal = contar_palabras(cajas, alto_caracter)
            ok.append(validar(len(cajas), n_pal))
            print(f"{campo}: {'OK' if ok[-1] else 'MAL'}")
        resultados.append(ok)

        # Registro correcto y alumno que no aprobó --> lo agrego a la imagen de salida
        if all(ok):
            th_c, _ = celdas[(i, 6)]
            cond = reconocer_condicion(th_c, analizar_celda(th_c)[0])
            if cond in ('L', 'R'):
                crop_nombre = celdas[(i, 2)][1]
                no_aprobados.append((crop_nombre, cond))

    return resultados, no_aprobados
    
    
# Aplico el algoritmo sobre las cuatro planillas
resultados_total = []
no_aprobados_total = []
for id_planilla in range(1, 5):
    resultados, no_aprobados = validar_planilla(f"grade_sheet_{id_planilla}.png", id_planilla)
    resultados_total += resultados
    no_aprobados_total += no_aprobados

# Imagen de salida
salida = generar_imagen_salida(no_aprobados_total, "Alumnos que no aprobaron")
cv2.imwrite("salida_no_aprobados.png", salida)

# CSV de salida
with open("validacion_planillas.csv", 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['ID', 'Legajo', 'Nombre y Apellido', 'Parcial 1', 'Parcial 2', 'Parcial 3', 'Condición Final'])
    for id_registro, ok in enumerate(resultados_total, start=1):
        w.writerow([id_registro] + ['OK' if o else 'MAL' for o in ok])
