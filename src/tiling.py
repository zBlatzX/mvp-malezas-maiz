import numpy as np


def dividir_imagen_en_recortes(
    imagen: np.ndarray,
    tam_recorte: int = 640,
    solapamiento: int = 100
):
    """
    Divide una imagen grande en recortes pequeños.

    Retorna:
    - recortes: lista de imágenes recortadas.
    - posiciones: lista con (x_inicio, y_inicio) de cada recorte.
    """

    alto, ancho = imagen.shape[:2]
    paso = tam_recorte - solapamiento

    recortes = []
    posiciones = []
    posiciones_usadas = set()

    for y in range(0, alto, paso):
        for x in range(0, ancho, paso):

            x_fin = min(x + tam_recorte, ancho)
            y_fin = min(y + tam_recorte, alto)

            x_ini = max(0, x_fin - tam_recorte)
            y_ini = max(0, y_fin - tam_recorte)

            if (x_ini, y_ini) in posiciones_usadas:
                continue

            posiciones_usadas.add((x_ini, y_ini))

            recorte = imagen[y_ini:y_fin, x_ini:x_fin]

            recortes.append(recorte)
            posiciones.append((x_ini, y_ini))

    return recortes, posiciones