from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling

from src.detector import detectar_malezas_en_recortes


def normalizar_a_uint8(arr):
    """
    Convierte una lectura de rasterio a imagen RGB uint8.

    Versión estable:
    - evita oscurecer ortomosaicos que ya vienen bien;
    - evita cambiar demasiado los colores;
    - si la imagen viene oscura, ajusta principalmente el brillo;
    - mantiene como blanco las zonas sin datos.
    """

    arr = np.nan_to_num(arr)

    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)

    if arr.shape[0] > 3:
        arr = arr[:3]

    # Pasar de bandas, alto, ancho a alto, ancho, bandas
    img = np.transpose(arr, (1, 2, 0)).astype(np.float32)

    # Detectar zonas sin datos
    fondo_negro = np.all(img <= 3, axis=2)
    fondo_blanco = np.all(img >= 252, axis=2)
    mascara_valida = ~(fondo_negro | fondo_blanco)

    # Si viene en rango mayor que 255, llevarlo a 0-255 usando percentiles conjuntos
    if img.max() > 255:
        valores = img[mascara_valida]

        if valores.size > 100:
            p2, p98 = np.percentile(valores, (2, 98))
        else:
            p2, p98 = np.percentile(img, (2, 98))

        if p98 - p2 > 1e-6:
            img = (img - p2) / (p98 - p2)
            img = np.clip(img, 0, 1)
            img = img * 255
        else:
            img = np.clip(img, 0, 255)

    img = np.clip(img, 0, 255).astype(np.uint8)

    # Calcular brillo de píxeles válidos
    if np.any(mascara_valida):
        gris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        brillo_mediano = np.median(gris[mascara_valida])
    else:
        brillo_mediano = np.median(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

    # Solo aclarar si realmente está demasiado oscuro
    if brillo_mediano < 45:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        valores_v = v[mascara_valida]

        if valores_v.size > 100:
            p2, p98 = np.percentile(valores_v, (2, 98))
        else:
            p2, p98 = np.percentile(v, (2, 98))

        if p98 - p2 > 1e-6:
            v_float = v.astype(np.float32)
            v_float = (v_float - p2) / (p98 - p2)
            v_float = np.clip(v_float, 0, 1)

            # Aclarar sin destruir tanto los colores
            gamma = 0.75
            v_float = np.power(v_float, gamma)
            v_float = np.clip(v_float * 1.15, 0, 1)

            v = (v_float * 255).astype(np.uint8)

            hsv_corregido = cv2.merge([h, s, v])
            img = cv2.cvtColor(hsv_corregido, cv2.COLOR_HSV2RGB)

    # Mantener zonas sin datos como blanco
    img[fondo_negro] = [255, 255, 255]
    img[fondo_blanco] = [255, 255, 255]

    return img
def obtener_resolucion_cm_pixel(ruta_geotiff):
    """
    Intenta obtener la resolución espacial del GeoTIFF.

    Si el CRS está proyectado en metros, retorna cm/píxel.
    Si no se puede determinar, retorna None.
    """

    with rasterio.open(ruta_geotiff) as src:
        if src.crs is None or not src.crs.is_projected:
            return None

        pixel_x = abs(src.transform.a)
        pixel_y = abs(src.transform.e)

        pixel_promedio_m = (pixel_x + pixel_y) / 2
        return pixel_promedio_m * 100


def generar_ventanas(ancho, alto, tam_recorte=640, solapamiento=100):
    """
    Genera ventanas para leer un GeoTIFF por partes.
    """

    paso = tam_recorte - solapamiento

    ventanas = []
    posiciones_usadas = set()

    for y in range(0, alto, paso):
        for x in range(0, ancho, paso):
            x_fin = min(x + tam_recorte, ancho)
            y_fin = min(y + tam_recorte, alto)

            x_ini = max(0, x_fin - tam_recorte)
            y_ini = max(0, y_fin - tam_recorte)

            ancho_ventana = x_fin - x_ini
            alto_ventana = y_fin - y_ini

            if (x_ini, y_ini) in posiciones_usadas:
                continue

            posiciones_usadas.add((x_ini, y_ini))

            ventana = Window(
                col_off=x_ini,
                row_off=y_ini,
                width=ancho_ventana,
                height=alto_ventana
            )

            ventanas.append((ventana, (x_ini, y_ini)))

    return ventanas


def seleccionar_ventanas_distribuidas(ventanas, max_ventanas):
    """
    Selecciona ventanas distribuidas a lo largo de todo el ortomosaico.

    Antes se usaban solo las primeras ventanas, lo que podía analizar solo
    una esquina del ortomosaico. Ahora se toman muestras repartidas.
    """

    if max_ventanas is None or max_ventanas <= 0:
        return ventanas

    if len(ventanas) <= max_ventanas:
        return ventanas

    indices = np.linspace(
        0,
        len(ventanas) - 1,
        max_ventanas,
        dtype=int
    )

    ventanas_seleccionadas = [ventanas[i] for i in indices]

    return ventanas_seleccionadas


def leer_ventana_rgb(src, ventana):
    """
    Lee una ventana del GeoTIFF y la convierte a RGB uint8.
    """

    if src.count >= 3:
        arr = src.read([1, 2, 3], window=ventana)
    else:
        arr = src.read([1], window=ventana)

    imagen_rgb = normalizar_a_uint8(arr)

    return imagen_rgb


def crear_preview_geotiff(ruta_geotiff, max_lado=3000):
    """
    Crea una vista previa reducida del GeoTIFF completo.

    Esto se usa solo para visualizar detecciones y mapa.
    No se usa para detectar, porque la detección se hace por ventanas
    sobre el GeoTIFF completo.
    """

    with rasterio.open(ruta_geotiff) as src:
        ancho = src.width
        alto = src.height

        escala = min(max_lado / ancho, max_lado / alto, 1.0)

        ancho_preview = max(1, int(ancho * escala))
        alto_preview = max(1, int(alto * escala))

        if src.count >= 3:
            arr = src.read(
                [1, 2, 3],
                out_shape=(3, alto_preview, ancho_preview),
                resampling=Resampling.bilinear
            )
        else:
            arr = src.read(
                [1],
                out_shape=(1, alto_preview, ancho_preview),
                resampling=Resampling.bilinear
            )

    preview = normalizar_a_uint8(arr)

    escala_x = ancho_preview / ancho
    escala_y = alto_preview / alto

    return preview, escala_x, escala_y


def escalar_detecciones(detecciones, escala_x, escala_y):
    """
    Escala las coordenadas de detecciones del ortomosaico completo
    hacia la vista previa reducida.
    """

    detecciones_escaladas = []

    for det in detecciones:
        detecciones_escaladas.append({
            "x1": det["x1"] * escala_x,
            "y1": det["y1"] * escala_y,
            "x2": det["x2"] * escala_x,
            "y2": det["y2"] * escala_y,
            "confianza": det["confianza"]
        })

    return detecciones_escaladas


def procesar_geotiff_por_ventanas(
    ruta_geotiff,
    modelo,
    confianza=0.15,
    tam_recorte=640,
    solapamiento=100,
    max_ventanas=0,
    progreso_callback=None
):
    """
    Procesa un GeoTIFF completo por ventanas.

    max_ventanas:
    - 0 significa procesar todas las ventanas.
    - un número mayor a 0 procesa una muestra distribuida por el ortomosaico.
    """

    ruta_geotiff = Path(ruta_geotiff)

    detecciones_globales = []

    with rasterio.open(ruta_geotiff) as src:
        ancho = src.width
        alto = src.height

        ventanas = generar_ventanas(
            ancho=ancho,
            alto=alto,
            tam_recorte=tam_recorte,
            solapamiento=solapamiento
        )

        total_ventanas = len(ventanas)

        ventanas = seleccionar_ventanas_distribuidas(
            ventanas,
            max_ventanas=max_ventanas
        )

        total_a_procesar = len(ventanas)

        for idx, (ventana, posicion) in enumerate(ventanas):
            recorte = leer_ventana_rgb(src, ventana)

            detecciones = detectar_malezas_en_recortes(
                modelo,
                [recorte],
                [posicion],
                confianza=confianza
            )

            detecciones_globales.extend(detecciones)

            if progreso_callback is not None:
                progreso_callback(idx + 1, total_a_procesar, total_ventanas)

    return detecciones_globales, ancho, alto, total_a_procesar, total_ventanas

def crear_zoom_sector_geotiff(
    ruta_geotiff,
    detecciones,
    ancho_total,
    alto_total,
    filas,
    columnas,
    fila_sector,
    columna_sector,
    mostrar_etiquetas=True,
    max_lado=1800
):
    """
    Crea un zoom del sector más afectado leyendo solo esa zona del GeoTIFF.
    No carga el ortomosaico completo en memoria.
    """

    from src.visualizacion import (
        dibujar_detecciones,
        escribir_texto_con_sombra,
        filtrar_detecciones_en_sector
    )

    x1_sector = int((columna_sector - 1) * ancho_total / columnas)
    x2_sector = int(columna_sector * ancho_total / columnas)

    y1_sector = int((fila_sector - 1) * alto_total / filas)
    y2_sector = int(fila_sector * alto_total / filas)

    ancho_sector = x2_sector - x1_sector
    alto_sector = y2_sector - y1_sector

    escala = min(
        max_lado / ancho_sector,
        max_lado / alto_sector,
        1.0
    )

    ancho_salida = max(1, int(ancho_sector * escala))
    alto_salida = max(1, int(alto_sector * escala))

    ventana = Window(
        col_off=x1_sector,
        row_off=y1_sector,
        width=ancho_sector,
        height=alto_sector
    )

    with rasterio.open(ruta_geotiff) as src:
        if src.count >= 3:
            arr = src.read(
                [1, 2, 3],
                window=ventana,
                out_shape=(3, alto_salida, ancho_salida),
                resampling=Resampling.bilinear
            )
        else:
            arr = src.read(
                [1],
                window=ventana,
                out_shape=(1, alto_salida, ancho_salida),
                resampling=Resampling.bilinear
            )

    recorte = normalizar_a_uint8(arr)

    detecciones_sector = filtrar_detecciones_en_sector(
        detecciones,
        x1_sector=x1_sector,
        y1_sector=y1_sector,
        x2_sector=x2_sector,
        y2_sector=y2_sector
    )

    detecciones_escaladas = []

    for det in detecciones_sector:
        detecciones_escaladas.append({
            "x1": det["x1"] * escala,
            "y1": det["y1"] * escala,
            "x2": det["x2"] * escala,
            "y2": det["y2"] * escala,
            "confianza": det["confianza"]
        })

    zoom = dibujar_detecciones(
        recorte,
        detecciones_escaladas,
        mostrar_etiquetas=mostrar_etiquetas,
        grosor=3,
        mostrar_grilla=False
    )

    escribir_texto_con_sombra(
        zoom,
        f"Zoom sector F{fila_sector}-C{columna_sector} | {len(detecciones_escaladas)} detecciones",
        (20, 40),
        0.8,
        color_texto=(255, 255, 255),
        color_sombra=(0, 0, 0),
        grosor=2
    )

    return zoom