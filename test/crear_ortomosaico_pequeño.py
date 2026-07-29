from pathlib import Path
import os

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from PIL import Image


# Usa uno de los ortomosaicos que cargan bien
RUTA_ORIGINAL = r"D:\Vision Computacional\datasets\weedsgalore\weedsgalore-orthomosaic\2023-06-06_om.tif"

# Carpeta donde se guardarán los recortes cercanos a 200 MB
CARPETA_SALIDA = r"D:\Vision Computacional\datasets\weedsgalore\recortes_casi_200mb_06_06"

# Límite real que quieres respetar
LIMITE_MB = 200

# Dejamos margen para no pasarnos por diferencias de medición
OBJETIVO_MB = 195

# Tamaños que el script probará
TAM_MIN = 3000
TAM_MAX = 14000


def calcular_mb(ruta):
    return Path(ruta).stat().st_size / (1024 * 1024)


def crear_preview(datos, mascara, ruta_preview):
    """
    Crea una vista previa PNG solo para revisar visualmente.
    No modifica el GeoTIFF final.
    """

    datos = np.nan_to_num(datos)

    if datos.shape[0] == 1:
        datos = np.repeat(datos, 3, axis=0)

    if datos.shape[0] > 3:
        datos = datos[:3]

    img = np.transpose(datos, (1, 2, 0))

    if img.dtype == np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img = img.astype(np.float32)

        if mascara is not None and np.any(mascara):
            valores = img[mascara]
        else:
            valores = img.reshape(-1, img.shape[-1])

        if valores.size > 100:
            p2, p98 = np.percentile(valores, (2, 98))
        else:
            p2, p98 = np.percentile(img, (2, 98))

        if p98 - p2 > 1e-6:
            img = (img - p2) / (p98 - p2)
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    if mascara is not None:
        img[~mascara] = [255, 255, 255]

    Image.fromarray(img).save(ruta_preview)


def obtener_ventana_sector(src, filas, columnas, fila_sector, columna_sector, tam_salida):
    """
    Calcula una ventana cuadrada centrada en un sector F-C.
    """

    ancho = src.width
    alto = src.height

    x1_sector = int((columna_sector - 1) * ancho / columnas)
    x2_sector = int(columna_sector * ancho / columnas)

    y1_sector = int((fila_sector - 1) * alto / filas)
    y2_sector = int(fila_sector * alto / filas)

    centro_x = (x1_sector + x2_sector) // 2
    centro_y = (y1_sector + y2_sector) // 2

    mitad = tam_salida // 2

    x_inicio = max(centro_x - mitad, 0)
    y_inicio = max(centro_y - mitad, 0)

    x_fin = min(x_inicio + tam_salida, ancho)
    y_fin = min(y_inicio + tam_salida, alto)

    x_inicio = max(x_fin - tam_salida, 0)
    y_inicio = max(y_fin - tam_salida, 0)

    ancho_recorte = x_fin - x_inicio
    alto_recorte = y_fin - y_inicio

    return Window(
        col_off=x_inicio,
        row_off=y_inicio,
        width=ancho_recorte,
        height=alto_recorte
    )


def guardar_recorte(
    src,
    ventana,
    ruta_salida,
    crear_png=False,
    ruta_preview=None
):
    """
    Guarda un recorte GeoTIFF comprimido manteniendo las bandas originales.
    """

    if src.count >= 3:
        datos = src.read([1, 2, 3], window=ventana)
    else:
        datos = src.read([1], window=ventana)

    try:
        mascara = src.dataset_mask(window=ventana) > 0
    except Exception:
        mascara = np.ones((int(ventana.height), int(ventana.width)), dtype=bool)

    perfil = src.profile.copy()

    perfil.update({
        "driver": "GTiff",
        "height": int(ventana.height),
        "width": int(ventana.width),
        "count": datos.shape[0],
        "dtype": datos.dtype,
        "transform": window_transform(ventana, src.transform),

        # Compresión para quedar cerca de 200 MB
        "compress": "deflate",
        "zlevel": 9,
        "predictor": 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": None
    })

    if datos.shape[0] == 3:
        perfil["photometric"] = "RGB"

    with rasterio.open(ruta_salida, "w", **perfil) as dst:
        dst.write(datos)
        dst.write_mask((mascara * 255).astype(np.uint8))

    if crear_png and ruta_preview is not None:
        crear_preview(datos, mascara, ruta_preview)


def buscar_tamano_maximo_para_sector(
    ruta_original,
    carpeta_salida,
    filas,
    columnas,
    fila_sector,
    columna_sector,
    tam_min=TAM_MIN,
    tam_max=TAM_MAX,
    objetivo_mb=OBJETIVO_MB
):
    """
    Busca automáticamente el mayor tamaño posible sin superar OBJETIVO_MB.
    Usa búsqueda binaria probando distintos tamaños.
    """

    ruta_original = Path(ruta_original)
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    nombre_base = f"ortomosaico_F{fila_sector}_C{columna_sector}_{ruta_original.stem}"

    ruta_temp = carpeta_salida / f"temp_{nombre_base}.tif"

    mejor_tam = None
    mejor_mb = None

    with rasterio.open(ruta_original) as src:
        bajo = tam_min
        alto = tam_max

        print("")
        print("=" * 80)
        print(f"Buscando recorte para sector F{fila_sector}-C{columna_sector}")
        print(f"Objetivo: bajo {objetivo_mb} MB")
        print(f"Ortomosaico: {ruta_original.name}")
        print(f"Tamaño original: {src.width} x {src.height}")
        print("=" * 80)

        while bajo <= alto:
            medio = (bajo + alto) // 2

            ventana = obtener_ventana_sector(
                src=src,
                filas=filas,
                columnas=columnas,
                fila_sector=fila_sector,
                columna_sector=columna_sector,
                tam_salida=medio
            )

            if ruta_temp.exists():
                ruta_temp.unlink()

            guardar_recorte(
                src=src,
                ventana=ventana,
                ruta_salida=ruta_temp,
                crear_png=False
            )

            peso_mb = calcular_mb(ruta_temp)

            print(f"Probando {medio}px -> {peso_mb:.2f} MB")

            if peso_mb <= objetivo_mb:
                mejor_tam = medio
                mejor_mb = peso_mb
                bajo = medio + 250
            else:
                alto = medio - 250

        if mejor_tam is None:
            print("No se encontró un tamaño bajo el límite. Prueba bajando TAM_MIN.")
            if ruta_temp.exists():
                ruta_temp.unlink()
            return None

        ruta_final = carpeta_salida / f"{nombre_base}_{mejor_tam}px_{mejor_mb:.1f}MB.tif"
        ruta_preview = carpeta_salida / f"{nombre_base}_{mejor_tam}px_preview.png"

        ventana_final = obtener_ventana_sector(
            src=src,
            filas=filas,
            columnas=columnas,
            fila_sector=fila_sector,
            columna_sector=columna_sector,
            tam_salida=mejor_tam
        )

        guardar_recorte(
            src=src,
            ventana=ventana_final,
            ruta_salida=ruta_final,
            crear_png=True,
            ruta_preview=ruta_preview
        )

    if ruta_temp.exists():
        ruta_temp.unlink()

    peso_final = calcular_mb(ruta_final)

    print("")
    print("Recorte final creado:")
    print(ruta_final)
    print("Preview:")
    print(ruta_preview)
    print(f"Tamaño elegido: {mejor_tam}px")
    print(f"Peso final: {peso_final:.2f} MB")

    if peso_final <= LIMITE_MB:
        print("Estado: OK, bajo 200 MB")
    else:
        print("Estado: OJO, superó 200 MB. Baja OBJETIVO_MB a 190.")

    return ruta_final, ruta_preview, peso_final


if __name__ == "__main__":
    sectores_a_probar = [
        (8, 2),
        (8, 3),
        (7, 2),
        (7, 3),
        (6, 3),
        (6, 4),
        (5, 5),
    ]

    print("Creando recortes grandes bajo 200 MB desde:")
    print(RUTA_ORIGINAL)

    resultados = []

    for fila, columna in sectores_a_probar:
        resultado = buscar_tamano_maximo_para_sector(
            ruta_original=RUTA_ORIGINAL,
            carpeta_salida=CARPETA_SALIDA,
            filas=10,
            columnas=10,
            fila_sector=fila,
            columna_sector=columna,
            tam_min=TAM_MIN,
            tam_max=TAM_MAX,
            objetivo_mb=OBJETIVO_MB
        )

        if resultado is not None:
            resultados.append(resultado)

    print("")
    print("=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)

    for ruta_tif, ruta_preview, peso_mb in resultados:
        print(f"{peso_mb:.2f} MB -> {ruta_tif}")

    print("")
    print("Listo. Revisa los preview PNG y usa en la app el .tif que se vea mejor.")