import cv2
import numpy as np


def escribir_texto_con_sombra(
    imagen,
    texto,
    posicion,
    escala,
    color_texto=(255, 255, 255),
    color_sombra=(0, 0, 0),
    grosor=2
):
    x, y = posicion

    cv2.putText(
        imagen,
        texto,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        escala,
        color_sombra,
        grosor + 1
    )

    cv2.putText(
        imagen,
        texto,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        escala,
        color_texto,
        grosor
    )


def escribir_texto_centrado(
    imagen,
    texto,
    centro_x,
    centro_y,
    escala,
    color=(30, 30, 30),
    grosor=2
):
    tamanio_texto, _ = cv2.getTextSize(
        texto,
        cv2.FONT_HERSHEY_SIMPLEX,
        escala,
        grosor
    )

    ancho_texto, alto_texto = tamanio_texto

    x = int(centro_x - ancho_texto / 2)
    y = int(centro_y + alto_texto / 2)

    cv2.putText(
        imagen,
        texto,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        escala,
        color,
        grosor
    )


def obtener_limites_sector(ancho, alto, filas, columnas, fila_sector, columna_sector):
    """
    Calcula los límites en píxeles de un sector.
    fila_sector y columna_sector usan numeración desde 1.
    """

    x1 = int((columna_sector - 1) * ancho / columnas)
    x2 = int(columna_sector * ancho / columnas)

    y1 = int((fila_sector - 1) * alto / filas)
    y2 = int(fila_sector * alto / filas)

    return x1, y1, x2, y2


def filtrar_detecciones_en_sector(detecciones, x1_sector, y1_sector, x2_sector, y2_sector):
    """
    Filtra detecciones cuyo centro cae dentro del sector.
    Además transforma sus coordenadas globales a coordenadas locales del recorte.
    """

    detecciones_sector = []

    for det in detecciones:
        x_centro = (det["x1"] + det["x2"]) / 2
        y_centro = (det["y1"] + det["y2"]) / 2

        if x1_sector <= x_centro <= x2_sector and y1_sector <= y_centro <= y2_sector:
            x1_local = max(det["x1"] - x1_sector, 0)
            y1_local = max(det["y1"] - y1_sector, 0)
            x2_local = min(det["x2"] - x1_sector, x2_sector - x1_sector)
            y2_local = min(det["y2"] - y1_sector, y2_sector - y1_sector)

            detecciones_sector.append({
                "x1": x1_local,
                "y1": y1_local,
                "x2": x2_local,
                "y2": y2_local,
                "confianza": det["confianza"]
            })

    return detecciones_sector


def redimensionar_imagen_y_detecciones(imagen, detecciones, max_lado=1800):
    """
    Redimensiona una imagen para mostrarla mejor en Streamlit.
    También escala las cajas de detección.
    """

    alto, ancho = imagen.shape[:2]
    escala = min(max_lado / ancho, max_lado / alto, 1.0)

    if escala >= 1.0:
        return imagen, detecciones

    nuevo_ancho = int(ancho * escala)
    nuevo_alto = int(alto * escala)

    imagen_redimensionada = cv2.resize(
        imagen,
        (nuevo_ancho, nuevo_alto),
        interpolation=cv2.INTER_AREA
    )

    detecciones_redimensionadas = []

    for det in detecciones:
        detecciones_redimensionadas.append({
            "x1": det["x1"] * escala,
            "y1": det["y1"] * escala,
            "x2": det["x2"] * escala,
            "y2": det["y2"] * escala,
            "confianza": det["confianza"]
        })

    return imagen_redimensionada, detecciones_redimensionadas


def agregar_grilla_con_ejes(
    imagen: np.ndarray,
    filas: int,
    columnas: int,
    mostrar_ejes: bool = True,
    mostrar_numeros: bool = False,
    matriz=None,
    sector_destacado=None
):
    alto, ancho = imagen.shape[:2]

    ancho_celda = ancho / columnas
    alto_celda = alto / filas

    margen_izquierdo = 75 if mostrar_ejes else 0
    margen_superior = 60 if mostrar_ejes else 0
    margen_derecho = 20 if mostrar_ejes else 0
    margen_inferior = 20 if mostrar_ejes else 0

    alto_canvas = alto + margen_superior + margen_inferior
    ancho_canvas = ancho + margen_izquierdo + margen_derecho

    canvas = np.full(
        (alto_canvas, ancho_canvas, 3),
        245,
        dtype=np.uint8
    )

    x_offset = margen_izquierdo
    y_offset = margen_superior

    canvas[
        y_offset:y_offset + alto,
        x_offset:x_offset + ancho
    ] = imagen.copy()

    if mostrar_ejes:
        escala_ejes = 0.55

        for j in range(columnas):
            centro_x = x_offset + int((j + 0.5) * ancho_celda)
            centro_y = int(margen_superior / 2)

            escribir_texto_centrado(
                canvas,
                f"C{j + 1}",
                centro_x,
                centro_y,
                escala_ejes,
                color=(30, 30, 30),
                grosor=2
            )

        for i in range(filas):
            centro_x = int(margen_izquierdo / 2)
            centro_y = y_offset + int((i + 0.5) * alto_celda)

            escribir_texto_centrado(
                canvas,
                f"F{i + 1}",
                centro_x,
                centro_y,
                escala_ejes,
                color=(30, 30, 30),
                grosor=2
            )

    for i in range(filas):
        for j in range(columnas):
            x1 = x_offset + int(j * ancho_celda)
            y1 = y_offset + int(i * alto_celda)
            x2 = x_offset + int((j + 1) * ancho_celda)
            y2 = y_offset + int((i + 1) * alto_celda)

            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )

            if mostrar_numeros and matriz is not None:
                cantidad = int(matriz[i, j])

                if cantidad > 0:
                    tam_celda = min(ancho_celda, alto_celda)
                    escala_cantidad = max(0.55, min(0.90, tam_celda / 150))

                    escribir_texto_con_sombra(
                        canvas,
                        str(cantidad),
                        (x1 + 10, y1 + 34),
                        escala_cantidad,
                        color_texto=(255, 255, 255),
                        color_sombra=(0, 0, 0),
                        grosor=2
                    )

    if sector_destacado is not None:
        fila_destacada, columna_destacada = sector_destacado

        i = fila_destacada - 1
        j = columna_destacada - 1

        x1 = x_offset + int(j * ancho_celda)
        y1 = y_offset + int(i * alto_celda)
        x2 = x_offset + int((j + 1) * ancho_celda)
        y2 = y_offset + int((i + 1) * alto_celda)

        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            (0, 0, 0),
            5
        )

    cv2.rectangle(
        canvas,
        (x_offset, y_offset),
        (x_offset + ancho, y_offset + alto),
        (30, 30, 30),
        3
    )

    return canvas


def dibujar_detecciones(
    imagen: np.ndarray,
    detecciones,
    mostrar_etiquetas: bool = False,
    grosor: int = 2,
    filas: int = None,
    columnas: int = None,
    mostrar_grilla: bool = True,
    sector_destacado=None
):
    imagen_dibujada = imagen.copy()

    for det in detecciones:
        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])
        conf = det["confianza"]

        color = (0, 90, 255)

        cv2.rectangle(
            imagen_dibujada,
            (x1, y1),
            (x2, y2),
            color,
            grosor
        )

        if mostrar_etiquetas:
            texto = f"maleza {conf:.2f}"

            cv2.putText(
                imagen_dibujada,
                texto,
                (x1, max(y1 - 6, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1
            )

    if mostrar_grilla and filas is not None and columnas is not None:
        imagen_dibujada = agregar_grilla_con_ejes(
            imagen_dibujada,
            filas=filas,
            columnas=columnas,
            mostrar_ejes=True,
            mostrar_numeros=False,
            sector_destacado=sector_destacado
        )

    return imagen_dibujada


def obtener_color_por_nivel(cantidad, umbral_bajo, umbral_medio):
    if cantidad == 0:
        return None

    if cantidad <= umbral_bajo:
        return (56, 161, 105)

    if cantidad <= umbral_medio:
        return (214, 158, 46)

    return (229, 62, 62)


def dibujar_mapa_infestacion(
    imagen: np.ndarray,
    matriz,
    opacidad: float = 0.35,
    mostrar_nombres_sector: bool = True,
    umbral_bajo: int = 2,
    umbral_medio: int = 5,
    sector_destacado=None
):
    alto, ancho = imagen.shape[:2]
    filas, columnas = matriz.shape

    ancho_celda = ancho / columnas
    alto_celda = alto / filas

    imagen_base = imagen.copy()

    for i in range(filas):
        for j in range(columnas):
            cantidad = int(matriz[i, j])

            x1 = int(j * ancho_celda)
            y1 = int(i * alto_celda)
            x2 = int((j + 1) * ancho_celda)
            y2 = int((i + 1) * alto_celda)

            color = obtener_color_por_nivel(
                cantidad,
                umbral_bajo,
                umbral_medio
            )

            if color is not None:
                overlay = imagen_base.copy()

                cv2.rectangle(
                    overlay,
                    (x1, y1),
                    (x2, y2),
                    color,
                    -1
                )

                imagen_base = cv2.addWeighted(
                    overlay,
                    opacidad,
                    imagen_base,
                    1 - opacidad,
                    0
                )

    imagen_mapa = agregar_grilla_con_ejes(
        imagen_base,
        filas=filas,
        columnas=columnas,
        mostrar_ejes=mostrar_nombres_sector,
        mostrar_numeros=True,
        matriz=matriz,
        sector_destacado=sector_destacado
    )

    return imagen_mapa


def crear_zoom_sector_imagen(
    imagen: np.ndarray,
    detecciones,
    fila_sector: int,
    columna_sector: int,
    filas: int,
    columnas: int,
    mostrar_etiquetas: bool = True,
    max_lado: int = 1800
):
    """
    Crea un zoom del sector más afectado para imágenes normales, mosaicos o PNG/JPG.
    """

    alto, ancho = imagen.shape[:2]

    x1, y1, x2, y2 = obtener_limites_sector(
        ancho=ancho,
        alto=alto,
        filas=filas,
        columnas=columnas,
        fila_sector=fila_sector,
        columna_sector=columna_sector
    )

    recorte = imagen[y1:y2, x1:x2].copy()

    detecciones_sector = filtrar_detecciones_en_sector(
        detecciones,
        x1_sector=x1,
        y1_sector=y1,
        x2_sector=x2,
        y2_sector=y2
    )

    recorte, detecciones_sector = redimensionar_imagen_y_detecciones(
        recorte,
        detecciones_sector,
        max_lado=max_lado
    )

    zoom = dibujar_detecciones(
        recorte,
        detecciones_sector,
        mostrar_etiquetas=mostrar_etiquetas,
        grosor=3,
        mostrar_grilla=False
    )

    escribir_texto_con_sombra(
        zoom,
        f"Zoom sector F{fila_sector}-C{columna_sector} | {len(detecciones_sector)} detecciones",
        (20, 40),
        0.8,
        color_texto=(255, 255, 255),
        color_sombra=(0, 0, 0),
        grosor=2
    )

    return zoom