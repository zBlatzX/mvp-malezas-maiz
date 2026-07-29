import numpy as np
import pandas as pd


def generar_matriz_infestacion(
    detecciones,
    ancho_imagen: int,
    alto_imagen: int,
    filas: int = 10,
    columnas: int = 10
):
    matriz = np.zeros((filas, columnas), dtype=int)

    for det in detecciones:
        x_centro = (det["x1"] + det["x2"]) / 2
        y_centro = (det["y1"] + det["y2"]) / 2

        columna = int((x_centro / ancho_imagen) * columnas)
        fila = int((y_centro / alto_imagen) * filas)

        columna = min(columna, columnas - 1)
        fila = min(fila, filas - 1)

        matriz[fila, columna] += 1

    return matriz


def clasificar_nivel(
    cantidad: int,
    umbral_bajo: int = 2,
    umbral_medio: int = 5
):
    if cantidad == 0:
        return "Sin presencia"
    elif cantidad <= umbral_bajo:
        return "Baja"
    elif cantidad <= umbral_medio:
        return "Media"
    else:
        return "Alta"


def generar_recomendacion(nivel: str, densidad_malezas_m2=None):
    """
    Genera una recomendación preliminar.

    La recomendación NO indica aplicar herbicida automáticamente.
    Solo orienta la priorización de revisión del sector.
    """

    if nivel == "Sin presencia":
        return "No requiere acción"

    if densidad_malezas_m2 is None:
        if nivel == "Alta":
            return "Priorizar revisión / posible tratamiento localizado"
        elif nivel == "Media":
            return "Revisar zona"
        else:
            return "Monitorear"

    if nivel == "Alta":
        return "Priorizar revisión / posible tratamiento localizado"

    if nivel == "Media":
        if densidad_malezas_m2 >= 1:
            return "Revisar con prioridad"
        return "Revisar zona"

    if nivel == "Baja":
        if densidad_malezas_m2 >= 1:
            return "Monitorear con atención"
        return "Monitorear"

    return "No requiere acción"


def obtener_prioridad(nivel: str):
    if nivel == "Alta":
        return 1
    elif nivel == "Media":
        return 2
    elif nivel == "Baja":
        return 3
    else:
        return 4


def crear_tabla_sectores(
    matriz,
    umbral_bajo: int = 2,
    umbral_medio: int = 5,
    area_sector_m2=None
):
    filas, columnas = matriz.shape
    datos = []

    for i in range(filas):
        for j in range(columnas):
            cantidad = int(matriz[i, j])

            nivel = clasificar_nivel(
                cantidad,
                umbral_bajo,
                umbral_medio
            )

            if area_sector_m2 is not None and area_sector_m2 > 0:
                densidad = cantidad / area_sector_m2
            else:
                densidad = None

            recomendacion = generar_recomendacion(
                nivel,
                densidad_malezas_m2=densidad
            )

            prioridad = obtener_prioridad(nivel)

            fila_datos = {
                "sector": f"F{i+1}-C{j+1}",
                "fila": i + 1,
                "columna": j + 1,
                "malezas_detectadas": cantidad,
                "nivel": nivel,
                "recomendacion": recomendacion,
                "prioridad": prioridad
            }

            if area_sector_m2 is not None:
                fila_datos["area_sector_m2"] = round(area_sector_m2, 2)

                if densidad is not None:
                    fila_datos["densidad_malezas_m2"] = round(densidad, 3)
                else:
                    fila_datos["densidad_malezas_m2"] = None

            datos.append(fila_datos)

    tabla = pd.DataFrame(datos)

    tabla = tabla.sort_values(
        by=["prioridad", "malezas_detectadas"],
        ascending=[True, False]
    )

    return tabla