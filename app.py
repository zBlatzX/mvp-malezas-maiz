import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from io import BytesIO
from copy import copy
import tempfile
import os

from src.detector import (
    cargar_modelo,
    detectar_malezas_en_recortes,
    eliminar_duplicados
)
from src.tiling import dividir_imagen_en_recortes
from src.infestacion import generar_matriz_infestacion, crear_tabla_sectores
from src.visualizacion import (
    dibujar_detecciones,
    dibujar_mapa_infestacion,
    crear_zoom_sector_imagen
)
from src.ortomosaico import (
    obtener_resolucion_cm_pixel,
    crear_preview_geotiff,
    escalar_detecciones,
    procesar_geotiff_por_ventanas,
    crear_zoom_sector_geotiff
)


RUTA_MODELO = "modelo/best_yolov8s.pt"


st.set_page_config(
    page_title="Detección de malezas en maíz",
    layout="wide"
)


@st.cache_resource
def obtener_modelo():
    return cargar_modelo(RUTA_MODELO)


def convertir_imagen_a_bytes(imagen_np):
    imagen_pil = Image.fromarray(imagen_np)
    buffer = BytesIO()
    imagen_pil.save(buffer, format="PNG")
    return buffer.getvalue()


def calcular_datos_espaciales(ancho, alto, filas, columnas, resolucion_cm_pixel):
    """
    Calcula área total, área por sector y área por píxel.

    La resolución espacial se ingresa en cm/píxel.
    """

    if resolucion_cm_pixel is None or resolucion_cm_pixel <= 0:
        return None, None, None

    resolucion_m_pixel = resolucion_cm_pixel / 100

    area_pixel_m2 = resolucion_m_pixel ** 2
    area_total_m2 = ancho * alto * area_pixel_m2
    area_sector_m2 = area_total_m2 / (filas * columnas)

    return area_pixel_m2, area_total_m2, area_sector_m2


def obtener_nivel_maximo(tabla):
    niveles = tabla["nivel"].tolist()

    if "Alta" in niveles:
        return "Alta"
    elif "Media" in niveles:
        return "Media"
    elif "Baja" in niveles:
        return "Baja"
    else:
        return "Sin presencia"


def filtrar_tabla(tabla, opcion_filtro):
    if opcion_filtro == "Todos los sectores":
        return tabla

    if opcion_filtro == "Solo sectores con malezas":
        return tabla[tabla["malezas_detectadas"] > 0]

    if opcion_filtro == "Solo sectores media y alta":
        return tabla[tabla["nivel"].isin(["Media", "Alta"])]

    if opcion_filtro == "Solo sectores alta":
        return tabla[tabla["nivel"] == "Alta"]

    return tabla


def limpiar_columnas_tabla(tabla):
    tabla = tabla.copy()

    columnas = {
        "sector": "Sector",
        "fila": "Fila",
        "columna": "Columna",
        "malezas_detectadas": "Malezas detectadas",
        "nivel": "Nivel de infestación",
        "area_sector_m2": "Área del sector (m²)",
        "densidad_malezas_m2": "Densidad (malezas/m²)",
        "recomendacion": "Recomendación"
    }

    tabla = tabla.rename(columns=columnas)

    if "prioridad" in tabla.columns:
        tabla = tabla.drop(columns=["prioridad"])

    orden_columnas = [
        "Sector",
        "Fila",
        "Columna",
        "Malezas detectadas",
        "Nivel de infestación",
        "Área del sector (m²)",
        "Densidad (malezas/m²)",
        "Recomendación"
    ]

    columnas_existentes = [
        col for col in orden_columnas
        if col in tabla.columns
    ]

    return tabla[columnas_existentes]


def crear_reporte_texto(
    nombre_imagen,
    tipo_entrada,
    ancho,
    alto,
    recortes,
    detecciones_antes,
    detecciones_finales,
    sectores_con_malezas,
    nivel_maximo,
    sector_max,
    tabla,
    umbral_bajo,
    umbral_medio,
    resolucion_cm_pixel,
    area_total_m2,
    area_sector_m2,
    densidad_general
):
    top_sectores = tabla[
        tabla["malezas_detectadas"] > 0
    ].head(10)

    lineas = []

    lineas.append("REPORTE DE ANÁLISIS DE MALEZAS")
    lineas.append("=" * 40)
    lineas.append("")
    lineas.append(f"Imagen analizada: {nombre_imagen}")
    lineas.append(f"Tipo de entrada: {tipo_entrada}")
    lineas.append(f"Tamaño de imagen: {ancho} x {alto} píxeles")
    lineas.append(f"Recortes / ventanas procesadas: {recortes}")
    lineas.append(f"Detecciones antes de eliminar duplicados: {detecciones_antes}")
    lineas.append(f"Detecciones finales: {detecciones_finales}")
    lineas.append(f"Sectores con presencia de malezas: {sectores_con_malezas}")
    lineas.append(f"Nivel máximo detectado: {nivel_maximo}")
    lineas.append(
        f"Sector más afectado: {sector_max['sector']} "
        f"({sector_max['malezas_detectadas']} detecciones)"
    )

    if resolucion_cm_pixel is not None and area_total_m2 is not None:
        lineas.append("")
        lineas.append("DATOS ESPACIALES APROXIMADOS")
        lineas.append("-" * 40)
        lineas.append(f"Resolución espacial: {resolucion_cm_pixel:.3f} cm/píxel")
        lineas.append(f"Área total aproximada: {area_total_m2:.2f} m²")
        lineas.append(f"Área aproximada por sector: {area_sector_m2:.2f} m²")
        lineas.append(f"Densidad general: {densidad_general:.3f} malezas/m²")

    lineas.append("")
    lineas.append("INTERPRETACIÓN DE NIVELES")
    lineas.append("-" * 40)
    lineas.append("0 detecciones: Sin presencia")
    lineas.append(f"1 a {umbral_bajo} detecciones: Baja")
    lineas.append(f"{umbral_bajo + 1} a {umbral_medio} detecciones: Media")
    lineas.append(f"Más de {umbral_medio} detecciones: Alta")
    lineas.append("")
    lineas.append("SECTORES PRIORITARIOS")
    lineas.append("-" * 40)

    if top_sectores.empty:
        lineas.append("No se detectaron sectores con presencia de malezas.")
    else:
        for _, fila in top_sectores.iterrows():
            linea = (
                f"{fila['sector']} | "
                f"{fila['malezas_detectadas']} malezas | "
                f"Nivel: {fila['nivel']} | "
            )

            if "densidad_malezas_m2" in fila and pd.notna(fila["densidad_malezas_m2"]):
                linea += f"Densidad: {fila['densidad_malezas_m2']} malezas/m² | "

            linea += f"Recomendación: {fila['recomendacion']}"

            lineas.append(linea)

    lineas.append("")
    lineas.append("Nota:")
    lineas.append(
        "Las recomendaciones son preliminares y sirven para priorizar "
        "revisión de sectores. No reemplazan una evaluación agronómica en terreno."
    )

    return "\n".join(lineas)


def crear_excel_resultados(resultados):
    buffer = BytesIO()

    resumen_general = []
    sectores_globales = []

    for resultado in resultados:
        fila_resumen = {
            "Imagen": resultado["nombre_imagen"],
            "Tipo de entrada": resultado["tipo_entrada"],
            "Ancho": resultado["ancho"],
            "Alto": resultado["alto"],
            "Recortes / ventanas procesadas": resultado["recortes"],
            "Detecciones finales": resultado["detecciones_finales"],
            "Sectores con presencia": resultado["sectores_con_malezas"],
            "Sector más afectado": resultado["sector_max"]["sector"],
            "Nivel máximo": resultado["nivel_maximo"]
        }

        if resultado["resolucion_cm_pixel"] is not None and resultado["area_total_m2"] is not None:
            fila_resumen["Resolución (cm/píxel)"] = round(resultado["resolucion_cm_pixel"], 3)
            fila_resumen["Área total (m²)"] = round(resultado["area_total_m2"], 2)
            fila_resumen["Área por sector (m²)"] = round(resultado["area_sector_m2"], 2)
            fila_resumen["Densidad general (malezas/m²)"] = round(resultado["densidad_general"], 3)

        resumen_general.append(fila_resumen)

        tabla_limpia = limpiar_columnas_tabla(resultado["tabla"])
        tabla_limpia.insert(0, "Imagen", resultado["nombre_imagen"])
        sectores_globales.append(tabla_limpia)

    df_resumen = pd.DataFrame(resumen_general)

    if sectores_globales:
        df_sectores = pd.concat(sectores_globales, ignore_index=True)
    else:
        df_sectores = pd.DataFrame()

    if not df_sectores.empty and "Malezas detectadas" in df_sectores.columns:
        df_prioritarios = df_sectores[
            df_sectores["Malezas detectadas"] > 0
        ].copy()

        df_prioritarios = df_prioritarios.sort_values(
            by=["Imagen", "Malezas detectadas"],
            ascending=[True, False]
        )
    else:
        df_prioritarios = pd.DataFrame()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_resumen.to_excel(writer, index=False, sheet_name="Resumen")
        df_sectores.to_excel(writer, index=False, sheet_name="Sectores")
        df_prioritarios.to_excel(writer, index=False, sheet_name="Prioritarios")

        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]

            for column_cells in worksheet.columns:
                max_length = 0
                column_letter = column_cells[0].column_letter

                for cell in column_cells:
                    value_length = len(str(cell.value)) if cell.value is not None else 0
                    max_length = max(max_length, value_length)

                worksheet.column_dimensions[column_letter].width = min(max_length + 3, 45)

            for cell in worksheet[1]:
                font = copy(cell.font)
                font.bold = True
                cell.font = font

    buffer.seek(0)
    return buffer.getvalue()


def mostrar_leyenda_mapa():
    st.markdown(
        """
        <div style="display: flex; gap: 16px; align-items: center; margin-bottom: 12px; flex-wrap: wrap;">
            <div style="display: flex; align-items: center; gap: 6px;">
                <div style="width: 22px; height: 14px; background-color: #38a169; border-radius: 3px;"></div>
                <span>Baja presencia</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px;">
                <div style="width: 22px; height: 14px; background-color: #d69e2e; border-radius: 3px;"></div>
                <span>Presencia media</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px;">
                <div style="width: 22px; height: 14px; background-color: #e53e3e; border-radius: 3px;"></div>
                <span>Alta presencia</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px;">
                <div style="width: 22px; height: 14px; background-color: transparent; border: 3px solid black; border-radius: 3px;"></div>
                <span>Sector más afectado</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def analizar_imagen(
    archivo,
    modelo,
    tipo_entrada,
    confianza,
    filas,
    columnas,
    umbral_bajo,
    umbral_medio,
    tam_recorte,
    solapamiento,
    mostrar_etiquetas,
    mostrar_nombres_sector,
    opacidad_mapa,
    resolucion_cm_pixel
):
    archivo.seek(0)
    imagen_pil = Image.open(archivo).convert("RGB")
    imagen = np.array(imagen_pil)

    alto, ancho = imagen.shape[:2]

    _, area_total_m2, area_sector_m2 = calcular_datos_espaciales(
        ancho=ancho,
        alto=alto,
        filas=filas,
        columnas=columnas,
        resolucion_cm_pixel=resolucion_cm_pixel
    )

    recortes, posiciones = dividir_imagen_en_recortes(
        imagen,
        tam_recorte=tam_recorte,
        solapamiento=solapamiento
    )

    detecciones = detectar_malezas_en_recortes(
        modelo,
        recortes,
        posiciones,
        confianza=confianza
    )

    detecciones_limpias = eliminar_duplicados(
        detecciones,
        umbral_iou=0.5
    )

    total_malezas = len(detecciones_limpias)

    matriz = generar_matriz_infestacion(
        detecciones_limpias,
        ancho_imagen=ancho,
        alto_imagen=alto,
        filas=filas,
        columnas=columnas
    )

    tabla = crear_tabla_sectores(
        matriz,
        umbral_bajo=umbral_bajo,
        umbral_medio=umbral_medio,
        area_sector_m2=area_sector_m2
    )

    sectores_con_malezas = tabla[
        tabla["malezas_detectadas"] > 0
    ].shape[0]

    sector_max = tabla.iloc[0]
    nivel_maximo = obtener_nivel_maximo(tabla)

    sector_destacado = (
        int(sector_max["fila"]),
        int(sector_max["columna"])
    )

    imagen_detecciones = dibujar_detecciones(
        imagen,
        detecciones_limpias,
        mostrar_etiquetas=mostrar_etiquetas,
        filas=filas,
        columnas=columnas,
        mostrar_grilla=mostrar_nombres_sector,
        sector_destacado=sector_destacado
    )

    imagen_mapa = dibujar_mapa_infestacion(
        imagen,
        matriz,
        opacidad=opacidad_mapa,
        mostrar_nombres_sector=mostrar_nombres_sector,
        umbral_bajo=umbral_bajo,
        umbral_medio=umbral_medio,
        sector_destacado=sector_destacado
    )

    imagen_zoom_sector = crear_zoom_sector_imagen(
        imagen=imagen,
        detecciones=detecciones_limpias,
        fila_sector=sector_destacado[0],
        columna_sector=sector_destacado[1],
        filas=filas,
        columnas=columnas,
        mostrar_etiquetas=mostrar_etiquetas,
        max_lado=1800
    )

    if area_total_m2 is not None and area_total_m2 > 0:
        densidad_general = total_malezas / area_total_m2
    else:
        densidad_general = None

    return {
        "nombre_imagen": archivo.name,
        "tipo_entrada": tipo_entrada,
        "ancho": ancho,
        "alto": alto,
        "recortes": len(recortes),
        "detecciones_antes": len(detecciones),
        "detecciones_finales": total_malezas,
        "sectores_con_malezas": sectores_con_malezas,
        "nivel_maximo": nivel_maximo,
        "sector_max": sector_max,
        "matriz": matriz,
        "tabla": tabla,
        "imagen_detecciones": imagen_detecciones,
        "imagen_mapa": imagen_mapa,
        "imagen_zoom_sector": imagen_zoom_sector,
        "umbral_bajo": umbral_bajo,
        "umbral_medio": umbral_medio,
        "resolucion_cm_pixel": resolucion_cm_pixel,
        "area_total_m2": area_total_m2,
        "area_sector_m2": area_sector_m2,
        "densidad_general": densidad_general
    }


def analizar_ortomosaico_geotiff(
    ruta_geotiff,
    nombre_imagen,
    modelo,
    tipo_entrada,
    confianza,
    filas,
    columnas,
    umbral_bajo,
    umbral_medio,
    tam_recorte,
    solapamiento,
    mostrar_etiquetas,
    mostrar_nombres_sector,
    opacidad_mapa,
    resolucion_cm_pixel,
    max_ventanas=0,
    progreso_callback=None
):
    """
    Analiza un ortomosaico GeoTIFF sin cargarlo completo en memoria.
    Lo procesa por ventanas y luego genera una vista previa para mostrar resultados.
    """

    detecciones, ancho, alto, ventanas_procesadas, ventanas_totales = procesar_geotiff_por_ventanas(
        ruta_geotiff=ruta_geotiff,
        modelo=modelo,
        confianza=confianza,
        tam_recorte=tam_recorte,
        solapamiento=solapamiento,
        max_ventanas=max_ventanas,
        progreso_callback=progreso_callback
    )

    detecciones_limpias = eliminar_duplicados(
        detecciones,
        umbral_iou=0.5
    )

    total_malezas = len(detecciones_limpias)

    _, area_total_m2, area_sector_m2 = calcular_datos_espaciales(
        ancho=ancho,
        alto=alto,
        filas=filas,
        columnas=columnas,
        resolucion_cm_pixel=resolucion_cm_pixel
    )

    matriz = generar_matriz_infestacion(
        detecciones_limpias,
        ancho_imagen=ancho,
        alto_imagen=alto,
        filas=filas,
        columnas=columnas
    )

    tabla = crear_tabla_sectores(
        matriz,
        umbral_bajo=umbral_bajo,
        umbral_medio=umbral_medio,
        area_sector_m2=area_sector_m2
    )

    sectores_con_malezas = tabla[
        tabla["malezas_detectadas"] > 0
    ].shape[0]

    sector_max = tabla.iloc[0]
    nivel_maximo = obtener_nivel_maximo(tabla)

    sector_destacado = (
        int(sector_max["fila"]),
        int(sector_max["columna"])
    )

    preview, escala_x, escala_y = crear_preview_geotiff(
        ruta_geotiff,
        max_lado=3000
    )

    detecciones_preview = escalar_detecciones(
        detecciones_limpias,
        escala_x=escala_x,
        escala_y=escala_y
    )

    imagen_detecciones = dibujar_detecciones(
        preview,
        detecciones_preview,
        mostrar_etiquetas=mostrar_etiquetas,
        filas=filas,
        columnas=columnas,
        mostrar_grilla=mostrar_nombres_sector,
        sector_destacado=sector_destacado
    )

    imagen_mapa = dibujar_mapa_infestacion(
        preview,
        matriz,
        opacidad=opacidad_mapa,
        mostrar_nombres_sector=mostrar_nombres_sector,
        umbral_bajo=umbral_bajo,
        umbral_medio=umbral_medio,
        sector_destacado=sector_destacado
    )

    imagen_zoom_sector = crear_zoom_sector_geotiff(
        ruta_geotiff=ruta_geotiff,
        detecciones=detecciones_limpias,
        ancho_total=ancho,
        alto_total=alto,
        filas=filas,
        columnas=columnas,
        fila_sector=sector_destacado[0],
        columna_sector=sector_destacado[1],
        mostrar_etiquetas=mostrar_etiquetas,
        max_lado=1800
    )

    if area_total_m2 is not None and area_total_m2 > 0:
        densidad_general = total_malezas / area_total_m2
    else:
        densidad_general = None

    return {
        "nombre_imagen": nombre_imagen,
        "tipo_entrada": tipo_entrada,
        "ancho": ancho,
        "alto": alto,
        "recortes": ventanas_procesadas,
        "ventanas_totales": ventanas_totales,
        "detecciones_antes": len(detecciones),
        "detecciones_finales": total_malezas,
        "sectores_con_malezas": sectores_con_malezas,
        "nivel_maximo": nivel_maximo,
        "sector_max": sector_max,
        "matriz": matriz,
        "tabla": tabla,
        "imagen_detecciones": imagen_detecciones,
        "imagen_mapa": imagen_mapa,
        "imagen_zoom_sector": imagen_zoom_sector,
        "umbral_bajo": umbral_bajo,
        "umbral_medio": umbral_medio,
        "resolucion_cm_pixel": resolucion_cm_pixel,
        "area_total_m2": area_total_m2,
        "area_sector_m2": area_sector_m2,
        "densidad_general": densidad_general
    }


st.title("Detección de malezas en cultivos de maíz")

st.write(
    "MVP para analizar imágenes UAV u ortomosaicos de cultivos de maíz, "
    "detectar malezas con YOLOv8 y generar mapas preliminares de infestación por sectores."
)

st.divider()


if not Path(RUTA_MODELO).exists():
    st.error(
        "No se encontró el modelo entrenado. Debes colocar el archivo "
        "`best_yolov8s.pt` dentro de la carpeta `modelo/`."
    )
    st.stop()


with st.sidebar:
    st.header("Configuración")

    # Se elimina el control visible "Modo presentación".
    # La variable se mantiene para no modificar el resto del flujo de la app.
    modo_presentacion = False

    tipo_entrada = st.radio(
        "Tipo de entrada",
        options=[
            "Imagen/mosaico de prueba",
            "Ortomosaico"
        ],
        index=0
    )

    usar_resolucion = st.checkbox(
        "Usar resolución espacial aproximada",
        value=tipo_entrada == "Ortomosaico"
    )

    if usar_resolucion:
        resolucion_cm_pixel = st.number_input(
            "Resolución espacial aproximada (cm/píxel)",
            min_value=0.01,
            max_value=50.0,
            value=1.0,
            step=0.1
        )
    else:
        resolucion_cm_pixel = None

    confianza = st.slider(
        "Confianza mínima",
        min_value=0.05,
        max_value=0.90,
        value=0.15,
        step=0.05
    )

    filas = st.slider(
        "Filas del mapa",
        min_value=3,
        max_value=20,
        value=10,
        step=1
    )

    columnas = st.slider(
        "Columnas del mapa",
        min_value=3,
        max_value=20,
        value=10,
        step=1
    )

    # Umbrales de infestación.
    # Los valores por defecto cambian según el tipo de entrada:
    # - Imagen/mosaico: sectores pequeños, por eso se usan umbrales bajos.
    # - Ortomosaico: sectores grandes, por eso se usan umbrales más altos.
    # El límite sigue siendo amplio, pero no exagerado.
    LIMITE_UMBRAL = 10_000

    if tipo_entrada == "Ortomosaico":
        umbral_bajo_defecto = 25
        umbral_medio_defecto = 75
    else:
        umbral_bajo_defecto = 2
        umbral_medio_defecto = 5

    umbral_bajo_usuario = st.number_input(
        "Máximo para nivel bajo",
        min_value=1,
        max_value=LIMITE_UMBRAL - 1,
        value=umbral_bajo_defecto,
        step=1,
        key=f"umbral_bajo_{tipo_entrada}",
        help=(
            "Cantidad máxima de detecciones para clasificar un sector como baja presencia. "
            "En ortomosaicos conviene usar valores más altos porque cada sector puede cubrir más terreno."
        )
    )

    umbral_medio_usuario = st.number_input(
        "Máximo para nivel medio",
        min_value=1,
        max_value=LIMITE_UMBRAL,
        value=umbral_medio_defecto,
        step=1,
        key=f"umbral_medio_{tipo_entrada}",
        help=(
            "Cantidad máxima de detecciones para clasificar un sector como presencia media. "
            "Todo valor superior a este se clasifica como alta presencia."
        )
    )

    umbral_bajo = int(umbral_bajo_usuario)
    umbral_medio = int(umbral_medio_usuario)

    if umbral_medio <= umbral_bajo:
        umbral_medio = min(umbral_bajo + 1, LIMITE_UMBRAL)
        st.warning(
            "El máximo para nivel medio debe ser mayor que el máximo para nivel bajo. "
            f"Para este análisis se usará automáticamente nivel medio = {umbral_medio}."
        )

    st.divider()

    modo_detecciones = st.radio(
        "Visualización de detecciones",
        options=[
            "Solo cajas",
            "Cajas con etiquetas"
        ],
        index=0
    )

    mostrar_nombres_sector = st.checkbox(
        "Mostrar filas y columnas",
        value=True
    )

    opacidad_mapa = st.slider(
        "Opacidad del mapa de infestación",
        min_value=0.10,
        max_value=0.70,
        value=0.35,
        step=0.05
    )

    with st.expander("Opciones avanzadas"):
        tam_recorte = st.selectbox(
            "Tamaño de recorte / ventana",
            options=[640, 768, 1024],
            index=0
        )

        solapamiento = st.slider(
            "Solapamiento entre recortes / ventanas",
            min_value=0,
            max_value=300,
            value=100,
            step=25
        )

        max_ventanas = st.number_input(
            "Máximo de ventanas GeoTIFF (0 = procesar completo)",
            min_value=0,
            value=50,
            step=10
        )


usar_geotiff_completo = False
ruta_geotiff_local = None
archivos = None

if tipo_entrada == "Ortomosaico":
    st.subheader("Entrada de ortomosaico")

    modo_ortomosaico = st.radio(
        "Forma de cargar el ortomosaico",
        options=[
            "Ruta local GeoTIFF",
            "Subir archivo GeoTIFF pequeño"
        ],
        index=0
    )

    if modo_ortomosaico == "Ruta local GeoTIFF":
        usar_geotiff_completo = True

        ruta_geotiff_local = st.text_input(
            "Ruta local del archivo .tif o .tiff",
            placeholder=r"D:\datasets\ortomosaico.tif"
        )

        st.caption(
            "Esta opción es recomendada para ortomosaicos grandes, porque evita subir el archivo por el navegador."
        )

    else:
        archivos = st.file_uploader(
            "Sube uno o varios GeoTIFF pequeños",
            type=["tif", "tiff"],
            accept_multiple_files=True
        )

else:
    archivos = st.file_uploader(
        "Sube una o varias imágenes aéreas del cultivo",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )


if archivos:
    st.write(f"Imágenes cargadas: **{len(archivos)}**")

    if not modo_presentacion:
        with st.expander("Ver imágenes cargadas"):
            for archivo in archivos:
                sufijo = Path(archivo.name).suffix.lower()

                if sufijo in [".jpg", ".jpeg", ".png"]:
                    archivo.seek(0)
                    imagen_previa = Image.open(archivo).convert("RGB")
                    st.image(
                        imagen_previa,
                        caption=archivo.name,
                        use_container_width=True
                    )
                else:
                    st.write(f"Archivo GeoTIFF cargado: {archivo.name}")


if st.button("Analizar imágenes", type="primary"):
    modelo = obtener_modelo()
    mostrar_etiquetas = modo_detecciones == "Cajas con etiquetas"

    resultados = []
    progreso = st.progress(0)
    estado = st.empty()

    if usar_geotiff_completo:
        if not ruta_geotiff_local:
            st.error("Debes ingresar la ruta local del GeoTIFF.")
            st.stop()

        if not Path(ruta_geotiff_local).exists():
            st.error("No se encontró el archivo GeoTIFF en la ruta indicada.")
            st.stop()

        resolucion_final = resolucion_cm_pixel

        if resolucion_final is None:
            resolucion_detectada = obtener_resolucion_cm_pixel(ruta_geotiff_local)

            if resolucion_detectada is not None:
                resolucion_final = resolucion_detectada
                st.info(
                    f"Resolución espacial detectada automáticamente: "
                    f"{resolucion_final:.3f} cm/píxel"
                )

        def progreso_geotiff(actual, total_procesar, total_ventanas):
            progreso.progress(actual / total_procesar)
            estado.write(
                f"Procesando ventanas del ortomosaico: "
                f"{actual}/{total_procesar} "
                f"(total disponibles: {total_ventanas})"
            )

        resultado = analizar_ortomosaico_geotiff(
            ruta_geotiff=ruta_geotiff_local,
            nombre_imagen=Path(ruta_geotiff_local).name,
            modelo=modelo,
            tipo_entrada=tipo_entrada,
            confianza=confianza,
            filas=filas,
            columnas=columnas,
            umbral_bajo=umbral_bajo,
            umbral_medio=umbral_medio,
            tam_recorte=tam_recorte,
            solapamiento=solapamiento,
            mostrar_etiquetas=mostrar_etiquetas,
            mostrar_nombres_sector=mostrar_nombres_sector,
            opacidad_mapa=opacidad_mapa,
            resolucion_cm_pixel=resolucion_final,
            max_ventanas=max_ventanas,
            progreso_callback=progreso_geotiff
        )

        resultados.append(resultado)

    else:
        if not archivos:
            st.error("Debes subir al menos una imagen o archivo GeoTIFF.")
            st.stop()

        for idx, archivo in enumerate(archivos):
            estado.write(f"Analizando {archivo.name}...")

            sufijo = Path(archivo.name).suffix.lower()

            if sufijo in [".tif", ".tiff"]:
                with tempfile.NamedTemporaryFile(delete=False, suffix=sufijo) as tmp:
                    tmp.write(archivo.getbuffer())
                    ruta_tmp = tmp.name

                resolucion_final = resolucion_cm_pixel

                if resolucion_final is None:
                    resolucion_detectada = obtener_resolucion_cm_pixel(ruta_tmp)

                    if resolucion_detectada is not None:
                        resolucion_final = resolucion_detectada

                def progreso_geotiff_subido(actual, total_procesar, total_ventanas):
                    progreso.progress(actual / total_procesar)
                    estado.write(
                        f"Procesando ventanas de {archivo.name}: "
                        f"{actual}/{total_procesar} "
                        f"(total disponibles: {total_ventanas})"
                    )

                resultado = analizar_ortomosaico_geotiff(
                    ruta_geotiff=ruta_tmp,
                    nombre_imagen=archivo.name,
                    modelo=modelo,
                    tipo_entrada=tipo_entrada,
                    confianza=confianza,
                    filas=filas,
                    columnas=columnas,
                    umbral_bajo=umbral_bajo,
                    umbral_medio=umbral_medio,
                    tam_recorte=tam_recorte,
                    solapamiento=solapamiento,
                    mostrar_etiquetas=mostrar_etiquetas,
                    mostrar_nombres_sector=mostrar_nombres_sector,
                    opacidad_mapa=opacidad_mapa,
                    resolucion_cm_pixel=resolucion_final,
                    max_ventanas=max_ventanas,
                    progreso_callback=progreso_geotiff_subido
                )

                os.remove(ruta_tmp)

            else:
                resultado = analizar_imagen(
                    archivo=archivo,
                    modelo=modelo,
                    tipo_entrada=tipo_entrada,
                    confianza=confianza,
                    filas=filas,
                    columnas=columnas,
                    umbral_bajo=umbral_bajo,
                    umbral_medio=umbral_medio,
                    tam_recorte=tam_recorte,
                    solapamiento=solapamiento,
                    mostrar_etiquetas=mostrar_etiquetas,
                    mostrar_nombres_sector=mostrar_nombres_sector,
                    opacidad_mapa=opacidad_mapa,
                    resolucion_cm_pixel=resolucion_cm_pixel
                )

            resultados.append(resultado)

            progreso.progress((idx + 1) / len(archivos))

    estado.write("Análisis completado.")
    st.session_state["resultados"] = resultados


if "resultados" in st.session_state:
    resultados = st.session_state["resultados"]

    total_general = sum(r["detecciones_finales"] for r in resultados)
    total_imagenes = len(resultados)
    sectores_con_malezas_total = sum(r["sectores_con_malezas"] for r in resultados)

    area_total_general = None
    densidad_general_lote = None

    resultados_con_area = [
        r for r in resultados
        if r["resolucion_cm_pixel"] is not None and r["area_total_m2"] is not None
    ]

    if resultados_con_area:
        area_total_general = sum(r["area_total_m2"] for r in resultados_con_area)

        if area_total_general > 0:
            densidad_general_lote = total_general / area_total_general

    resumen_general = pd.DataFrame([
        {
            "Imagen": r["nombre_imagen"],
            "Malezas detectadas": r["detecciones_finales"],
            "Sectores con presencia": r["sectores_con_malezas"],
            "Sector más afectado": r["sector_max"]["sector"],
            "Nivel máximo": r["nivel_maximo"],
            **(
                {
                    "Área aproximada (m²)": round(r["area_total_m2"], 2),
                    "Densidad (malezas/m²)": round(r["densidad_general"], 3)
                }
                if r["resolucion_cm_pixel"] is not None and r["area_total_m2"] is not None
                else {}
            )
        }
        for r in resultados
    ])

    st.success("Análisis completado correctamente.")

    st.subheader("Resumen general")

    if area_total_general is None:
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Imágenes analizadas", total_imagenes)

        with col_b:
            st.metric("Total de malezas detectadas", total_general)

        with col_c:
            st.metric("Sectores con presencia", sectores_con_malezas_total)

    else:
        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            st.metric("Imágenes analizadas", total_imagenes)

        with col_b:
            st.metric("Total de malezas detectadas", total_general)

        with col_c:
            st.metric("Área aproximada", f"{area_total_general:.2f} m²")

        with col_d:
            st.metric("Densidad general", f"{densidad_general_lote:.3f} malezas/m²")

    st.dataframe(
        resumen_general,
        use_container_width=True
    )

    excel = crear_excel_resultados(resultados)

    st.download_button(
        label="Descargar reporte Excel",
        data=excel,
        file_name="reporte_malezas.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    nombres_imagenes = [r["nombre_imagen"] for r in resultados]

    imagen_seleccionada = st.selectbox(
        "Selecciona una imagen para revisar el detalle",
        options=nombres_imagenes
    )

    resultado = next(
        r for r in resultados
        if r["nombre_imagen"] == imagen_seleccionada
    )

    tabla_ordenada = resultado["tabla"]
    total_malezas = resultado["detecciones_finales"]
    sectores_con_malezas = resultado["sectores_con_malezas"]
    sector_max = resultado["sector_max"]
    nivel_maximo = resultado["nivel_maximo"]

    st.subheader(f"Detalle de análisis: {resultado['nombre_imagen']}")

    if resultado["resolucion_cm_pixel"] is None or resultado["area_total_m2"] is None:
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)

        with col_d1:
            st.metric("Malezas detectadas", total_malezas)

        with col_d2:
            st.metric("Sectores con presencia", sectores_con_malezas)

        with col_d3:
            st.metric("Sector más afectado", f'{sector_max["sector"]}')

        with col_d4:
            st.metric("Nivel máximo", nivel_maximo)

    else:
        col_d1, col_d2, col_d3, col_d4, col_d5 = st.columns(5)

        with col_d1:
            st.metric("Malezas detectadas", total_malezas)

        with col_d2:
            st.metric("Área aprox.", f'{resultado["area_total_m2"]:.2f} m²')

        with col_d3:
            st.metric("Densidad", f'{resultado["densidad_general"]:.3f} malezas/m²')

        with col_d4:
            st.metric("Sector más afectado", f'{sector_max["sector"]}')

        with col_d5:
            st.metric("Nivel máximo", nivel_maximo)

    if not modo_presentacion:
        st.caption(
            f'Recortes / ventanas procesadas: {resultado["recortes"]} | '
            f'Detecciones antes de eliminar duplicados: {resultado["detecciones_antes"]} | '
            f'Detecciones finales: {resultado["detecciones_finales"]}'
        )

    st.subheader("Sectores prioritarios")

    sectores_prioritarios = tabla_ordenada[
        tabla_ordenada["malezas_detectadas"] > 0
    ].head(5)

    if sectores_prioritarios.empty:
        st.info("No se detectaron sectores con presencia de malezas.")
    else:
        st.dataframe(
            limpiar_columnas_tabla(sectores_prioritarios),
            use_container_width=True
        )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Detecciones",
            "Mapa de infestación",
            "Zoom sector prioritario",
            "Tabla de sectores",
            "Reporte",
            "Flujo real con dron"
        ]
    )

    with tab1:
        st.subheader("Imagen con detecciones")

        st.image(
            resultado["imagen_detecciones"],
            use_container_width=True
        )

        st.download_button(
            label="Descargar imagen con detecciones",
            data=convertir_imagen_a_bytes(resultado["imagen_detecciones"]),
            file_name=f"detecciones_{resultado['nombre_imagen']}.png",
            mime="image/png"
        )

    with tab2:
        st.subheader("Mapa preliminar de infestación")

        mostrar_leyenda_mapa()

        st.image(
            resultado["imagen_mapa"],
            use_container_width=True
        )

        st.download_button(
            label="Descargar mapa de infestación",
            data=convertir_imagen_a_bytes(resultado["imagen_mapa"]),
            file_name=f"mapa_{resultado['nombre_imagen']}.png",
            mime="image/png"
        )

    with tab3:
        st.subheader("Zoom del sector prioritario")

        st.write(
            f'Sector mostrado: **{sector_max["sector"]}**. '
            "Este acercamiento permite revisar con más detalle las detecciones "
            "dentro del sector con mayor presencia de malezas."
        )

        st.image(
            resultado["imagen_zoom_sector"],
            use_container_width=True
        )

        st.download_button(
            label="Descargar zoom del sector",
            data=convertir_imagen_a_bytes(resultado["imagen_zoom_sector"]),
            file_name=f"zoom_sector_{resultado['nombre_imagen']}.png",
            mime="image/png"
        )

    with tab4:
        st.subheader("Tabla de sectores")

        opcion_filtro = st.radio(
            "Filtrar tabla",
            options=[
                "Todos los sectores",
                "Solo sectores con malezas",
                "Solo sectores media y alta",
                "Solo sectores alta"
            ],
            horizontal=True
        )

        tabla_filtrada = filtrar_tabla(
            tabla_ordenada,
            opcion_filtro
        )

        tabla_descarga = limpiar_columnas_tabla(tabla_filtrada)

        if tabla_filtrada.empty:
            st.info("No hay sectores que cumplan con el filtro seleccionado.")
        else:
            st.dataframe(
                tabla_descarga,
                use_container_width=True
            )

        csv = tabla_descarga.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="Descargar tabla CSV",
            data=csv,
            file_name=f"tabla_sectores_{resultado['nombre_imagen']}.csv",
            mime="text/csv"
        )

    with tab5:
        st.subheader("Reporte del análisis")

        reporte = crear_reporte_texto(
            nombre_imagen=resultado["nombre_imagen"],
            tipo_entrada=resultado["tipo_entrada"],
            ancho=resultado["ancho"],
            alto=resultado["alto"],
            recortes=resultado["recortes"],
            detecciones_antes=resultado["detecciones_antes"],
            detecciones_finales=resultado["detecciones_finales"],
            sectores_con_malezas=sectores_con_malezas,
            nivel_maximo=nivel_maximo,
            sector_max=sector_max,
            tabla=tabla_ordenada,
            umbral_bajo=resultado["umbral_bajo"],
            umbral_medio=resultado["umbral_medio"],
            resolucion_cm_pixel=resultado["resolucion_cm_pixel"],
            area_total_m2=resultado["area_total_m2"],
            area_sector_m2=resultado["area_sector_m2"],
            densidad_general=resultado["densidad_general"]
        )

        st.text_area(
            "Vista previa del reporte",
            reporte,
            height=350
        )

        st.download_button(
            label="Descargar reporte TXT",
            data=reporte.encode("utf-8-sig"),
            file_name=f"reporte_{resultado['nombre_imagen']}.txt",
            mime="text/plain"
        )

    with tab6:
        st.subheader("Flujo real con dron")

        st.markdown(
            """
            En un escenario real, el flujo recomendado es:

            1. Capturar imágenes con dron sobre el cultivo.
            2. Procesar las imágenes en un software de fotogrametría.
            3. Generar un ortomosaico del terreno.
            4. Cargar el ortomosaico GeoTIFF en esta aplicación.
            5. Procesar el ortomosaico por ventanas, sin cargarlo completo en memoria.
            6. Detectar malezas con YOLOv8.
            7. Agrupar las detecciones por sectores.
            8. Priorizar zonas con mayor presencia o densidad de malezas.

            Esta versión permite analizar imágenes comunes y también ortomosaicos GeoTIFF
            mediante lectura por ventanas.
            """
        )

else:
    st.info(
        "Sube una o varias imágenes aéreas, o selecciona un ortomosaico GeoTIFF, "
        "y presiona **Analizar imágenes** para comenzar."
    )