# Detección de malezas en cultivos de maíz

MVP desarrollado para analizar imágenes UAV y ortomosaicos de cultivos de maíz, detectar malezas mediante YOLOv8 y generar mapas preliminares de infestación por sectores.

El sistema está pensado como una herramienta de apoyo para agricultura de precisión. No aplica herbicidas automáticamente; entrega información visual y tabular para priorizar sectores de revisión o posible tratamiento localizado.

## Funcionalidades principales

- Carga de imágenes comunes (`.jpg`, `.jpeg`, `.png`).
- Carga de ortomosaicos GeoTIFF (`.tif`, `.tiff`) por ruta local o subida de archivo pequeño.
- Detección de malezas con YOLOv8.
- Procesamiento de imágenes grandes mediante ventanas.
- Agrupación de detecciones por sectores del cultivo.
- Generación de mapa preliminar de infestación.
- Zoom automático del sector prioritario.
- Tabla de sectores con nivel de infestación y recomendación.
- Exportación de resultados a CSV, Excel y TXT.

## Estructura sugerida del repositorio

```text
mvp_malezas/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── modelo/
│   └── best_yolov8s.pt
├── src/
│   ├── detector.py
│   ├── tiling.py
│   ├── infestacion.py
│   ├── visualizacion.py
│   └── ortomosaico.py
```

## Instalación

Se recomienda usar Python 3.10 o 3.11.

```powershell
cd "C:\RUTA\mvp_malezas"
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si PowerShell bloquea la activación del entorno virtual:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\activate
```

## Dependencias principales

El archivo `requirements.txt` debería incluir, al menos:

```text
streamlit
ultralytics
opencv-python
numpy
pandas
pillow
matplotlib
openpyxl
rasterio
```

## Modelo entrenado

El modelo debe estar en:

```text
modelo/best_yolov8s.pt
```

Si el archivo pesa más de 100 MB, no se recomienda subirlo directamente a GitHub. En ese caso se puede usar Git LFS, Google Drive, OneDrive o Zenodo, y dejar la ruta o enlace en el README.

## Datasets utilizados

### USU-Corn-WeedDB

Usado principalmente para entrenamiento y pruebas en el dominio del modelo. Contiene imágenes UAV RGB de maíz forrajero y anotaciones para malezas. En este proyecto se usó para entrenar YOLOv8.

El notebook `01_descargar_usu_crear_mosaicos.ipynb` permite descargar el dataset desde Zenodo y crear mosaicos artificiales con imágenes del dataset para probar la aplicación.

### WeedsGalore

Usado principalmente para validar el flujo con ortomosaicos reales. Los ortomosaicos completos están en formato GeoTIFF y permiten probar el procesamiento por ventanas.

El notebook `02_descargar_weedsgalore_ortomosaicos.ipynb` permite descargar los ortomosaicos y extraer los archivos más convenientes para la demo.

## Ejecución de la aplicación

Con el entorno virtual activado:

```powershell
streamlit run app.py
```

Luego, desde la interfaz:

1. Seleccionar si se analizará una imagen/mosaico o un ortomosaico.
2. Ajustar confianza, filas, columnas, tamaño de ventana, solapamiento y umbrales.
3. Cargar la imagen o indicar la ruta del GeoTIFF.
4. Presionar **Analizar imágenes**.
5. Revisar detecciones, mapa de infestación, zoom, tabla y reportes.

## Flujo real con dron

En un escenario real, la aplicación parte desde un ortomosaico ya generado. Si solo se tienen fotografías individuales del dron, primero deben procesarse en un software de fotogrametría como WebODM, Agisoft Metashape, Pix4D u otro similar.

```text
Captura UAV
→ procesamiento fotogramétrico externo
→ ortomosaico GeoTIFF
→ lectura por ventanas en la app
→ YOLOv8
→ mapa de infestación
→ tabla / reporte
```

## Consideraciones importantes

- El modelo fue entrenado con USU-Corn-WeedDB.
- Los ortomosaicos de WeedsGalore se usaron para validar funcionalmente el procesamiento de GeoTIFF.
- Como son datasets distintos, puede existir cambio de dominio: especies, escala, resolución, iluminación y condiciones visuales diferentes.
- Por eso, los resultados en WeedsGalore no deben interpretarse como una validación agronómica definitiva.
- Para uso real en Arica u otra zona local, se recomienda capturar imágenes propias y realizar fine-tuning del detector.

## Trabajo futuro

- Capturar imágenes UAV locales.
- Etiquetar malezas con apoyo experto.
- Ajustar el modelo con datos del mismo dominio visual.
- Evaluar segmentación para delimitar mejor las áreas de maleza.
- Exportar coordenadas geográficas reales para apoyar recorridos o aplicación localizada.
- Comparar resultados con anotaciones de terreno.
