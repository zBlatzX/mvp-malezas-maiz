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
- Estimación de área y densidad cuando se indica una resolución espacial aproximada.
- Exportación de resultados a CSV, Excel y TXT.
- Notebooks de apoyo para descargar datasets y generar mosaicos de prueba.
- Script auxiliar para crear recortes pequeños de ortomosaicos GeoTIFF.

## Estructura del repositorio

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
├── notebooks/
│   ├── 01_descargar_usu_crear_mosaicos_pc.ipynb
│   └── 02_descargar_weedsgalore_ortomosaicos_pc.ipynb
└── test/
    └── crear_ortomosaico_pequeño.py
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

El modelo entrenado debe estar en:

```text
modelo/best_yolov8s.pt
```

Este archivo corresponde al modelo YOLOv8 ajustado para la detección de malezas.

Si el archivo pesa más de 100 MB, no se recomienda subirlo directamente a GitHub. En ese caso se puede usar Git LFS, Google Drive, OneDrive o Zenodo, y dejar la ruta o enlace de descarga en este README.

## Datasets utilizados

### USU-Corn-WeedDB

USU-Corn-WeedDB se utilizó principalmente para entrenamiento y pruebas en el dominio del modelo. Contiene imágenes UAV RGB de cultivos de maíz y anotaciones de malezas.

En este proyecto se utilizó para entrenar YOLOv8. Las clases originales del dataset fueron tratadas como una clase general de maleza, ya que el objetivo del MVP era detectar presencia de malezas y representarlas por sectores.

El notebook:

```text
notebooks/01_descargar_usu_crear_mosaicos_pc.ipynb
```

permite descargar el dataset en Google Colab, crear mosaicos artificiales con imágenes del dataset y descargar el resultado al computador.

Los mosaicos generados no son ortomosaicos reales. Son imágenes grandes construidas a partir de parches del dataset, útiles para probar la aplicación con imágenes más amplias y cercanas al dominio usado para entrenar el modelo.

### WeedsGalore

WeedsGalore se utilizó principalmente para validar funcionalmente el procesamiento de ortomosaicos reales. Este dataset incluye ortomosaicos GeoTIFF de campos de maíz, lo que permite probar el flujo de lectura por ventanas en imágenes grandes.

El notebook:

```text
notebooks/02_descargar_weedsgalore_ortomosaicos_pc.ipynb
```

permite descargar los ortomosaicos en Google Colab y luego descargarlos al computador.

Debido al tamaño de los archivos GeoTIFF, también se recomienda la descarga directa desde el computador cuando sea necesario.

Para la demostración se recomienda comenzar con:

```text
2023-06-06_om.tif
2023-05-30_om.tif
```

Estos fueron los archivos que se trabajaron de forma más estable durante las pruebas.

## Notebooks de descarga

Los notebooks incluidos no guardan archivos en Google Drive. Utilizan el almacenamiento temporal de Google Colab (`/content`) y al final descargan los archivos generados al computador.

### Notebook 1: USU-Corn-WeedDB

```text
01_descargar_usu_crear_mosaicos_pc.ipynb
```

Este notebook realiza las siguientes acciones:

1. Descarga USU-Corn-WeedDB.
2. Descomprime el dataset.
3. Busca imágenes del conjunto de entrenamiento, validación o prueba.
4. Genera mosaicos artificiales de 5×5 y 10×10.
5. Comprime los mosaicos en un archivo `.zip`.
6. Descarga el `.zip` al computador.

### Notebook 2: WeedsGalore

```text
02_descargar_weedsgalore_ortomosaicos_pc.ipynb
```

Este notebook realiza las siguientes acciones:

1. Descarga el archivo de ortomosaicos de WeedsGalore.
2. Extrae los GeoTIFF recomendados para pruebas.
3. Permite descargar los `.tif` al computador.
4. Incluye una alternativa con PowerShell para descargar directamente desde Windows.

## Script auxiliar para ortomosaicos pequeños

La carpeta `test/` contiene el script:

```text
test/crear_ortomosaico_pequeño.py
```

Este script se utiliza como apoyo para crear un recorte más pequeño de un ortomosaico GeoTIFF grande. Su objetivo es facilitar pruebas locales, demostraciones o validaciones rápidas sin tener que procesar siempre un archivo completo de varios GB.

Este script no forma parte del flujo principal de la aplicación Streamlit. Es una herramienta auxiliar para preparar archivos de prueba.

Uso general:

```powershell
python test/crear_ortomosaico_pequeño.py
```

Antes de ejecutarlo, se deben revisar las rutas configuradas dentro del archivo, por ejemplo:

```text
RUTA_ORIGINAL
RUTA_SALIDA
```

El archivo generado puede usarse luego en la aplicación seleccionando:

```text
Tipo de entrada: Ortomosaico
Forma de cargar el ortomosaico: Ruta local GeoTIFF
```

Importante: los ortomosaicos o recortes `.tif` generados por este script no deberían subirse al repositorio, ya que pueden ser archivos pesados. Estos archivos deben mantenerse de forma local.

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

En un escenario real, la aplicación parte desde un ortomosaico ya generado. Si solo se tienen fotografías individuales capturadas por un dron, primero deben procesarse en un software de fotogrametría, como WebODM, Agisoft Metashape, Pix4D u otro similar.

El flujo recomendado es:

```text
Captura UAV
→ procesamiento fotogramétrico externo
→ ortomosaico GeoTIFF
→ lectura por ventanas en la app
→ detección con YOLOv8
→ conteo por sectores
→ mapa de infestación
→ tabla / reporte
```

La aplicación no construye el ortomosaico desde las fotografías originales. Su función es analizar imágenes ya disponibles u ortomosaicos GeoTIFF previamente generados.

## Procesamiento por ventanas

Los ortomosaicos pueden tener un tamaño muy grande, por lo que no se cargan completos en memoria.

Para evitar problemas de rendimiento, la aplicación los divide en ventanas o recortes. Cada ventana se procesa con YOLOv8 y luego las detecciones se convierten nuevamente a coordenadas globales dentro del ortomosaico.

Finalmente, las detecciones se agrupan en sectores para generar el mapa de infestación.

En la aplicación, el tamaño de ventana puede configurarse como:

```text
640 × 640 píxeles
768 × 768 píxeles
1024 × 1024 píxeles
```

El solapamiento entre ventanas también puede ajustarse para reducir la pérdida de detecciones en los bordes de cada recorte.

## Interpretación de niveles

La aplicación clasifica los sectores según la cantidad de malezas detectadas:

```text
0 detecciones                  → Sin presencia
1 hasta máximo bajo             → Baja presencia
máximo bajo + 1 hasta medio     → Presencia media
más del máximo medio            → Alta presencia
```

Los umbrales pueden ajustarse desde la interfaz.

Para imágenes o mosaicos pequeños se recomiendan valores bajos, por ejemplo:

```text
Bajo: 2
Medio: 5
```

Para ortomosaicos, donde los sectores pueden ser más grandes, se recomiendan valores más altos, por ejemplo:

```text
Bajo: 25
Medio: 75
```

Estos valores no representan una recomendación agronómica definitiva. Sirven para organizar visualmente los sectores y priorizar la revisión.

## Resultados de referencia

Durante las pruebas del proyecto se entrenaron y compararon los modelos YOLOv8n y YOLOv8s. El modelo YOLOv8s obtuvo el mejor desempeño general:

```text
Modelo     Precisión   Recall   mAP@0.5   mAP@0.5:0.95
YOLOv8n    0,729       0,762    0,806     0,460
YOLOv8s    0,763       0,773    0,837     0,486
```

En una prueba con mosaico de imágenes se obtuvieron:

```text
405 detecciones
87 sectores con presencia
grilla 10×10
```

En una prueba funcional con el ortomosaico `2023-06-06_om.tif` de WeedsGalore se obtuvieron:

```text
624 ventanas procesadas
253 detecciones antes de eliminar duplicados
238 detecciones finales
área aproximada: 52056,34 m²
densidad estimada: 0,005 malezas/m²
sector más afectado: F6-C4
```

Estos resultados deben interpretarse como una validación funcional del flujo. En el caso de WeedsGalore, se observaron detecciones incompletas debido al cambio de dominio entre el dataset utilizado para entrenar el modelo y el ortomosaico usado para probar el procesamiento.

## Consideraciones importantes

- El modelo fue entrenado con USU-Corn-WeedDB.
- Los ortomosaicos de WeedsGalore se usaron para validar funcionalmente el procesamiento de GeoTIFF.
- Como son datasets distintos, puede existir cambio de dominio.
- El cambio de dominio puede afectar las detecciones debido a diferencias de especies, escala, resolución, iluminación, suelo y condiciones visuales.
- Algunas malezas observadas en el ortomosaico presentaban una forma más fina o puntiaguda, lo que pudo afectar la detección del modelo. Esta observación corresponde a una inferencia visual y no a una validación cuantitativa definitiva.
- Los resultados en WeedsGalore no deben interpretarse como una validación agronómica definitiva.
- Para uso real en Arica u otra zona local, se recomienda capturar imágenes propias y realizar fine-tuning del detector.
- El sistema entrega apoyo visual y tabular, pero no reemplaza una evaluación agronómica en terreno.

## Archivos que no deben subirse al repositorio

No se recomienda subir al repositorio:

```text
.venv/
datasets/
weedsgalore/
USU-Corn-WeedDB/
*.tif
*.tiff
*.zip
runs/
resultados/
__pycache__/
```

Estos archivos pueden ser muy pesados o depender del computador local.

El repositorio debe contener el código, la documentación, los notebooks, el script auxiliar y los archivos necesarios para reproducir el entorno.

## Trabajo futuro

- Capturar imágenes UAV locales.
- Etiquetar malezas con apoyo experto.
- Ajustar el modelo con datos del mismo dominio visual.
- Evaluar segmentación para delimitar mejor las áreas de maleza.
- Exportar coordenadas geográficas reales para apoyar recorridos o aplicación localizada.
- Comparar resultados con anotaciones de terreno.
- Integrar criterios agronómicos más precisos para definir umbrales de intervención.