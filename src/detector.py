import cv2
from ultralytics import YOLO


def cargar_modelo(ruta_modelo: str):
    """
    Carga el modelo YOLO entrenado.
    """
    return YOLO(ruta_modelo)


def detectar_malezas_en_recortes(modelo, recortes, posiciones, confianza: float = 0.15):
    """
    Aplica YOLO a cada recorte y transforma las coordenadas locales
    a coordenadas globales de la imagen completa.

    Importante:
    - La app carga la imagen en formato RGB usando PIL.
    - YOLO/Ultralytics trabaja normalmente con imágenes tipo OpenCV, es decir, BGR.
    - Por eso se convierte cada recorte de RGB a BGR antes de hacer la predicción.
    """

    detecciones_globales = []

    for recorte, (x_offset, y_offset) in zip(recortes, posiciones):

        # Convertir de RGB a BGR antes de pasarlo a YOLO
        recorte_bgr = cv2.cvtColor(recorte, cv2.COLOR_RGB2BGR)

        resultados = modelo.predict(
            recorte_bgr,
            conf=confianza,
            imgsz=640,
            verbose=False
        )

        resultado = resultados[0]

        if resultado.boxes is None:
            continue

        for box in resultado.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())

            detecciones_globales.append({
                "x1": float(x1 + x_offset),
                "y1": float(y1 + y_offset),
                "x2": float(x2 + x_offset),
                "y2": float(y2 + y_offset),
                "confianza": conf
            })

    return detecciones_globales


def calcular_iou(caja1, caja2):
    """
    Calcula el IoU entre dos cajas.
    IoU significa Intersection over Union, o intersección sobre unión.
    Se usa para saber si dos cajas se superponen mucho.
    """

    x1 = max(caja1["x1"], caja2["x1"])
    y1 = max(caja1["y1"], caja2["y1"])
    x2 = min(caja1["x2"], caja2["x2"])
    y2 = min(caja1["y2"], caja2["y2"])

    inter_ancho = max(0, x2 - x1)
    inter_alto = max(0, y2 - y1)
    inter_area = inter_ancho * inter_alto

    area1 = (caja1["x2"] - caja1["x1"]) * (caja1["y2"] - caja1["y1"])
    area2 = (caja2["x2"] - caja2["x1"]) * (caja2["y2"] - caja2["y1"])

    union = area1 + area2 - inter_area

    if union == 0:
        return 0

    return inter_area / union


def eliminar_duplicados(detecciones, umbral_iou: float = 0.5):
    """
    Elimina detecciones duplicadas que pueden aparecer por el solapamiento
    entre recortes.

    Mantiene la caja con mayor confianza y elimina otras cajas muy parecidas.
    """

    if len(detecciones) == 0:
        return []

    detecciones = sorted(
        detecciones,
        key=lambda d: d["confianza"],
        reverse=True
    )

    detecciones_finales = []

    while detecciones:
        mejor = detecciones.pop(0)
        detecciones_finales.append(mejor)

        detecciones = [
            det for det in detecciones
            if calcular_iou(mejor, det) < umbral_iou
        ]

    return detecciones_finales