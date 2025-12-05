
# Sistema Integral de Reconocimiento Facial y Análisis de Video con IA

Este repositorio contiene una suite de herramientas de visión por computadora e inteligencia artificial, dividida en dos grandes áreas: un **Analizador de Video Multimodal** potenciado por Google Gemini y un **Sistema de Reconocimiento Facial Dual** con dos enfoques de implementación distintos.

---

## Estructura del Proyecto

El proyecto se organiza en tres módulos principales:

```text
├── video_analysis/          # Módulo 1: Análisis de video con Gemini Pro
├── Face_recognition_dati/   # Módulo 2 (Variación A): Sistema de Vigilancia y Captura
└── face_recognition/        # Módulo 2 (Variación B): Sistema de Verificación y Testing
````

---

## Módulo de Reconocimiento Facial (Dos Variaciones)

El proyecto implementa **dos estrategias diferentes** para el reconocimiento facial, ubicadas en carpetas separadas. Ambas utilizan la librería **DeepFace** como motor y **OpenCV** para el procesamiento de imágenes, pero tienen propósitos distintos.

### Variación A: Sistema de Vigilancia y Captura (`Face_recognition_dati`)

**Ubicación:** `Face_recognition_dati/facial_recognition_realtime.py`

Esta variación está diseñada como un **sistema de seguridad o monitoreo en tiempo real**. Su objetivo es identificar personas conocidas y registrar automáticamente a los intrusos.

**Lógica de funcionamiento:**

1. Carga de base de datos: Lee las imágenes de la carpeta `known_faces/`.
2. Detección en vivo: Analiza el flujo de la webcam.
3. Clasificación:

   * Conocido (ROJO): Si el rostro coincide con la base de datos, dibuja un recuadro rojo y muestra el nombre.
   * Desconocido (AMARILLO/VERDE): Si el rostro no coincide, lo marca como desconocido.
4. Captura automática: Si detecta un desconocido, toma una foto automáticamente y la guarda en `known_faces` con un timestamp.
5. Cooldown: Sistema de enfriamiento (5 segundos) para evitar guardar múltiples fotos seguidas de la misma persona.
6. Optimización: Utiliza *threading* para que el reconocimiento (proceso pesado) no congele la imagen de la cámara.

**Uso ideal:** Control de acceso, registro de visitas, seguridad doméstica.

---

### Variación B: Sistema de Verificación y Testing (`face_recognition`)

**Ubicación:** `face_recognition/` (Scripts: `main.py`, `webcam_match.py`)

Esta variación funciona como **laboratorio de pruebas y verificación**. Compara conjuntos de datos (Train vs Test) para medir precisión del modelo y permite emparejamiento en vivo sin la lógica de guardado automático.

**Componentes:**

* **`main.py` (Comparador Estático):** Compara imágenes en `test/` contra `train/`. Genera un reporte en consola indicando coincidencias y porcentaje de similitud.
* **`webcam_match.py` (Verificador en Vivo):** Abre la cámara y busca coincidencias contra la carpeta `train/`. Muestra porcentaje de similitud en tiempo real.
* **Base de datos estructurada:** Usa carpetas separadas con nombres específicos (`josefina_1.png`, `luis_4.png`, etc.).

**Uso ideal:** Evaluar modelos (VGG-Face, Facenet), pruebas de concepto, demos de similitud.

---

## Módulo de Análisis de Video (`video_analysis`)

**Ubicación:** `video_analysis/app.py`

Aplicación web construida con **Streamlit**, utilizando la API de **Google Gemini 2.5 Pro** para analizar videos de forma multimodal.

**Características:**

* Subida de archivos de video (MP4, MOV, AVI).
* Prompting en lenguaje natural.
* Extracción de frames: La IA devuelve el *timestamp* de la acción y la app extrae y muestra la imagen correspondiente.

---

## ¿Cómo funciona la detección por similitud?

Ambos sistemas usan **Embeddings Vectoriales**:

1. **Detección:** Se localiza una cara en la imagen.
2. **Vectorización:** La red neuronal (VGG-Face o Facenet) genera un vector (embedding) que representa los rasgos únicos de la persona.
3. **Comparación (Distancia del Coseno):**

   * Valores cercanos a 0 → misma persona.
   * Valores cercanos a 1 → personas distintas.
   * Se utiliza un **umbral (threshold)** (ej. 0.40) para decidir si hay coincidencia.

---

## Instalación y Uso

### Prerrequisitos

* Python 3.10 o superior.
* Webcam funcional.
* API Key de Google (para el módulo de video).

### 1. Instalación de dependencias

```bash
python -m venv venv
source venv/bin/activate      # Windows: .\venv\Scripts\activate

pip install -r requirements.txt
```

*Nota: Puede ser necesario instalar `tf-keras` según la versión de TensorFlow.*

### 2. Ejecutar Variación A (Vigilancia)

```bash
cd Face_recognition_dati
python facial_recognition_realtime.py
```

*Debe existir la carpeta `known_faces` con al menos una foto.*

### 3. Ejecutar Variación B (Testing)

```bash
cd face_recognition

# Prueba con webcam
python webcam_match.py --train_dir train

# Comparar carpetas
python main.py --train_dir train --test_dir test
```

### 4. Ejecutar Análisis de Video

```bash
streamlit run video_analysis/app.py
```

---

## Tecnologías Utilizadas

* DeepFace: Framework de reconocimiento facial.
* OpenCV (cv2): Procesamiento de imagen y captura de cámara.
* Google Gemini API: Procesamiento multimodal de video.
* Streamlit: Interfaz web.
* TensorFlow/Keras: Backend de deep learning.

---

**Autor:** Lostulachi Team
**Licencia:** MIT

```

Si quieres que genere también una versión **aún más minimalista**, **con tabla de contenidos**, o **con enlaces automáticos**, puedo hacerlo.
```
