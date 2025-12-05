"""
Archivo de configuración de ejemplo
Copia este archivo a config.py y ajusta los valores según tu entorno
"""

# Configuración de la base de datos
DATABASE_PATH = "data/database/faces.db"

# Configuración de reconocimiento facial
FACE_RECOGNITION_MODEL = "hog"  # "hog" o "cnn" (CNN es más preciso pero más lento)
TOLERANCE = 0.6  # Tolerancia para matching (menor = más estricto)

# Configuración de cámara
CAMERA_INDEX = 0  # Índice de la cámara (0 para la cámara por defecto)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Colores para los recuadros
COLOR_TARGET = (0, 0, 255)  # Rojo (BGR)
COLOR_NON_TARGET = (0, 255, 0)  # Verde (BGR)
COLOR_UNKNOWN = (255, 255, 255)  # Blanco (BGR)

# Configuración de rutas
IMAGES_DIR = "data/images"
MODELS_DIR = "data/models"

