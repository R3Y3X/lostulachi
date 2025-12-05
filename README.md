# 🎬 Sistema de Análisis de Video y Reconocimiento Facial

Proyecto completo que combina análisis de video multimodal con Google Gemini y reconocimiento facial en tiempo real.

## 📁 Estructura del Proyecto

```
prueba_gemini/
├── video_analysis/          # 🎬 Análisis de video con Gemini
│   ├── app.py              # Aplicación Streamlit principal
│   └── README.md           # Documentación del módulo
│
├── face_recognition/        # 👤 Sistema de reconocimiento facial
│   ├── README.md           # Documentación del módulo
│   └── (scripts a implementar)
│
├── shared/                  # 🔧 Utilidades compartidas
│   └── utils/
│       ├── config_loader.py # Carga de configuración
│       └── __init__.py
│
├── config/                  # ⚙️ Configuración
│   ├── config.example.py   # Ejemplo de configuración
│   └── __init__.py
│
├── data/                    # 💾 Datos del proyecto
│   ├── models/             # Modelos de ML
│   ├── images/             # Imágenes de personas
│   └── database/           # Bases de datos
│
├── requirements.txt        # Dependencias del proyecto
├── README.md              # Este archivo
└── .env                   # Variables de entorno (crear manualmente)
```

## 🚀 Inicio Rápido

### 1. Instalación

```bash
# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configuración

Crea un archivo `.env` en la raíz del proyecto:

```
GEMINI_API_KEY=tu_clave_api_aqui
```

### 3. Ejecutar Aplicaciones

#### Análisis de Video (Streamlit)

```bash
streamlit run video_analysis/app.py
```

#### Reconocimiento Facial

```bash
# (Próximamente)
python face_recognition/main.py
```

## 📦 Módulos

### 🎬 Video Analysis (`video_analysis/`)

Aplicación web para análisis de video usando Google Gemini 2.5 Pro:
- Carga de videos
- Análisis multimodal con IA
- Extracción de frames en timestamps específicos
- Interfaz web con Streamlit

**Ver más:** [video_analysis/README.md](video_analysis/README.md)

### 👤 Face Recognition (`face_recognition/`)

Sistema de reconocimiento facial (en desarrollo):
- Registro de personas en base de datos
- Detección en tiempo real desde cámara
- Marcado visual: rojo para target, verde para no-target

**Ver más:** [face_recognition/README.md](face_recognition/README.md)

## 🛠️ Tecnologías

- **Python 3.10+**
- **Streamlit**: Interfaz web
- **Google Gemini 2.5 Pro**: Análisis multimodal
- **OpenCV**: Procesamiento de video e imágenes
- **python-dotenv**: Manejo de variables de entorno

## 📋 Requisitos Previos

- Python 3.10 o superior
- Cuenta de Google AI Studio con API Key
- Cámara web (para reconocimiento facial)

## 🔑 Obtener API Key de Google AI Studio

1. Ve a [Google AI Studio](https://ai.google.dev/)
2. Inicia sesión con tu cuenta de Google
3. Genera una nueva API Key
4. Agrega la clave al archivo `.env`:

```
GEMINI_API_KEY=tu_clave_aqui
```

## 📝 Notas

- El archivo `.env` no debe subirse a repositorios públicos
- Los videos se procesan temporalmente durante el análisis
- La base de datos de reconocimiento facial se almacena en `data/database/`

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

---

**Desarrollado con ❤️ usando Google Gemini, Streamlit y OpenCV**
