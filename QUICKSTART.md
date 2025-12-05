# 🚀 Guía de Inicio Rápido

## Estructura del Proyecto

```
prueba_gemini/
├── video_analysis/          # 🎬 Análisis de video con Gemini
│   └── app.py              # Ejecutar: streamlit run video_analysis/app.py
│
├── face_recognition/        # 👤 Reconocimiento facial (en desarrollo)
│   └── (scripts a implementar)
│
├── shared/                  # 🔧 Utilidades compartidas
│   └── utils/
│
├── config/                  # ⚙️ Configuración
│   └── config.example.py
│
└── data/                    # 💾 Datos
    ├── models/
    ├── images/
    └── database/
```

## ⚡ Comandos Rápidos

### 1. Análisis de Video (Streamlit)

```bash
# Desde la raíz del proyecto
streamlit run video_analysis/app.py
```

### 2. Reconocimiento Facial

```bash
# (Próximamente)
python face_recognition/main.py
```

## 📝 Configuración Inicial

1. **Crear archivo `.env`** en la raíz:
```
GEMINI_API_KEY=tu_clave_api_aqui
```

2. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

3. **Ejecutar aplicación**:
```bash
streamlit run video_analysis/app.py
```

## 📚 Más Información

- **Documentación completa**: Ver [README.md](README.md)
- **Video Analysis**: Ver [video_analysis/README.md](video_analysis/README.md)
- **Face Recognition**: Ver [face_recognition/README.md](face_recognition/README.md)

