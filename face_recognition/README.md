# 👤 Sistema de Reconocimiento Facial

Este módulo implementa un sistema de reconocimiento facial que permite:

1. **Registrar personas**: Subir imágenes de personas y guardarlas en una base de datos
2. **Marcar target**: Seleccionar una persona como "target" (objetivo)
3. **Detección en tiempo real**: Usar la cámara para detectar personas
4. **Marcado visual**:
   - **Recuadro ROJO**: Si la persona detectada es el target
   - **Recuadro VERDE**: Si la persona detectada NO es el target

## 🚀 Próximos Pasos

Este módulo está preparado para implementación. Las carpetas y estructura base están creadas.

### Estructura Propuesta

```
face_recognition/
├── README.md              # Este archivo
├── face_detector.py       # Detección de caras usando OpenCV/Face Recognition
├── face_database.py       # Gestión de base de datos de personas
├── camera_stream.py      # Stream de cámara en tiempo real
└── main.py               # Script principal para ejecutar el sistema
```

### Tecnologías Sugeridas

- **OpenCV**: Para captura de video y procesamiento de imágenes
- **face_recognition** (biblioteca de Python): Para reconocimiento facial
- **SQLite/PostgreSQL**: Para almacenar información de personas
- **Streamlit/Flask**: Para interfaz de usuario (opcional)

### Funcionalidades a Implementar

1. **Registro de Personas**
   - Subir imagen de una persona
   - Extraer características faciales (encodings)
   - Guardar en base de datos con nombre/ID

2. **Gestión de Target**
   - Listar todas las personas registradas
   - Marcar una persona como "target"
   - Cambiar el target cuando sea necesario

3. **Detección en Tiempo Real**
   - Capturar frames de la cámara
   - Detectar caras en cada frame
   - Comparar con personas registradas
   - Mostrar recuadros de colores según corresponda

