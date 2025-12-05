"""
Aplicación Web MVP para Análisis de Video Multimodal
Usando Google Gemini 2.5 Pro, Streamlit y OpenCV

Autor: Generado para análisis de video con IA
"""

import streamlit as st
import google.generativeai as genai
import cv2
import os
import json
import time
import tempfile
import re
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde archivo .env
root_dir = Path(__file__).parent.parent
load_dotenv(dotenv_path=root_dir / '.env')

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Análisis de Video Multimodal",
    page_icon="🎬",
    layout="wide"
)

# Título de la aplicación
st.title("🎬 Análisis de Video Multimodal con Gemini")
st.markdown("Carga un video y haz preguntas sobre su contenido. La IA identificará acciones y te mostrará el momento exacto.")

# Inicializar variables de sesión
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'video_path' not in st.session_state:
    st.session_state.video_path = None

# Configurar API Key de Google Gemini
API_KEY = os.getenv('GEMINI_API_KEY')

if not API_KEY:
    st.error("⚠️ Error: No se encontró la variable de entorno GEMINI_API_KEY.")
    st.info("Por favor, crea un archivo `.env` en la raíz del proyecto con: `GEMINI_API_KEY=tu_clave_aqui`")
    st.stop()
else:
    try:
        genai.configure(api_key=API_KEY)
        st.sidebar.success("✅ API Key configurada correctamente")
    except Exception as e:
        st.error(f"Error al configurar la API Key: {str(e)}")
        st.stop()

# Widgets de la interfaz de usuario
uploaded_file = st.file_uploader(
    "📁 Carga un archivo de video",
    type=["mp4", "mov", "avi", "mkv"],
    help="Formatos soportados: MP4, MOV, AVI, MKV"
)

user_prompt = st.text_input(
    "💬 Escribe tu prompt de texto",
    placeholder="Ejemplo: ¿Qué acción realiza la persona?",
    help="Describe qué quieres que la IA analice en el video"
)

# Botón para procesar
process_button = st.button("🚀 Procesar Video", type="primary", use_container_width=True)

# Función para convertir timestamp MM:SS a segundos totales
def timestamp_to_seconds(timestamp_str):
    """
    Convierte un timestamp en formato MM:SS a segundos totales.
    
    Args:
        timestamp_str: String en formato "MM:SS" o "M:SS"
    
    Returns:
        int: Segundos totales
    """
    try:
        parts = timestamp_str.split(':')
        if len(parts) != 2:
            raise ValueError("Formato de timestamp inválido")
        minutes = int(parts[0])
        seconds = int(parts[1])
        return minutes * 60 + seconds
    except Exception as e:
        st.error(f"Error al parsear timestamp '{timestamp_str}': {str(e)}")
        return None

# Función para extraer frame del video usando OpenCV
def extract_frame_from_video(video_path, timestamp_seconds):
    """
    Extrae un frame específico del video basado en el timestamp.
    
    Args:
        video_path: Ruta al archivo de video
        timestamp_seconds: Tiempo en segundos donde extraer el frame
    
    Returns:
        numpy.ndarray: Frame en formato RGB, o None si hay error
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Error: No se pudo abrir el archivo de video")
            return None
        
        # Obtener FPS del video para cálculo más preciso
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp_seconds * fps)
        
        # Establecer posición en milisegundos
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)
        
        # Leer el frame
        ret, frame = cap.read()
        
        if not ret:
            st.warning(f"No se pudo leer el frame en el segundo {timestamp_seconds}")
            cap.release()
            return None
        
        # Convertir de BGR (OpenCV) a RGB (para Streamlit)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        cap.release()
        return frame_rgb
    
    except Exception as e:
        st.error(f"Error al extraer frame: {str(e)}")
        return None

# Función para parsear respuesta JSON de Gemini
def parse_gemini_response(response_text):
    """
    Intenta extraer JSON de la respuesta de Gemini.
    La respuesta puede venir envuelta en markdown o texto plano.
    
    Args:
        response_text: Texto de respuesta de Gemini
    
    Returns:
        dict: Diccionario con 'accion' y 'timestamp', o None si hay error
    """
    try:
        # Intentar encontrar JSON en el texto (puede estar en bloques de código)
        # Buscar patrón JSON en el texto
        json_match = re.search(r'\{[^{}]*"accion"[^{}]*"timestamp"[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        
        # Si no se encuentra, intentar parsear todo el texto como JSON
        return json.loads(response_text)
    
    except json.JSONDecodeError:
        # Si falla, intentar extraer información manualmente
        st.warning("No se pudo parsear JSON automáticamente. Intentando extracción manual...")
        try:
            # Buscar timestamp en formato MM:SS
            timestamp_match = re.search(r'(\d{1,2}):(\d{2})', response_text)
            if timestamp_match:
                timestamp = timestamp_match.group(0)
                # Buscar descripción de acción (texto antes del timestamp o en líneas cercanas)
                accion = "Acción identificada en el video"
                return {'accion': accion, 'timestamp': timestamp}
        except Exception as e:
            st.error(f"Error al parsear respuesta: {str(e)}")
            return None
    
    except Exception as e:
        st.error(f"Error inesperado al parsear respuesta: {str(e)}")
        return None

# Procesamiento principal
if process_button and uploaded_file is not None and user_prompt:
    
    # Guardar video temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_video_path = tmp_file.name
    
    st.session_state.video_path = temp_video_path
    
    # Mostrar progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Paso 1: Subir video a Google AI Studio
        status_text.text("📤 Subiendo video a Google AI Studio...")
        progress_bar.progress(10)
        
        uploaded_video = genai.upload_file(path=temp_video_path)
        st.success(f"✅ Video subido. ID: {uploaded_video.name}")
        progress_bar.progress(30)
        
        # Paso 2: Polling - Esperar hasta que el estado sea ACTIVE
        status_text.text("⏳ Esperando procesamiento del video en Google AI Studio...")
        
        max_attempts = 60  # Máximo 5 minutos (60 * 5 segundos)
        attempt = 0
        
        while uploaded_video.state.name == "PROCESSING":
            if attempt >= max_attempts:
                st.error("⏱️ Tiempo de espera excedido. El video está tardando demasiado en procesarse.")
                st.stop()
            
            time.sleep(5)  # Esperar 5 segundos entre verificaciones
            uploaded_video = genai.get_file(uploaded_video.name)
            attempt += 1
            
            # Mostrar progreso
            progress_value = 30 + (attempt * 50 // max_attempts)
            progress_bar.progress(min(progress_value, 80))
            status_text.text(f"⏳ Procesando... (Intento {attempt}/{max_attempts})")
        
        if uploaded_video.state.name != "ACTIVE":
            st.error(f"❌ Error: El video no pudo ser procesado. Estado: {uploaded_video.state.name}")
            st.stop()
        
        st.success("✅ Video procesado y listo para análisis")
        progress_bar.progress(80)
        status_text.text("🤖 Realizando análisis con Gemini 2.5 Pro...")
        
        # Paso 3: Realizar inferencia con Gemini 2.5 Pro
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        # Construir prompt del sistema con instrucciones para formato JSON
        system_prompt = """Identifica la acción principal en el video y provee el timestamp exacto donde ocurre de forma más clara en formato MM:SS. 

IMPORTANTE: Responde ÚNICAMENTE con un JSON válido en el siguiente formato:
{
  "accion": "descripción detallada de la acción",
  "timestamp": "MM:SS"
}

No incluyas texto adicional fuera del JSON."""
        
        full_prompt = f"{system_prompt}\n\nPregunta del usuario: {user_prompt}"
        
        # Generar contenido
        response = model.generate_content([uploaded_video, full_prompt])
        response_text = response.text
        
        progress_bar.progress(90)
        status_text.text("📊 Procesando resultados...")
        
        # Paso 4: Parsear respuesta JSON
        result_json = parse_gemini_response(response_text)
        
        if not result_json:
            st.error("❌ No se pudo extraer información estructurada de la respuesta de Gemini.")
            st.text_area("Respuesta completa de Gemini:", response_text, height=200)
            st.stop()
        
        accion = result_json.get('accion', 'No especificada')
        timestamp_str = result_json.get('timestamp', '00:00')
        
        progress_bar.progress(95)
        
        # Paso 5: Extraer frame del video usando OpenCV
        timestamp_seconds = timestamp_to_seconds(timestamp_str)
        
        if timestamp_seconds is not None:
            status_text.text("📸 Extrayendo frame del momento exacto...")
            frame_image = extract_frame_from_video(temp_video_path, timestamp_seconds)
        else:
            frame_image = None
        
        progress_bar.progress(100)
        status_text.text("✅ Procesamiento completado")
        time.sleep(0.5)  # Pequeña pausa para mostrar el 100%
        
        # Limpiar elementos de progreso
        progress_bar.empty()
        status_text.empty()
        
        # Paso 6: Mostrar resultados
        st.markdown("---")
        st.header("📊 Resultados del Análisis")
        
        # Crear columnas para mejor layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎬 Video Original")
            st.video(temp_video_path)
        
        with col2:
            st.subheader("📝 Descripción de la Acción")
            st.info(f"**{accion}**")
            
            st.subheader("⏰ Timestamp")
            st.success(f"**{timestamp_str}**")
            
            if frame_image is not None:
                st.subheader("📸 Frame Extraído")
                st.image(frame_image, caption=f"Momento exacto en {timestamp_str}", use_column_width=True)
            else:
                st.warning("No se pudo extraer el frame del video.")
        
        # Mostrar respuesta completa de Gemini (expandible)
        with st.expander("🔍 Ver respuesta completa de Gemini"):
            st.text(response_text)
        
        # Limpiar archivo temporal (opcional - comentado para debugging)
        # os.unlink(temp_video_path)
        
        st.session_state.video_processed = True
        
    except genai.types.StopCandidateException as e:
        st.error(f"❌ Error en la generación de contenido: {str(e)}")
        st.info("Esto puede deberse a contenido inapropiado o restricciones de seguridad.")
    
    except Exception as e:
        st.error(f"❌ Error durante el procesamiento: {str(e)}")
        st.exception(e)
        
        # Limpiar archivo temporal en caso de error
        if os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
            except:
                pass

elif process_button:
    if uploaded_file is None:
        st.warning("⚠️ Por favor, carga un archivo de video primero.")
    if not user_prompt:
        st.warning("⚠️ Por favor, escribe un prompt de texto.")

# Información adicional en la sidebar
with st.sidebar:
    st.markdown("### ℹ️ Información")
    st.markdown("""
    **Cómo usar:**
    1. Carga un video (MP4, MOV, AVI, MKV)
    2. Escribe una pregunta sobre el video
    3. Haz clic en "Procesar Video"
    4. Espera el análisis (puede tardar unos minutos)
    5. Revisa los resultados
    
    **Nota:** El video se sube a Google AI Studio para su procesamiento.
    """)
    
    st.markdown("### 🔧 Tecnologías")
    st.markdown("""
    - **Streamlit**: Interfaz web
    - **Google Gemini 2.5 Pro**: Análisis multimodal
    - **OpenCV**: Extracción de frames
    - **Python 3.12.3**: Lenguaje base
    """)

