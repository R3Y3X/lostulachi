"""
Sistema de Reconocimiento Facial en Tiempo Real
Utiliza DeepFace y OpenCV para detectar y reconocer rostros desde la webcam.
"""

import cv2
import os
import time
from datetime import datetime
from threading import Thread, Lock
from deepface import DeepFace
import numpy as np


class FacialRecognitionSystem:
    """
    Clase principal que gestiona el sistema de reconocimiento facial en tiempo real.
    """
    
    def __init__(self, known_faces_dir="known_faces", camera_source=0, 
                 model_name="VGG-Face", recognition_interval=30, cooldown_seconds=5):
        """
        Inicializa el sistema de reconocimiento facial.
        
        Args:
            known_faces_dir: Directorio donde se almacenan las caras conocidas
            camera_source: Índice de la cámara a usar (0 = webcam predeterminada)
            model_name: Modelo de DeepFace a usar ('VGG-Face', 'Facenet512', etc.)
            recognition_interval: Cada cuántos frames ejecutar reconocimiento (30 = ~1 segundo)
            cooldown_seconds: Tiempo de espera antes de guardar otra cara desconocida
        """
        self.known_faces_dir = known_faces_dir
        self.camera_source = camera_source
        self.model_name = model_name
        self.recognition_interval = recognition_interval
        self.cooldown_seconds = cooldown_seconds
        
        # Crear directorio de caras conocidas si no existe
        self._create_known_faces_directory()
        
        # Variables de control
        self.frame_count = 0
        self.last_unknown_save_time = {}  # Diccionario por rostro para cooldown individual
        self.current_frame = None
        self.frame_lock = Lock()
        self.running = False
        
        # Almacenar resultados de reconocimiento para visualización continua
        self.face_results = {}  # {face_id: (x, y, w, h, is_known, name, color, label)}
        self.detected_faces = []  # Lista de rostros detectados (para mostrar mientras se procesa)
        self.face_id_counter = 0  # Contador para IDs únicos de rostros
        self.face_tracking = {}  # Tracking de rostros por posición
        
        # Inicializar cámara
        self.cap = cv2.VideoCapture(self.camera_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {self.camera_source}")
        
        # Configurar resolución de la cámara (opcional, para mejor rendimiento)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"Sistema inicializado correctamente.")
        print(f"Modelo: {self.model_name}")
        print(f"Directorio de caras conocidas: {self.known_faces_dir}")
        print(f"Intervalo de reconocimiento: cada {self.recognition_interval} frames")
        print(f"Cooldown para guardar desconocidos: {self.cooldown_seconds} segundos")
    
    def _create_known_faces_directory(self):
        """Crea el directorio de caras conocidas si no existe."""
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
            print(f"Directorio '{self.known_faces_dir}' creado exitosamente.")
    
    def _detect_faces(self, frame):
        """
        Detecta rostros en el frame usando OpenCV.
        
        Args:
            frame: Frame de video (numpy array)
            
        Returns:
            Lista de tuplas (x, y, w, h) con las coordenadas de los rostros detectados
        """
        # Convertir a escala de grises para la detección (más rápido)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Cargar el clasificador Haar Cascade para detección de rostros
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Detectar rostros
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces
    
    def _recognize_face(self, face_roi):
        """
        Reconoce un rostro usando DeepFace.
        
        Args:
            face_roi: Región de interés (recorte) del rostro (numpy array)
            
        Returns:
            Tupla (is_known, name, distance) where:
            - is_known: True if match found, False otherwise
            - name: Best match filename or None
            - distance: Similarity distance (lower is better) or None
        """
        try:
            # Verificar si hay imágenes en la carpeta known_faces
            image_files = [f for f in os.listdir(self.known_faces_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if len(image_files) == 0:
                # No hay imágenes conocidas, considerar como desconocido
                return False, None, None
            
            # Guardar temporalmente el rostro para DeepFace
            temp_path = "temp_face.jpg"
            cv2.imwrite(temp_path, face_roi)
            
            # Buscar coincidencias en la base de datos
            # DeepFace.find busca en el directorio especificado
            df = DeepFace.find(
                img_path=temp_path,
                db_path=self.known_faces_dir,
                model_name=self.model_name,
                enforce_detection=False,  # No fallar si no detecta rostro
                silent=True  # No mostrar logs
            )
            
            # Eliminar archivo temporal
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Verificar si se encontró alguna coincidencia
            # DeepFace.find devuelve una lista de DataFrames, uno por cada imagen en la DB
            if df is not None and isinstance(df, list) and len(df) > 0:
                # Obtener el primer DataFrame (resultados de la búsqueda)
                result_df = df[0]
                
                # Verificar que el DataFrame no esté vacío
                if result_df is not None and len(result_df) > 0:
                    # Obtener el nombre del archivo más similar
                    best_match_path = result_df.iloc[0]['identity']
                    name = os.path.basename(best_match_path).split('.')[0]
                    
                    # Verificar distancia (threshold de similitud)
                    # Si la distancia es muy alta, considerar desconocido
                    distance = result_df.iloc[0]['distance']
                    
                    # Threshold ajustado según el modelo
                    # VGG-Face: ~0.4-0.6, Facenet512: ~0.3-0.4
                    threshold = 0.6 if self.model_name == "VGG-Face" else 0.3
                    
                    if distance < threshold:
                        return True, name, distance
                    else:
                        return False, name, distance
            
            # No se encontraron coincidencias
            return False, None, None
                
        except Exception as e:
            # Si hay error, considerar como desconocido
            print(f"Error en reconocimiento: {str(e)}")
            # Limpiar archivo temporal si existe
            if os.path.exists("temp_face.jpg"):
                os.remove("temp_face.jpg")
            return False, None, None
    
    def _get_face_id(self, x, y, w, h):
        """
        Obtiene o asigna un ID único a un rostro basado en su posición.
        Usa un sistema de matching por proximidad para rastrear el mismo rostro.
        
        Args:
            x, y, w, h: Coordenadas del rostro
            
        Returns:
            ID único del rostro
        """
        # Calcular centro del rostro
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Buscar rostro existente cercano (dentro de 50 píxeles)
        threshold_distance = 50
        
        # Acceder a face_results con lock para thread-safety
        with self.frame_lock:
            for face_id, (old_x, old_y, old_w, old_h, *_) in self.face_results.items():
                old_center_x = old_x + old_w // 2
                old_center_y = old_y + old_h // 2
                
                distance = ((center_x - old_center_x)**2 + (center_y - old_center_y)**2)**0.5
                
                if distance < threshold_distance:
                    return face_id
        
        # Si no se encuentra un rostro cercano, crear nuevo ID
        self.face_id_counter += 1
        return self.face_id_counter
    
    def _save_unknown_face(self, face_roi, face_id):
        """
        Guarda un rostro desconocido en el directorio known_faces.
        Implementa cooldown por rostro individual para evitar guardar múltiples veces.
        
        Args:
            face_roi: Región de interés (recorte) del rostro (numpy array)
            face_id: ID único del rostro
        """
        current_time = time.time()
        
        # Verificar cooldown específico para este rostro
        if face_id in self.last_unknown_save_time:
            if current_time - self.last_unknown_save_time[face_id] < self.cooldown_seconds:
                return
        
        # Generar nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"unknown_{timestamp}_face{face_id}.jpg"
        filepath = os.path.join(self.known_faces_dir, filename)
        
        # Guardar el rostro
        cv2.imwrite(filepath, face_roi)
        print(f"Rostro desconocido guardado: {filename} (ID: {face_id})")
        
        # Actualizar tiempo del último guardado para este rostro
        self.last_unknown_save_time[face_id] = current_time
    
    def _process_single_face(self, frame, x, y, w, h, face_id):
        """
        Procesa el reconocimiento de un solo rostro.
        Esta función puede ejecutarse en paralelo para múltiples rostros.
        
        Args:
            frame: Frame completo de video
            x, y, w, h: Coordenadas del rostro
            face_id: ID único del rostro
            
        Returns:
        Returns:
            Tupla (face_id, x, y, w, h, is_known, name, color, label, distance)
        """
        try:
            # Extraer región de interés (ROI) del rostro
            face_roi = frame[y:y+h, x:x+w]
            
            # Validar que el ROI no esté vacío
            if face_roi.size == 0:
                return None
            
            # Reconocer el rostro
            is_known, name, distance = self._recognize_face(face_roi)
            
            # Determinar color y etiqueta
            if is_known:
                # Caso A: Rostro conocido - Cuadro ROJO
                color = (0, 0, 255)  # ROJO en BGR
                label = f"Conocido: {name}"
                print(f"✓ Rostro ID {face_id} reconocido: {name}")
            else:
                # Caso B: Rostro desconocido - Cuadro AMARILLO
                color = (0, 255, 255)  # AMARILLO en BGR
                label = f"Desconocido (ID: {face_id})"
                
                # Guardar automáticamente el rostro desconocido (con cooldown por rostro)
                self._save_unknown_face(face_roi, face_id)
            
            return (face_id, x, y, w, h, is_known, name, color, label, distance)
        
        except Exception as e:
            print(f"Error procesando rostro ID {face_id}: {str(e)}")
            return None
    
    def _process_frame_recognition(self, frame):
        """
        Procesa el reconocimiento facial en un frame para múltiples rostros simultáneamente.
        Esta función se ejecuta en un hilo separado para no bloquear el video.
        
        Args:
            frame: Frame de video a procesar
        """
        try:
            # Detectar rostros en el frame
            faces = self._detect_faces(frame)
            
            # Si no se detectan rostros, limpiar resultados
            if len(faces) == 0:
                with self.frame_lock:
                    self.face_results = {}
                    self.detected_faces = []
                return
            
            # Actualizar rostros detectados (para mostrar mientras se procesa)
            with self.frame_lock:
                self.detected_faces = faces.tolist() if len(faces) > 0 else []
            
            # Procesar cada rostro detectado
            # Primero asignar IDs a cada rostro
            face_ids = []
            for (x, y, w, h) in faces:
                face_id = self._get_face_id(x, y, w, h)
                face_ids.append((face_id, x, y, w, h))
            
            # Procesar todos los rostros (secuencialmente por ahora, pero estructurado para paralelización)
            new_results = {}
            for face_id, x, y, w, h in face_ids:
                result = self._process_single_face(frame, x, y, w, h, face_id)
                if result is not None:
                    face_id_result, x, y, w, h, is_known, name, color, label, distance = result
                    new_results[face_id_result] = (x, y, w, h, is_known, name, color, label, distance)
            
            # Actualizar resultados con lock
            with self.frame_lock:
                # Mantener resultados de rostros que ya no están visibles por un tiempo
                # (para suavizar la transición cuando alguien se mueve)
                self.face_results = new_results
                
                # Limpiar cooldowns de rostros que ya no están presentes
                active_ids = set(new_results.keys())
                self.last_unknown_save_time = {
                    fid: time_val for fid, time_val in self.last_unknown_save_time.items()
                    if fid in active_ids
                }
        
        except Exception as e:
            print(f"Error procesando frame: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """
        Ejecuta el bucle principal de reconocimiento facial en tiempo real.
        """
        self.running = True
        recognition_thread = None
        
        print("\nIniciando sistema de reconocimiento facial...")
        print("Presiona 'q' para salir\n")
        
        try:
            while self.running:
                # Leer frame de la cámara
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error leyendo frame de la cámara")
                    break
                
                # Actualizar frame actual
                with self.frame_lock:
                    self.current_frame = frame.copy()
                    display_frame = frame.copy()
                
                # Incrementar contador de frames
                self.frame_count += 1
                
                # Ejecutar reconocimiento cada N frames o en un hilo separado
                if self.frame_count % self.recognition_interval == 0:
                    # Si hay un hilo de reconocimiento anterior, esperar a que termine
                    if recognition_thread is not None and recognition_thread.is_alive():
                        pass  # Saltar este ciclo si aún está procesando
                    else:
                        # Crear nuevo hilo para procesamiento
                        recognition_thread = Thread(
                            target=self._process_frame_recognition,
                            args=(frame.copy(),)
                        )
                        recognition_thread.daemon = True
                        recognition_thread.start()
                
                # Dibujar cuadros y etiquetas de rostros detectados
                with self.frame_lock:
                    # Obtener IDs de rostros ya procesados
                    processed_face_positions = {
                        (x, y, w, h): face_id 
                        for face_id, (x, y, w, h, *_) in self.face_results.items()
                    }
                    
                    # Primero dibujar rostros detectados pero aún no procesados (gris)
                    for (x, y, w, h) in self.detected_faces:
                        # Verificar si este rostro ya está procesado
                        is_processed = any(
                            abs(x - px) < 30 and abs(y - py) < 30 
                            for px, py, pw, ph in processed_face_positions.keys()
                        )
                        
                        if not is_processed:
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (128, 128, 128), 2)
                            cv2.putText(
                                display_frame,
                                "Procesando...",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 255),
                                2
                            )
                    
                    # Dibujar todos los rostros detectados con sus resultados
                    for face_id, (x, y, w, h, is_known, name, color, label, distance) in self.face_results.items():
                        # Dibujar cuadro alrededor del rostro (más grueso para mejor visibilidad)
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 3)
                        
                        # Dibujar ID del rostro en la esquina superior izquierda del cuadro
                        id_text = f"ID: {face_id}"
                        cv2.putText(
                            display_frame,
                            id_text,
                            (x + 5, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1
                        )
                        
                        # Dibujar fondo para el texto principal (para mejor legibilidad)
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                        )
                        # Asegurar que el fondo no se salga de la imagen
                        text_y = max(y - 5, text_height + 5)
                        cv2.rectangle(
                            display_frame,
                            (x, text_y - text_height - 5),
                            (x + text_width + 10, text_y + 5),
                            color,
                            -1
                        )
                        
                        # Mostrar etiqueta sobre el cuadro
                        cv2.putText(
                            display_frame,
                            label,
                            (x + 5, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),  # Texto blanco
                            2
                        )
                        
                        # Mostrar similitud en la esquina inferior derecha
                        if name is not None and distance is not None:
                            sim_text = f"{name} ({distance:.2f})"
                            (sim_w, sim_h), _ = cv2.getTextSize(sim_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            
                            # Posición inferior derecha del cuadro
                            sim_x = x + w - sim_w - 5
                            sim_y = y + h - 5
                            
                            # Asegurar que no se salga por la izquierda
                            if sim_x < x: 
                                sim_x = x + 5
                            
                            # Fondo para el texto de similitud
                            cv2.rectangle(
                                display_frame,
                                (sim_x - 2, sim_y - sim_h - 2),
                                (sim_x + sim_w + 2, sim_y + 2),
                                color,
                                -1
                            )
                            
                            # Texto de similitud
                            cv2.putText(
                                display_frame,
                                sim_text,
                                (sim_x, sim_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1
                            )
                
                # Mostrar información en pantalla
                cv2.putText(
                    display_frame,
                    f"Frame: {self.frame_count} | Presiona 'q' para salir",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                
                # Mostrar leyenda de colores y contador de rostros
                with self.frame_lock:
                    num_faces = len(self.face_results)
                
                legend_text = f"ROJO: Conocido | AMARILLO: Desconocido | Rostros detectados: {num_faces}"
                cv2.putText(
                    display_frame,
                    legend_text,
                    (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                
                # Mostrar frame
                cv2.imshow('Reconocimiento Facial en Tiempo Real', display_frame)
                
                # Salir si se presiona 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nDeteniendo sistema...")
                    self.running = False
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupción del usuario. Deteniendo sistema...")
            self.running = False
        
        finally:
            # Limpiar recursos
            self.cleanup()
    
    def cleanup(self):
        """Libera recursos del sistema."""
        print("Liberando recursos...")
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        print("Sistema detenido correctamente.")


def main():
    """
    Función principal para ejecutar el sistema de reconocimiento facial.
    """
    try:
        # Crear instancia del sistema
        # Puedes cambiar los parámetros aquí:
        # - model_name: 'VGG-Face', 'Facenet512', 'OpenFace', etc.
        # - recognition_interval: frames entre reconocimientos (30 = ~1 segundo a 30fps)
        # - cooldown_seconds: segundos de espera entre guardados de desconocidos
        
        system = FacialRecognitionSystem(
            known_faces_dir="known_faces",
            camera_source=0,
            model_name="VGG-Face",  # Cambiar a 'Facenet512' para mayor precisión
            recognition_interval=30,  # Procesar cada 30 frames
            cooldown_seconds=5  # 5 segundos entre guardados
        )
        
        # Ejecutar sistema
        system.run()
    
    except Exception as e:
        print(f"Error fatal: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
