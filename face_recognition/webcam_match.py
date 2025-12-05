import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
from deepface import DeepFace


# ============================
#  MÉTRICAS: DISTANCIA / SIMILITUD
# ============================

def cosine_distance(embedding1: List[float], embedding2: List[float]) -> float:
    v1 = np.array(embedding1, dtype=float)
    v2 = np.array(embedding2, dtype=float)

    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 1.0

    cos_sim = float(np.dot(v1, v2) / denom)
    return 1.0 - cos_sim  # distancia = 1 - similitud


def distance_to_similarity(distance: float) -> float:
    sim = 1.0 - distance
    sim = max(0.0, min(1.0, sim))
    return sim * 100.0


# ============================
#  EMBEDDINGS DE ENTRENAMIENTO
# ============================

def compute_embedding_image(
    image_path: str,
    model_name: str = "Facenet",
    enforce_detection: bool = False,
) -> List[float]:
    reps = DeepFace.represent(
        img_path=image_path,
        model_name=model_name,
        enforce_detection=enforce_detection,
    )
    return reps[0]["embedding"]


def load_train_embeddings(
    train_dir: str,
    model_name: str = "Facenet",
    enforce_detection: bool = False,
) -> List[Dict]:
    root = Path(train_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"No es una carpeta válida: {root}")

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".PNG"}
    image_paths = [
        p for p in sorted(root.iterdir())
        if p.is_file() and p.suffix in exts
    ]

    if not image_paths:
        raise FileNotFoundError(f"No se encontraron imágenes en {root}")

    items: List[Dict] = []
    print(f"[INFO] Calculando embeddings para imágenes de entrenamiento en {root}...")
    for img_path in image_paths:
        try:
            print(f"   - {img_path}")
            emb = compute_embedding_image(
                image_path=str(img_path),
                model_name=model_name,
                enforce_detection=enforce_detection,
            )
            stem = img_path.stem
            parts = stem.split("_")
            label = parts[0].capitalize() if parts else stem.capitalize()

            items.append({
                "path": img_path,
                "label": label,
                "embedding": emb,
            })
        except Exception as e:
            print(f"[WARN] No se pudo procesar {img_path}: {e}")

    if not items:
        raise ValueError(
            f"No se pudo generar ningún embedding para imágenes en {root}. "
            "Revisa los archivos o considera ejecutar sin --enforce_detection."
        )

    print(f"[INFO] Embeddings de entrenamiento generados: {len(items)} imágenes válidas.")
    return items


# ============================
#  EMBEDDINGS DESDE FRAME
# ============================

def compute_embeddings_from_frame(
    frame: np.ndarray,
    model_name: str = "Facenet",
    enforce_detection: bool = False,
) -> List[Dict]:
    try:
        reps = DeepFace.represent(
            img_path=frame,
            model_name=model_name,
            enforce_detection=enforce_detection,
        )
    except Exception:
        return []

    if isinstance(reps, dict):
        reps = [reps]

    faces = []
    for rep in reps:
        emb = rep.get("embedding")
        area = rep.get("facial_area", None)
        if emb is None:
            continue
        faces.append({
            "embedding": emb,
            "facial_area": area,
        })

    return faces


# ============================
#  BUSCAR MEJOR MATCH
# ============================

def find_best_match(
    face_embedding: List[float],
    train_items: List[Dict],
) -> Tuple[str, Path, float]:
    best_label = "Desconocido"
    best_path: Path | None = None
    best_sim = 0.0

    for item in train_items:
        train_emb = item["embedding"]
        dist = cosine_distance(face_embedding, train_emb)
        sim = distance_to_similarity(dist)

        if sim > best_sim:
            best_sim = sim
            best_label = item["label"]
            best_path = item["path"]

    if best_path is None:
        best_path = Path("N/A")

    return best_label, best_path, best_sim


# ============================
#  APERTURA ROBUSTA DE CÁMARA
# ============================

def open_camera() -> cv2.VideoCapture:
    """
    Intenta abrir la cámara probando varias combinaciones típicas en macOS.
    """
    candidates = [
        ("Default index 0", lambda: cv2.VideoCapture(0)),
        ("AVFOUNDATION index 0", lambda: cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)),
        ("Default index 1", lambda: cv2.VideoCapture(1)),
        ("AVFOUNDATION index 1", lambda: cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)),
    ]

    for desc, ctor in candidates:
        cap = ctor()
        if cap.isOpened():
            print(f"[INFO] Cámara abierta con backend: {desc}")
            return cap
        else:
            cap.release()

    raise RuntimeError("No se pudo abrir la cámara. Revisa permisos en macOS y cierra otras apps que usen la cámara.")


# ============================
#  MAIN: CÁMARA EN VIVO
# ============================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Abre la cámara del MacBook y compara las caras en vivo "
            "contra las imágenes de entrenamiento de una carpeta."
        )
    )

    parser.add_argument(
        "--train_dir",
        required=True,
        help="Carpeta con imágenes de entrenamiento (ej: train/).",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=80.0,
        help="Umbral de similitud en % para considerar match (default: 80.0).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Facenet",
        help="Modelo de DeepFace a usar (ej: 'Facenet', 'VGG-Face', 'ArcFace').",
    )
    parser.add_argument(
        "--enforce_detection",
        action="store_true",
        help=(
            "Si se pasa este flag, DeepFace exigirá detección clara de cara. "
            "Si no se pasa, se usará enforce_detection=False (más tolerante)."
        ),
    )

    args = parser.parse_args()

    # 1. Cargar embeddings de entrenamiento
    train_items = load_train_embeddings(
        train_dir=args.train_dir,
        model_name=args.model,
        enforce_detection=args.enforce_detection,
    )

    # 2. Abrir cámara
    cap = open_camera()
    print("[INFO] Presiona 'q' en la ventana de video para salir.")

    similarity_threshold = args.similarity_threshold
    failed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            failed_frames += 1
            print(f"[WARN] No se pudo leer frame de la cámara (intento {failed_frames}).")
            if failed_frames > 30:
                print("[ERROR] Muchos frames fallidos seguidos. Cerrando.")
                break
            continue

        failed_frames = 0  # reset si logramos leer frame

        faces = compute_embeddings_from_frame(
            frame=frame,
            model_name=args.model,
            enforce_detection=args.enforce_detection,
        )

        for face in faces:
            emb = face["embedding"]
            area = face.get("facial_area", None)

            label, best_path, best_sim = find_best_match(emb, train_items)
            is_match = best_sim >= similarity_threshold
            display_label = label if is_match else "Desconocido"

            text = f"{display_label} [{best_path.name}] ({best_sim:.1f}%)"

            if area is not None:
                x = area.get("x", 0)
                y = area.get("y", 0)
                w = area.get("w", 0)
                h = area.get("h", 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    text,
                    (x, y - 10 if y - 10 > 0 else y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        info_text = f"Umbral: {similarity_threshold:.1f}% | Modelo: {args.model} | q = salir"
        cv2.putText(
            frame,
            info_text,
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Face Recognition - Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Programa terminado.")


if __name__ == "__main__":
    main()
