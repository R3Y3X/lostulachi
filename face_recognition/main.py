import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from deepface import DeepFace


# ============================
#  MÉTRICAS: DISTANCIA / SIMILITUD
# ============================

def cosine_distance(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Distancia de coseno entre dos embeddings.
    0  -> vectores muy parecidos
    ~1 -> diferentes
    """
    v1 = np.array(embedding1, dtype=float)
    v2 = np.array(embedding2, dtype=float)

    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 1.0

    cos_sim = float(np.dot(v1, v2) / denom)
    return 1.0 - cos_sim  # distancia = 1 - similitud


def distance_to_similarity(distance: float) -> float:
    """
    Convierte una distancia en un % de similitud [0, 100].

    Usamos:
        similitud = max(0, min(1, 1 - distance)) * 100

    Ejemplos:
        distance = 0.0 -> 100%
        distance = 0.2 -> 80%
        distance = 0.5 -> 50%
    """
    sim = 1.0 - distance
    sim = max(0.0, min(1.0, sim))
    return sim * 100.0


# ============================
#  EMBEDDINGS
# ============================

def compute_embedding(
    image_path: str,
    model_name: str = "Facenet",
    enforce_detection: bool = False,
) -> List[float]:
    """
    Obtiene el embedding de una imagen usando DeepFace.represent.
    Por defecto enforce_detection=False para ser más tolerante.
    """
    reps = DeepFace.represent(
        img_path=image_path,
        model_name=model_name,
        enforce_detection=enforce_detection,
    )
    return reps[0]["embedding"]


def load_embeddings_from_dir(
    dir_path: str,
    model_name: str = "Facenet",
    enforce_detection: bool = False,
) -> Dict[Path, List[float]]:
    """
    Recorre una carpeta y calcula embeddings para todas las imágenes válidas.
    Ignora archivos que fallen (no se cargan o no se detecta cara).
    """
    root = Path(dir_path)
    if not root.is_dir():
        raise NotADirectoryError(f"No es una carpeta válida: {root}")

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".PNG"}
    image_paths = [
        p for p in sorted(root.iterdir())
        if p.is_file() and p.suffix in exts
    ]

    if not image_paths:
        raise FileNotFoundError(f"No se encontraron imágenes en {root}")

    embeddings: Dict[Path, List[float]] = {}
    print(f"[INFO] Calculando embeddings para imágenes en {root}...")
    for img_path in image_paths:
        try:
            print(f"   - {img_path}")
            emb = compute_embedding(
                image_path=str(img_path),
                model_name=model_name,
                enforce_detection=enforce_detection,
            )
            embeddings[img_path] = emb
        except Exception as e:
            print(f"[WARN] No se pudo procesar {img_path}: {e}")

    if not embeddings:
        raise ValueError(
            f"No se pudo generar ningún embedding para imágenes en {root}. "
            "Revisa los archivos o considera ejecutar sin --enforce_detection."
        )

    print(f"[INFO] Embeddings generados: {len(embeddings)} imágenes válidas en {root}")
    return embeddings


# ============================
#  COMPARAR TRAIN vs TEST
# ============================

def compare_train_test(
    train_embeddings: Dict[Path, List[float]],
    test_embeddings: Dict[Path, List[float]],
    similarity_threshold: float,
) -> List[Tuple[Path, Path, float, float, bool]]:
    """
    Para cada imagen de test, busca la mejor coincidencia en train.

    Devuelve lista de tuplas:
        (test_path, best_train_path, best_distance, best_similarity, match_bool)
    """
    results: List[Tuple[Path, Path, float, float, bool]] = []

    for test_path, test_emb in test_embeddings.items():
        best_train_path: Path | None = None
        best_dist: float = float("inf")
        best_sim: float = 0.0

        for train_path, train_emb in train_embeddings.items():
            dist = cosine_distance(test_emb, train_emb)
            sim = distance_to_similarity(dist)

            if sim > best_sim:
                best_sim = sim
                best_dist = dist
                best_train_path = train_path

        match = best_sim >= similarity_threshold
        # En teoría best_train_path no debería ser None porque hay al menos 1 embedding de train
        results.append((test_path, best_train_path, best_dist, best_sim, match))

    return results


# ============================
#  MAIN (CLI)
# ============================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compara todas las imágenes de una carpeta de test contra "
            "todas las imágenes de una carpeta de train usando DeepFace."
        )
    )

    parser.add_argument(
        "--train_dir",
        required=True,
        help="Carpeta con imágenes de entrenamiento (ej: train/).",
    )
    parser.add_argument(
        "--test_dir",
        required=True,
        help="Carpeta con imágenes de test (ej: test/).",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=80.0,
        help="Umbral de similitud en % para considerar que hay match (default: 80.0).",
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
            "Si se pasa este flag, DeepFace exigirá detección de cara en todas las imágenes. "
            "Si no se pasa, se usará enforce_detection=False (más tolerante)."
        ),
    )

    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)

    if not train_dir.is_dir():
        raise NotADirectoryError(f"La carpeta de entrenamiento no existe: {train_dir}")
    if not test_dir.is_dir():
        raise NotADirectoryError(f"La carpeta de test no existe: {test_dir}")

    enforce = args.enforce_detection

    # 1. Cargar embeddings de train
    train_embeddings = load_embeddings_from_dir(
        dir_path=str(train_dir),
        model_name=args.model,
        enforce_detection=enforce,
    )

    # 2. Cargar embeddings de test
    test_embeddings = load_embeddings_from_dir(
        dir_path=str(test_dir),
        model_name=args.model,
        enforce_detection=enforce,
    )

    # 3. Comparar
    print("\n[INFO] Comparando embeddings test vs train...")
    results = compare_train_test(
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        similarity_threshold=args.similarity_threshold,
    )

    # 4. Mostrar resultados
    print("\n========== RESULTADOS ==========")
    print(
        f"{'TEST IMAGE':30s} | {'BEST TRAIN':30s} | {'SIMILARIDAD (%)':15s} | MATCH"
    )
    print("-" * 100)

    for test_path, best_train_path, best_dist, best_sim, match in results:
        test_name = test_path.name
        train_name = best_train_path.name if best_train_path is not None else "N/A"
        match_str = "✔️" if match else "❌"
        print(
            f"{test_name:30s} | {train_name:30s} | {best_sim:15.2f} | {match_str}"
        )

    print("\n[INFO] Umbral de similitud aplicado:", args.similarity_threshold, "%")
    print("[INFO] Total de imágenes de test evaluadas:", len(results))


if __name__ == "__main__":
    main()
