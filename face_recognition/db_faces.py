from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

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
#  MODELOS DE DATOS
# ============================

@dataclass
class Person:
    id: int
    name: str
    embedding: List[float]


class FaceDatabase:
    """
    Base de datos de rostros basada en embeddings de DeepFace.
    Puedes construirla desde:
      - Subcarpetas en train/ (una carpeta por persona)
      - O directamente desde imágenes planas agrupadas por prefijo (nombre_1.jpg, nombre_2.jpg, etc.)
    """

    def __init__(self, model_name: str = "Facenet"):
        """
        model_name:
            Modelo de DeepFace para representar las caras.
            Ejemplo: 'Facenet', 'VGG-Face', 'ArcFace', etc.
        """
        self.people: List[Person] = []
        self.model_name = model_name

    # ============================
    #  EMBEDDINGS
    # ============================

    def _embedding_from_image(self, image_path: str) -> List[float]:
        """
        Obtiene el embedding de una imagen usando DeepFace.represent.
        Si no detecta cara y enforce_detection=True, DeepFace lanza excepción.
        """
        reps = DeepFace.represent(
            img_path=image_path,
            model_name=self.model_name,
            enforce_detection=True,
        )
        return reps[0]["embedding"]

    def _add_person_from_images(self, person_id: int, name: str, image_paths: List[Path]) -> None:
        """
        Calcula embeddings para varias imágenes de la misma persona
        y guarda el promedio. Si alguna imagen falla, se ignora.
        """
        embeddings = []
        print(f"[INFO] Generando embeddings para '{name}' ({len(image_paths)} imágenes)")
        for img_path in image_paths:
            try:
                print(f"   - {img_path}")
                emb = self._embedding_from_image(str(img_path))
                embeddings.append(emb)
            except Exception as e:
                # Aquí capturamos justo el "Exception while loading ..." de DeepFace
                print(f"[WARN] No se pudo procesar {img_path}: {e}")

        if not embeddings:
            # Ninguna imagen sirvió para esa persona
            raise ValueError(
                f"No se pudo generar ningún embedding para '{name}'. "
                f"Revisa las imágenes de entrenamiento de esa persona."
            )

        mean_embedding = np.mean(np.array(embeddings, dtype=float), axis=0).tolist()
        person = Person(id=person_id, name=name, embedding=mean_embedding)
        self.people.append(person)
        print(f"[INFO] Persona '{name}' añadida a la base con {len(embeddings)} imágenes válidas.")

    # ============================
    #  MODOS DE CARGA DESDE DISCO
    # ============================

    def add_person_from_folder(self, person_id: int, name: str, folder_path: str) -> None:
        """
        Lee todas las imágenes (.jpg, .jpeg, .png) de una carpeta,
        calcula embeddings y guarda el promedio como representación de la persona.
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            raise NotADirectoryError(f"No es una carpeta válida: {folder}")

        exts = {".jpg", ".jpeg", ".png"}
        image_paths = [
            p for p in folder.iterdir()
            if p.suffix.lower() in exts and p.is_file()
        ]
        if not image_paths:
            raise FileNotFoundError(f"No se encontraron imágenes en: {folder}")

        self._add_person_from_images(person_id, name, image_paths)

    def _build_from_subdirs(self, root: Path) -> None:
        """
        Caso 1: train/ tiene subcarpetas, cada una es una persona.
        Ejemplo:
            train/
              felipe/
              luis/
              martina/
        """
        subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
        if not subdirs:
            raise FileNotFoundError(f"No se encontraron subcarpetas en {root}")

        print(f"[INFO] Construyendo base de rostros desde subcarpetas de {root}")
        self.people.clear()

        for idx, person_dir in enumerate(subdirs, start=1):
            name = person_dir.name
            self.add_person_from_folder(
                person_id=idx,
                name=name,
                folder_path=str(person_dir),
            )

        print(f"[INFO] Total de personas en la base: {len(self.people)}")

    def _build_from_flat_images(self, root: Path) -> None:
        """
        Caso 2: train/ solo tiene imágenes del tipo nombre_1.jpg, nombre_2.jpg, etc.
        Agrupamos por el prefijo antes del "_" (ej: luis_1.jpeg, luis_2.jpeg -> 'luis').
        """
        exts = {".jpg", ".jpeg", ".png"}
        image_files = [
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        ]
        if not image_files:
            raise FileNotFoundError(f"No se encontraron imágenes en {root}")

        print(f"[INFO] Construyendo base de rostros desde imágenes planas en {root}")
        groups: Dict[str, List[Path]] = {}

        for img in image_files:
            stem = img.stem  # ej: "luis_1"
            parts = stem.split("_")
            person_name = parts[0] if parts else stem
            person_name = person_name.lower()

            groups.setdefault(person_name, []).append(img)

        self.people.clear()
        for idx, (name, imgs) in enumerate(sorted(groups.items()), start=1):
            self._add_person_from_images(
                person_id=idx,
                name=name,
                image_paths=imgs,
            )

        print(f"[INFO] Total de personas en la base: {len(self.people)}")

    def build_from_train_dir(self, train_dir: str) -> None:
        """
        Decide automáticamente cómo cargar el entrenamiento:

        - Si hay subcarpetas -> las usa como clases/personas.
        - Si no hay subcarpetas pero sí imágenes -> agrupa por prefijo del nombre.
        """
        root = Path(train_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"No es una carpeta válida: {root}")

        subdirs = [d for d in root.iterdir() if d.is_dir()]

        if subdirs:
            self._build_from_subdirs(root)
        else:
            self._build_from_flat_images(root)

    # ============================
    #  COMPARACIÓN / IDENTIFICACIÓN
    # ============================

    def compare_with_all(self, image_path: str) -> List[Tuple[Person, float, float]]:
        """
        Compara la imagen dada con todas las personas en la base.

        Devuelve lista de tuplas:
            (persona, distancia, similitud%)
        """
        if not self.people:
            raise ValueError("La base de rostros está vacía. Carga entrenamiento primero.")

        query_embedding = self._embedding_from_image(image_path)

        results: List[Tuple[Person, float, float]] = []
        for person in self.people:
            dist = cosine_distance(query_embedding, person.embedding)
            sim = distance_to_similarity(dist)
            results.append((person, dist, sim))

        return results

    def identify(
        self,
        image_path: str,
        similarity_threshold: float = 80.0,
    ) -> Tuple[Optional[Person], float, float]:
        """
        Identifica la persona de una imagen dada.

        similarity_threshold:
            Umbral mínimo de similitud (%) para decir que es la misma persona.

        Devuelve:
            (mejor_persona_o_None, mejor_distancia, mejor_similitud%)
        """
        results = self.compare_with_all(image_path)
        results.sort(key=lambda x: x[1])  # orden por distancia (menor es mejor)

        best_person, best_dist, best_sim = results[0]

        if best_sim >= similarity_threshold:
            return best_person, best_dist, best_sim
        else:
            return None, best_dist, best_sim
