"""
Pipeline para construir el dataset completo de LSC, dividirlo y entrenar
el modelo de signjoey sobre los videos de ``Data/RawVideos``.

El módulo está pensado para ejecutarse desde la raíz del proyecto:

    python lsc_pipeline.py dataset --workers 4
    python lsc_pipeline.py train

La extracción de características es reanudable: cada video se guarda en
``dataset_completo/features/<hash>.npz`` y se omite si el archivo ya existe
y es válido.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# Evitar que TensorFlow/MediaPipe llenen la salida de avisos.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "Data" / "RawVideos"
MEDIAPIPE_DIR = ROOT / "mediapipe_models"
DATASET_DIR = ROOT / "dataset_completo"
FEATURES_DIR = DATASET_DIR / "features"
MODEL_DIR = ROOT / "modelos" / "lsc_full"
CONFIG_PATH = ROOT / "slt" / "configs" / "lsc_full.yaml"

FEATURE_DIM = 225
EXPECTED_VIDEOS = 2880

# Split fijo por señalante. Fue elegido para aproximar 60/20/20 por video,
# mantener las 54 glosas en los tres conjuntos y no compartir personas.
TRAIN_SIGNERS = {
    "1", "3", "4", "6", "7", "8", "9", "11", "14", "15",
    "16", "17", "18", "19", "20", "25", "26", "27", "29", "31",
}
VAL_SIGNERS = {"5", "10", "13", "23", "24", "32", "34"}
TEST_SIGNERS = {"2", "12", "21", "22", "28", "30", "33"}

MODELOS_MP = {
    "pose_landmarker_full.task": (
        "https://raw.githubusercontent.com/juanesmz/SignCapture/main/models/"
        "pose_landmarker_full.task"
    ),
    "hand_landmarker.task": (
        "https://raw.githubusercontent.com/juanesmz/SignCapture/main/models/"
        "hand_landmarker.task"
    ),
}


@dataclass
class VideoItem:
    """Un video de RawVideos y su ubicación en la caché de features."""

    name: str
    category: str
    signer: str
    gloss: str
    text: str
    path: str
    cache: str
    frames: int | None = None


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _cache_name(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:20]
    return f"{digest}.npz"


def scan_videos(raw: Path = RAW) -> list[VideoItem]:
    """Recorre RawVideos y devuelve un item por cada .avi."""

    if not raw.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de videos: {raw}")

    items: list[VideoItem] = []
    for path in sorted(raw.rglob("*.avi")):
        rel = path.relative_to(raw)
        parts = rel.parts
        if len(parts) != 5:
            raise ValueError(
                "Ruta inesperada en RawVideos: "
                f"{rel} (se esperaban 5 niveles)"
            )
        category = parts[0].removeprefix("Videos ").strip()
        signer = parts[1]
        gloss = parts[3]
        name = rel.as_posix()
        items.append(
            VideoItem(
                name=name,
                category=category,
                signer=signer,
                gloss=gloss,
                text=gloss.lower(),
                path=_relative(path),
                cache=_relative(FEATURES_DIR / _cache_name(name)),
            )
        )
    if not items:
        raise RuntimeError(f"No se encontraron videos .avi en {raw}")
    return items


def ensure_mediapipe_models() -> None:
    """Descarga los modelos .task si no están presentes."""

    import urllib.request

    MEDIAPIPE_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in MODELOS_MP.items():
        destination = MEDIAPIPE_DIR / filename
        if destination.is_file() and destination.stat().st_size > 0:
            continue
        print(f"Descargando {filename} ...")
        urllib.request.urlretrieve(url, destination)
        print(f"  guardado en {destination}")


def verify_feature_file(path: Path) -> np.ndarray | None:
    """Carga y valida un .npz de features; devuelve None si no sirve."""

    if not Path(path).is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            sign = data["sign"]
    except Exception:
        return None
    if sign.ndim != 2 or sign.shape[0] == 0 or sign.shape[1] != FEATURE_DIM:
        return None
    if not np.isfinite(sign).all():
        return None
    return np.asarray(sign, dtype=np.float32)


def _landmarks_to_array(landmarks: Any) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)


def _frame_vector(pose: Any, hands: Any) -> np.ndarray:
    """Pose (33*3) + mano izquierda (21*3) + mano derecha (21*3) = 225."""

    pose_arr = (
        _landmarks_to_array(pose)
        if pose is not None
        else np.zeros((33, 3), np.float32)
    )
    left = np.zeros((21, 3), np.float32)
    right = np.zeros((21, 3), np.float32)

    if hands:
        wrists = {"izquierda": pose_arr[15, :2], "derecha": pose_arr[16, :2]}
        used = set()
        for hand in hands:
            hand_arr = _landmarks_to_array(hand)
            wrist = hand_arr[0, :2]
            if np.allclose(wrists["izquierda"], 0) and np.allclose(
                wrists["derecha"], 0
            ):
                side = "izquierda"
            else:
                dist_left = np.linalg.norm(wrist - wrists["izquierda"])
                dist_right = np.linalg.norm(wrist - wrists["derecha"])
                side = "izquierda" if dist_left <= dist_right else "derecha"
            if side in used:
                side = "derecha" if side == "izquierda" else "izquierda"
            used.add(side)
            if side == "izquierda":
                left = hand_arr
            else:
                right = hand_arr

    return np.concatenate(
        [pose_arr.reshape(-1), left.reshape(-1), right.reshape(-1)]
    )


def extract_features(video_path: Path) -> np.ndarray:
    """Convierte un video en una matriz (frames, 225) con MediaPipe."""

    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    ensure_mediapipe_models()

    pose_options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(MEDIAPIPE_DIR / "pose_landmarker_full.task")
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(MEDIAPIPE_DIR / "hand_landmarker.task")
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        pose_landmarker.close()
        hand_landmarker.close()
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    rows: list[np.ndarray] = []
    index = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp = int(index * 1000 / fps)
            pose_result = pose_landmarker.detect_for_video(image, timestamp)
            hand_result = hand_landmarker.detect_for_video(image, timestamp)
            pose = (
                pose_result.pose_landmarks[0]
                if pose_result.pose_landmarks
                else None
            )
            hands = hand_result.hand_landmarks if hand_result.hand_landmarks else None
            rows.append(_frame_vector(pose, hands))
            index += 1
    finally:
        cap.release()
        pose_landmarker.close()
        hand_landmarker.close()

    if not rows:
        raise RuntimeError(f"El video no produjo frames: {video_path}")
    return np.asarray(rows, dtype=np.float32)


def process_one(item: VideoItem, force: bool = False) -> dict[str, Any]:
    """Extrae (o reutiliza) las features de un video."""

    cache_path = ROOT / item.cache
    started = time.perf_counter()
    try:
        if not force:
            cached = verify_feature_file(cache_path)
            if cached is not None:
                return {
                    "name": item.name,
                    "status": "cached",
                    "frames": int(cached.shape[0]),
                    "seconds": time.perf_counter() - started,
                }

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        sign = extract_features(ROOT / item.path)
        np.savez(cache_path, sign=sign)
        checked = verify_feature_file(cache_path)
        if checked is None:
            raise RuntimeError(f"La caché quedó inválida: {cache_path}")
        return {
            "name": item.name,
            "status": "extracted",
            "frames": int(checked.shape[0]),
            "seconds": time.perf_counter() - started,
        }
    except Exception as exc:  # pragma: no cover - se reporta en el resumen
        return {
            "name": item.name,
            "status": "error",
            "frames": None,
            "seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _examples_from_items(items: Sequence[VideoItem]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for item in items:
        sign = verify_feature_file(ROOT / item.cache)
        if sign is None:
            raise RuntimeError(f"Features inválidas para {item.name}")
        examples.append(
            {
                "name": item.name,
                "signer": item.signer,
                "gloss": item.gloss,
                "text": item.text,
                "sign": torch.from_numpy(sign),
            }
        )
    return examples


def _save_gzip_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_gzip_pickle(path: Path) -> Any:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def build_dataset(
    items: Sequence[VideoItem],
    workers: int = 1,
    limit: int | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Extrae features para ``items`` y devuelve los ejemplos del dataset."""

    subset = list(items[:limit]) if limit is not None else list(items)
    if not subset:
        raise ValueError("No hay videos para procesar")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    results: dict[str, dict[str, Any]] = {}

    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - tqdm está instalado en este entorno
        tqdm = None

    if workers > 1:
        import multiprocessing as mp

        context = mp.get_context("spawn")
        with context.Pool(processes=workers) as pool:
            iterator = pool.imap_unordered(
                _process_one_task,
                [(item, force) for item in subset],
                chunksize=1,
            )
            if tqdm is not None:
                iterator = tqdm(iterator, total=len(subset), desc="Extrayendo")
            for result in iterator:
                results[result["name"]] = result
    else:
        iterator = subset
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(subset), desc="Extrayendo")
        for item in iterator:
            results[item.name] = process_one(item, force=force)

    errors = [r for r in results.values() if r["status"] == "error"]
    if errors:
        lines = "\n".join(f"  {r['name']}: {r['error']}" for r in errors[:20])
        raise RuntimeError(
            f"Fallaron {len(errors)} videos:\n{lines}"
        )

    for item in subset:
        result = results[item.name]
        item.frames = int(result["frames"])
        if item.frames <= 0:
            raise RuntimeError(f"Video sin frames: {item.name}")

    manifest = []
    for item in subset:
        entry = asdict(item)
        entry["seconds"] = round(results[item.name]["seconds"], 3)
        entry["status"] = results[item.name]["status"]
        manifest.append(entry)

    suffix = "" if limit is None else f"_limit{limit}"
    manifest_path = DATASET_DIR / f"manifest{suffix}.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    examples = _examples_from_items(subset)
    dataset_path = DATASET_DIR / ("lsc.all" if limit is None else f"lsc{suffix}.all")
    _save_gzip_pickle(dataset_path, examples)

    total_frames = sum(item.frames or 0 for item in subset)
    elapsed = time.perf_counter() - started
    print(
        f"\nDataset: {len(examples)} ejemplos, {total_frames} frames, "
        f"{elapsed / 60:.1f} min"
    )
    print(f"Manifest: {manifest_path}")
    print(f"Archivo:  {dataset_path}")
    return examples


def _process_one_task(args: tuple[VideoItem, bool]) -> dict[str, Any]:
    item, force = args
    return process_one(item, force=force)


def split_dataset(
    examples: Sequence[dict[str, Any]],
    train_signers: set[str] = TRAIN_SIGNERS,
    val_signers: set[str] = VAL_SIGNERS,
    test_signers: set[str] = TEST_SIGNERS,
) -> dict[str, list[dict[str, Any]]]:
    """Divide por señalante y escribe lsc.train/dev/test."""

    if train_signers & val_signers or train_signers & test_signers or val_signers & test_signers:
        raise ValueError("Los conjuntos de señalantes se solapan")

    all_signers = {str(ex["signer"]) for ex in examples}
    expected = train_signers | val_signers | test_signers
    if all_signers != expected:
        missing = sorted(expected - all_signers, key=int)
        extra = sorted(all_signers - expected, key=int)
        raise ValueError(
            f"El split no cubre los señalantes. Faltan={missing}, sobran={extra}"
        )

    splits = {
        "train": [ex for ex in examples if str(ex["signer"]) in train_signers],
        "dev": [ex for ex in examples if str(ex["signer"]) in val_signers],
        "test": [ex for ex in examples if str(ex["signer"]) in test_signers],
    }

    total = len(examples)
    for name, split in splits.items():
        if not split:
            raise ValueError(f"El split {name} quedó vacío")
        print(
            f"{name:5s}: {len(split):4d} videos "
            f"({100 * len(split) / total:5.2f} %)"
        )

    all_glosses = {ex["gloss"] for ex in examples}
    train_glosses = {ex["gloss"] for ex in splits["train"]}
    missing = sorted(all_glosses - train_glosses)
    if missing:
        raise ValueError(f"Glosas ausentes en train: {missing}")

    for name, split in splits.items():
        path = DATASET_DIR / f"lsc.{name}"
        _save_gzip_pickle(path, split)
        print(f"  guardado {path}")

    split_info = {
        "train_signers": sorted(train_signers, key=int),
        "val_signers": sorted(val_signers, key=int),
        "test_signers": sorted(test_signers, key=int),
        "counts": {name: len(split) for name, split in splits.items()},
        "gloss_counts": {
            name: len({ex["gloss"] for ex in split})
            for name, split in splits.items()
        },
        "total": total,
    }
    (DATASET_DIR / "split.json").write_text(
        json.dumps(split_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Split guardado en {DATASET_DIR / 'split.json'}")
    return splits


def write_config(
    config_path: Path = CONFIG_PATH,
    model_dir: Path = MODEL_DIR,
    epochs: int = 100,
) -> dict[str, Any]:
    """Genera slt/configs/lsc_full.yaml a partir de sign.yaml."""

    base_path = ROOT / "slt" / "configs" / "sign.yaml"
    cfg = copy.deepcopy(yaml.safe_load(base_path.read_text(encoding="utf-8")))

    cfg["name"] = "lsc_full"
    cfg["data"].update(
        {
            "data_path": "./dataset_completo/",
            "version": "phoenix_2014_trans",
            "sgn": "sign",
            "txt": "text",
            "gls": "gloss",
            "train": "lsc.train",
            "dev": "lsc.dev",
            "test": "lsc.test",
            "feature_size": FEATURE_DIM,
            "level": "word",
            "txt_lowercase": True,
            "max_sent_length": 400,
            "random_train_subset": -1,
            "random_dev_subset": -1,
        }
    )
    cfg["testing"] = {
        "recognition_beam_sizes": [1],
        "translation_beam_sizes": [1],
        "translation_beam_alphas": [-1],
    }
    cfg["training"].update(
        {
            "reset_best_ckpt": False,
            "reset_scheduler": False,
            "reset_optimizer": False,
            "random_seed": 42,
            "model_dir": _relative(model_dir),
            "recognition_loss_weight": 1.0,
            "translation_loss_weight": 1.0,
            # Las referencias son de una sola palabra: BLEU-4 siempre vale 0.
            # Por eso evaluamos y seleccionamos por WER de reconocimiento.
            "eval_metric": "wer",
            "optimizer": "adam",
            "learning_rate": 0.0003,
            "batch_size": 32,
            "eval_batch_size": 32,
            "num_valid_log": 5,
            "epochs": epochs,
            "early_stopping_metric": "eval_metric",
            "batch_type": "sentence",
            "translation_normalization": "batch",
            "eval_recognition_beam_size": 1,
            "eval_translation_beam_size": 1,
            "eval_translation_beam_alpha": -1,
            "overwrite": False,
            "shuffle": True,
            "use_cuda": False,
            "translation_max_output_length": 30,
            "keep_last_ckpts": 1,
            "batch_multiplier": 1,
            "logging_freq": 20,
            "validation_freq": 50,
            "betas": [0.9, 0.998],
            "scheduling": "plateau",
            "learning_rate_min": 1.0e-6,
            "weight_decay": 0.001,
            # Paciencia alta: con 1.733 ejemplos el CTC tarda en salir del
            # colapso a blank; no queremos que la segunda reducción de LR
            # corte el entrenamiento antes de tiempo.
            "patience": 1000,
            "decrease_factor": 0.5,
            "label_smoothing": 0.0,
        }
    )

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Configuración guardada en {config_path}")
    return cfg


def train_model(
    config_path: Path = CONFIG_PATH,
    model_dir: Path = MODEL_DIR,
    force: bool = False,
) -> Path:
    """Entrena con signjoey y devuelve la ruta del mejor checkpoint."""

    best = model_dir / "best.ckpt"
    if best.is_file() and not force:
        print(f"Ya existe un modelo entrenado: {best}")
        return best
    if model_dir.exists() and not force:
        raise FileExistsError(
            f"{model_dir} existe pero no contiene best.ckpt. "
            "Usa --force para recrearlo."
        )
    if force and model_dir.exists():
        print(f"Borrando {model_dir} por --force")
        shutil.rmtree(model_dir)

    env = os.environ.copy()
    pythonpath = str(ROOT / "slt")
    env["PYTHONPATH"] = pythonpath + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [sys.executable, "-m", "signjoey", "train", str(config_path)]
    print("Ejecutando:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)

    if not best.is_file():
        raise FileNotFoundError(f"No se generó el checkpoint esperado: {best}")
    print(f"Modelo guardado en {best}")
    return best


def load_trained_model() -> tuple[Any, Any, Any]:
    """Carga best.ckpt y devuelve (modelo, vocabulario de glosas, de texto)."""

    slt_dir = str(ROOT / "slt")
    if slt_dir not in sys.path:
        sys.path.insert(0, slt_dir)

    from signjoey.helpers import load_checkpoint, load_config
    from signjoey.model import build_model
    from signjoey.vocabulary import GlossVocabulary, TextVocabulary

    # Preferimos la copia guardada junto al checkpoint: es la configuración
    # exacta con la que se construyó el modelo.
    saved_config = MODEL_DIR / "config.yaml"
    cfg = load_config(str(saved_config if saved_config.is_file() else CONFIG_PATH))
    gls_vocab = GlossVocabulary(file=str(MODEL_DIR / "gls.vocab"))
    txt_vocab = TextVocabulary(file=str(MODEL_DIR / "txt.vocab"))
    model = build_model(
        cfg["model"],
        sgn_dim=FEATURE_DIM,
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        do_recognition=True,
        do_translation=True,
    )
    checkpoint = load_checkpoint(str(MODEL_DIR / "best.ckpt"), use_cuda=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, gls_vocab, txt_vocab


def predict(
    model: Any,
    gls_vocab: Any,
    txt_vocab: Any,
    sign: np.ndarray | torch.Tensor,
) -> dict[str, list[str]]:
    """Predice glosa y texto para una matriz (frames, 225)."""

    from types import SimpleNamespace

    from signjoey.batch import Batch

    if isinstance(sign, torch.Tensor):
        tensor = sign.detach().cpu().float()
    else:
        tensor = torch.from_numpy(np.asarray(sign, dtype=np.float32))
    batch = Batch(
        torch_batch=SimpleNamespace(
            sequence=["demo"],
            signer=["demo"],
            sgn=(tensor.unsqueeze(0), torch.tensor([tensor.shape[0]])),
        ),
        txt_pad_index=txt_vocab.stoi["<pad>"],
        sgn_dim=FEATURE_DIM,
        is_train=False,
    )
    with torch.no_grad():
        gls_hyp, txt_hyp, _ = model.run_batch(
            batch,
            recognition_beam_size=1,
            translation_beam_size=1,
            translation_max_output_length=10,
        )
    glosa = [gls_vocab.itos[i] for i in gls_hyp[0]]
    texto = txt_vocab.arrays_to_sentences(np.asarray(txt_hyp))[0]
    return {"glosa": glosa, "texto": texto}


def training_summary(model_dir: Path = MODEL_DIR, tail: int = 40) -> str:
    """Devuelve el final del log y las validaciones, si existen."""

    parts = []
    for name in ("validations.txt", "train.log"):
        path = model_dir / name
        if path.is_file():
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            parts.append(f"--- {name} (últimas {tail} líneas) ---")
            parts.append("\n".join(lines[-tail:]))
    return "\n".join(parts)


def _load_all_examples() -> list[dict[str, Any]]:
    path = DATASET_DIR / "lsc.all"
    if not path.is_file():
        raise FileNotFoundError(f"No existe {path}; ejecuta primero el dataset.")
    return load_gzip_pickle(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_parser = subparsers.add_parser(
        "dataset", help="Extrae features, crea el dataset y el split"
    )
    dataset_parser.add_argument("--workers", type=int, default=1)
    dataset_parser.add_argument("--limit", type=int, default=None)
    dataset_parser.add_argument("--force", action="store_true")

    split_parser = subparsers.add_parser(
        "split", help="Divide lsc.all en train/dev/test"
    )
    split_parser.add_argument("--force", action="store_true")

    config_parser = subparsers.add_parser(
        "config", help="Genera slt/configs/lsc_full.yaml"
    )
    config_parser.add_argument("--epochs", type=int, default=100)

    train_parser = subparsers.add_parser("train", help="Entrena con signjoey")
    train_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("all", help="Dataset completo + split + entrenamiento")

    args = parser.parse_args(argv)

    if args.command == "dataset":
        items = scan_videos()
        print(
            f"Videos: {len(items)} | señalantes: "
            f"{len({i.signer for i in items})} | glosas: "
            f"{len({i.gloss for i in items})}"
        )
        build_dataset(items, workers=args.workers, limit=args.limit, force=args.force)
        if args.limit is None:
            examples = _load_all_examples()
            split_dataset(examples)
            write_config()
            print("\nListo. Siguiente paso: python lsc_pipeline.py train")
        else:
            print("\nModo limitado: no se generó el split 60/20/20.")
        return 0

    if args.command == "split":
        examples = _load_all_examples()
        split_dataset(examples)
        write_config()
        return 0

    if args.command == "config":
        write_config(epochs=args.epochs)
        return 0

    if args.command == "train":
        train_model(force=args.force)
        print("\n" + training_summary())
        return 0

    if args.command == "all":
        items = scan_videos()
        build_dataset(items, workers=4)
        examples = _load_all_examples()
        split_dataset(examples)
        write_config()
        train_model()
        print("\n" + training_summary())
        return 0

    raise ValueError(f"Comando desconocido: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
