"""SL-TSSI-DenseNet con aumento de datos.

Módulo autónomo para la parte de aumento de datos del cuaderno. Contiene:

- Carga de los splits y de las categorías desde disco (no depende de celdas previas).
- Una copia fiel del pipeline TSSI del modelo v2 (normalización, orden de
  articulaciones, escalado y entrada a la red). `verificar_fidelidad_v2` comprueba
  que la copia reproduce exactamente las predicciones guardadas del v2.
- Las transformaciones de aumento, que operan sobre la matriz cruda (T, 225).
- El entrenamiento, que guarda el modelo, se salta si ya existe y se reanuda si se
  interrumpió.
- Las métricas y los archivos de resultados, con el mismo formato que `sl_tssi_v2/`.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
DATOS = ROOT / "dataset_completo"
MODELOS = ROOT / "modelos"
DIR_V2 = MODELOS / "sl_tssi_v2"
DIR_AUG = MODELOS / "sl_tssi_aug"

CATEGORIAS = ("Colores", "Cortesía", "Números")

# Parámetros que no cambian respecto al v2.
TAMANO = 128
MEDIA_IMAGENET = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
DESV_IMAGENET = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Configuración del aumento. Cada transformación se aplica con su probabilidad.
AUMENTO = {
    "p_recortar": 0.5, "recorte_max": 0.10,        # fracción total quitada al inicio y al final
    "p_deformar": 0.5, "deformar": (0.8, 1.2),     # cambio de velocidad de una parte frente a la otra
    "p_rotar": 0.5, "grados": 12.0,                # giro en el plano x, y
    "p_estirar": 0.5, "estirar": (0.9, 1.1),       # factor distinto en x y en y
    "p_ruido": 0.5, "sigma": 0.002,                # ruido gaussiano en x, y (coordenadas de imagen)
    "p_quitar_mano": 0.3, "quitar_frac": 0.10,     # bloque de fotogramas sin una mano
}
SIN_AUMENTO = {k: (0.0 if k.startswith("p_") else v) for k, v in AUMENTO.items()}


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def cargar_split(nombre: str) -> list[dict]:
    with gzip.open(DATOS / f"lsc.{nombre}", "rb") as f:
        return pickle.load(f)


def cargar_datos() -> SimpleNamespace:
    """Carga train/dev/test, las clases y la categoría de cada seña."""
    train, dev, test = cargar_split("train"), cargar_split("dev"), cargar_split("test")
    clases = sorted({e["gloss"] for e in train})

    manifest = json.loads((DATOS / "manifest.json").read_text(encoding="utf-8"))
    categorias_por_glosa = defaultdict(set)
    for e in manifest:
        categorias_por_glosa[e["gloss"]].add(e["category"])
    # Los números aparecen en dos carpetas (Courtesy y Numbers).
    grupo = {g: ("Números" if len(c) > 1 else "Colores" if "Colors" in c else "Cortesía")
             for g, c in categorias_por_glosa.items()}

    return SimpleNamespace(train=train, dev=dev, test=test, clases=clases,
                           a_indice={g: i for i, g in enumerate(clases)}, grupo=grupo)


# ---------------------------------------------------------------------------
# Pipeline TSSI (copia del v2)
# ---------------------------------------------------------------------------

def normalizar(secuencia: torch.Tensor) -> torch.Tensor:
    """Centra el cuerpo entre los hombros y cada mano en su propia muñeca."""
    s = secuencia.clone()
    pose = s[:, 0:99].reshape(-1, 33, 3)
    izquierda = s[:, 99:162].reshape(-1, 21, 3)
    derecha = s[:, 162:225].reshape(-1, 21, 3)

    hombro_izq, hombro_der = pose[:, 11, :], pose[:, 12, :]
    centro = ((hombro_izq + hombro_der) / 2).unsqueeze(1)
    ancho = torch.norm(hombro_izq[:, :2] - hombro_der[:, :2], dim=1)
    ancho = torch.clamp(ancho, min=1e-3).view(-1, 1, 1)
    hay_pose = (pose.abs().sum(dim=(1, 2)) > 0).view(-1, 1, 1)
    pose = torch.where(hay_pose, (pose - centro) / ancho, pose)

    def normalizar_mano(mano):
        hay_mano = (mano.abs().sum(dim=(1, 2)) > 0).view(-1, 1, 1)
        centrada = mano - mano[:, 0:1, :]
        escala = centrada.norm(dim=2).max(dim=1).values
        escala = torch.clamp(escala, min=1e-3).view(-1, 1, 1)
        return torch.where(hay_mano, centrada / escala, mano)

    izquierda = normalizar_mano(izquierda)
    derecha = normalizar_mano(derecha)
    return torch.cat(
        [pose.reshape(-1, 99), izquierda.reshape(-1, 63), derecha.reshape(-1, 63)], dim=1)


NARIZ = 0
HOMBRO_IZQ, HOMBRO_DER = 11, 12
CODO_IZQ, CODO_DER = 13, 14
MUNECA_IZQ, MUNECA_DER = 15, 16
CADERA_IZQ, CADERA_DER = 23, 24

MANO = [0]
for _dedo in ([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
              [13, 14, 15, 16], [17, 18, 19, 20]):
    MANO += _dedo + _dedo[-2::-1] + [0]

BASE_IZQ, BASE_DER = 33, 54

ORDEN_TSSI = (
    [NARIZ]
    + [HOMBRO_IZQ, CODO_IZQ, MUNECA_IZQ]
    + [BASE_IZQ + i for i in MANO]
    + [MUNECA_IZQ, CODO_IZQ, HOMBRO_IZQ]
    + [NARIZ]
    + [HOMBRO_DER, CODO_DER, MUNECA_DER]
    + [BASE_DER + i for i in MANO]
    + [MUNECA_DER, CODO_DER, HOMBRO_DER]
    + [NARIZ, CADERA_IZQ, NARIZ, CADERA_DER]
)


def a_imagen_tssi(secuencia: torch.Tensor) -> torch.Tensor:
    """(T, 225) -> (3, T, 100), sobre coordenadas normalizadas."""
    s = normalizar(secuencia.float())
    puntos = s.reshape(-1, 75, 3)
    return puntos[:, ORDEN_TSSI, :].permute(2, 0, 1)


def calcular_percentiles(train: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """Percentiles 1 y 99 por canal, con los primeros 300 videos de train (igual que el v2)."""
    muestra = torch.cat([a_imagen_tssi(e["sign"]).reshape(3, -1) for e in train[:300]], dim=1)
    bajo = torch.quantile(muestra, 0.01, dim=1).view(3, 1, 1)
    alto = torch.quantile(muestra, 0.99, dim=1).view(3, 1, 1)
    return bajo, alto


def escalar_a_imagen(imagen, bajo, alto):
    return ((imagen - bajo) / (alto - bajo)).clamp(0, 1)


def a_entrada_red(imagen: torch.Tensor) -> torch.Tensor:
    """Redimensiona a 128x128 y aplica la normalización de ImageNet."""
    img = F.interpolate(imagen.unsqueeze(0), size=(TAMANO, TAMANO),
                        mode="bilinear", align_corners=False).squeeze(0)
    return (img - MEDIA_IMAGENET) / DESV_IMAGENET


class DatosTSSI(Dataset):
    """Sin aumento: las imágenes TSSI se construyen una sola vez."""

    def __init__(self, datos, a_indice, bajo, alto):
        self.imagenes = [escalar_a_imagen(a_imagen_tssi(e["sign"]), bajo, alto) for e in datos]
        self.etiquetas = torch.tensor([a_indice[e["gloss"]] for e in datos])

    def __len__(self):
        return len(self.etiquetas)

    def __getitem__(self, i):
        return a_entrada_red(self.imagenes[i]), self.etiquetas[i]


class DatosTSSIAumentados(Dataset):
    """Con aumento: guarda la matriz cruda y arma una TSSI distinta cada vez que se pide."""

    def __init__(self, datos, a_indice, bajo, alto, aumento, generador):
        self.secuencias = [e["sign"].float() for e in datos]
        self.etiquetas = torch.tensor([a_indice[e["gloss"]] for e in datos])
        self.bajo, self.alto = bajo, alto
        self.aumento, self.generador = aumento, generador

    def __len__(self):
        return len(self.etiquetas)

    def __getitem__(self, i):
        s = aumentar(self.secuencias[i], self.generador, self.aumento)
        imagen = escalar_a_imagen(a_imagen_tssi(s), self.bajo, self.alto)
        return a_entrada_red(imagen), self.etiquetas[i]


def construir_densenet(n_clases: int, imagenet: bool = True) -> nn.Module:
    from torchvision.models import DenseNet121_Weights, densenet121
    modelo = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if imagenet else None)
    modelo.classifier = nn.Linear(1024, n_clases)
    return modelo


@torch.no_grad()
def evaluar(modelo, dataset, lote=32):
    """Devuelve (pérdida media, exactitud, predicciones)."""
    modelo.eval()
    criterio = nn.CrossEntropyLoss(reduction="sum")
    perdida, predichas = 0.0, []
    for xb, yb in DataLoader(dataset, batch_size=lote):
        salida = modelo(xb)
        perdida += criterio(salida, yb).item()
        predichas.append(salida.argmax(1))
    predichas = torch.cat(predichas)
    acc = (predichas == dataset.etiquetas).float().mean().item()
    return perdida / len(dataset), acc, predichas


# ---------------------------------------------------------------------------
# Aumento de datos (sobre la matriz cruda (T, 225))
# ---------------------------------------------------------------------------

def _puntos(s):
    return s.reshape(s.shape[0], 75, 3)


def _presentes(p):
    """(T, 75): True donde MediaPipe detectó el punto (los ausentes son ceros)."""
    return p.abs().sum(-1) > 0


def _centro(p, presentes):
    """Centro de los hombros, promediado sobre los fotogramas con pose."""
    validos = presentes[:, HOMBRO_IZQ] & presentes[:, HOMBRO_DER]
    if not validos.any():
        return torch.tensor([0.5, 0.5], dtype=p.dtype)
    return ((p[validos, HOMBRO_IZQ, :2] + p[validos, HOMBRO_DER, :2]) / 2).mean(0)


def _transformar_xy(s, funcion):
    """Aplica `funcion` a las x, y de los puntos presentes; los ceros quedan en cero."""
    p = _puntos(s).clone()
    presentes = _presentes(p)
    c = _centro(p, presentes)
    nuevo = funcion(p[..., :2] - c) + c
    p[..., :2] = torch.where(presentes[..., None], nuevo, p[..., :2])
    return p.reshape(s.shape)


def rotar(s, grados):
    a = math.radians(grados)
    R = torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], dtype=s.dtype)
    return _transformar_xy(s, lambda xy: xy @ R.T)


def estirar(s, fx, fy):
    f = torch.tensor([fx, fy], dtype=s.dtype)
    return _transformar_xy(s, lambda xy: xy * f)


def escalar(s, factor):
    """Escala uniforme en x, y, z (cámara más cerca o más lejos).

    No se usa para aumentar: la normalización la anula.
    """
    s = _transformar_xy(s, lambda xy: xy * factor)
    p = _puntos(s)
    p[..., 2] = p[..., 2] * factor           # los ceros siguen en cero
    return p.reshape(s.shape)


def trasladar(s, dx, dy):
    """Traslación. No se usa para aumentar: la normalización la anula."""
    d = torch.tensor([dx, dy], dtype=s.dtype)
    return _transformar_xy(s, lambda xy: xy + d)


def ruido(s, sigma, generador=None):
    p = _puntos(s).clone()
    presentes = _presentes(p)
    r = torch.randn(p[..., :2].shape, generator=generador, dtype=s.dtype) * sigma
    p[..., :2] = torch.where(presentes[..., None], p[..., :2] + r, p[..., :2])
    return p.reshape(s.shape)


def recortar_tiempo(s, inicio, fin):
    """Quita `inicio` fotogramas al principio y `fin` al final."""
    return s[inicio: s.shape[0] - fin].clone()


def _interpolar(s, posiciones):
    """Muestrea la secuencia en posiciones fraccionarias.

    Interpola linealmente solo cuando el punto está presente en los dos fotogramas
    vecinos; si no, toma el vecino más cercano. Así una mano ausente no se convierte
    en una mano "a medias".
    """
    T = s.shape[0]
    p = _puntos(s)
    presentes = _presentes(p)
    i0 = posiciones.floor().long().clamp(0, T - 1)
    i1 = (i0 + 1).clamp(max=T - 1)
    w = (posiciones - i0.to(posiciones.dtype)).clamp(0, 1).view(-1, 1, 1)
    p0, p1 = p[i0], p[i1]
    lineal = (1 - w) * p0 + w * p1
    cercano = torch.where(w < 0.5, p0, p1)
    ambos = (presentes[i0] & presentes[i1])[..., None]
    return torch.where(ambos, lineal, cercano).reshape(len(posiciones), -1)


def deformar_tiempo(s, factor, corte=0.5):
    """La parte antes de `corte` dura `factor` veces más (o menos) que la de después.

    Se conserva el número de fotogramas: el redimensionado a 128x128 anularía un
    cambio de velocidad uniforme, así que solo tiene efecto un cambio desigual.
    """
    T = s.shape[0]
    if T < 4:
        return s.clone()
    n = T - 1
    a = corte * n                                   # punto de quiebre en la entrada
    a_salida = n * a * factor / (a * factor + (n - a))  # dónde cae ese punto en la salida
    u = torch.arange(T, dtype=torch.float32)
    t = torch.where(u <= a_salida,
                    u * a / a_salida,
                    a + (u - a_salida) * (n - a) / (n - a_salida))
    return _interpolar(s, t)


def quitar_mano(s, mano, inicio, largo):
    """Pone en cero una mano ('izquierda' o 'derecha') en un bloque de fotogramas."""
    s = s.clone()
    columnas = slice(99, 162) if mano == "izquierda" else slice(162, 225)
    s[inicio: inicio + largo, columnas] = 0
    return s


def _uniforme(g, a, b):
    return a + (b - a) * torch.rand(1, generator=g).item()


def _moneda(g, p):
    return p > 0 and torch.rand(1, generator=g).item() < p


def aumentar(secuencia, generador=None, cfg=AUMENTO):
    """Aplica al azar las transformaciones de `cfg` a una matriz cruda (T, 225)."""
    s = secuencia.float().clone()
    g = generador

    if _moneda(g, cfg["p_recortar"]):
        T = s.shape[0]
        total = int(_uniforme(g, 0, cfg["recorte_max"]) * T)
        total = min(total, max(0, T - 8))           # dejar al menos 8 fotogramas
        inicio = int(torch.randint(0, total + 1, (1,), generator=g).item())
        s = recortar_tiempo(s, inicio, total - inicio)

    if _moneda(g, cfg["p_deformar"]):
        s = deformar_tiempo(s, _uniforme(g, *cfg["deformar"]), _uniforme(g, 0.3, 0.7))

    if _moneda(g, cfg["p_rotar"]):
        s = rotar(s, _uniforme(g, -cfg["grados"], cfg["grados"]))

    if _moneda(g, cfg["p_estirar"]):
        s = estirar(s, _uniforme(g, *cfg["estirar"]), _uniforme(g, *cfg["estirar"]))

    if _moneda(g, cfg["p_ruido"]):
        s = ruido(s, cfg["sigma"], g)

    if _moneda(g, cfg["p_quitar_mano"]):
        T = s.shape[0]
        largo = max(1, round(cfg["quitar_frac"] * T))
        inicio = int(torch.randint(0, T - largo + 1, (1,), generator=g).item())
        mano = "izquierda" if _moneda(g, 0.5) else "derecha"
        s = quitar_mano(s, mano, inicio, largo)

    return s


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def _guardar_atomico(objeto, ruta: Path):
    """Guarda en un temporal y lo renombra: si se corta a mitad, no queda un archivo roto."""
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    torch.save(objeto, temporal)
    os.replace(temporal, ruta)


def _mostrar(ruta: Path) -> str:
    """Ruta relativa al proyecto si está dentro de él; si no, la ruta completa."""
    try:
        return Path(ruta).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(ruta)


def _cargar(ruta: Path):
    return torch.load(ruta, map_location="cpu", weights_only=False)


def entrenar_o_cargar(train, dev, clases, dir_modelo: Path = DIR_AUG, epocas=40, lr=1e-4,
                      paciencia=6, semilla=42, aumento=AUMENTO, lote=16, n_train_limpio=300):
    """Entrena la DenseNet con aumento, o la carga si ya está entrenada.

    - Si existe `best.pt`, carga el modelo y no entrena.
    - Si existe `ultimo.pt` (entrenamiento interrumpido), continúa desde esa época.
    - Si no hay nada, empieza de cero.

    Devuelve (modelo, historial).
    """
    dir_modelo = Path(dir_modelo)
    ruta_best, ruta_ultimo = dir_modelo / "best.pt", dir_modelo / "ultimo.pt"
    ruta_hist = dir_modelo / "historial.json"

    if ruta_best.is_file():
        guardado = _cargar(ruta_best)
        assert guardado["clases"] == clases, "Las clases del modelo guardado no coinciden"
        modelo = construir_densenet(len(clases), imagenet=False)
        modelo.load_state_dict(guardado["model_state"])
        modelo.eval()
        print(f"Modelo ya entrenado, no se reentrena: {_mostrar(ruta_best)}")
        print(f"Mejor validación: {guardado['acc_dev']:.1%} (época {guardado['epoca_mejor']} "
              f"de {len(guardado['historial'])} entrenadas)")
        return modelo, guardado["historial"]

    a_indice = {g: i for i, g in enumerate(clases)}
    config = {"epocas": epocas, "lr": lr, "paciencia": paciencia, "semilla": semilla,
              "aumento": dict(aumento), "lote": lote, "n_train_limpio": n_train_limpio}

    # --- Datos ---
    bajo, alto = calcular_percentiles(train)
    g_aumento = torch.Generator().manual_seed(semilla)
    g_orden = torch.Generator().manual_seed(semilla + 1)
    datos_tr = DatosTSSIAumentados(train, a_indice, bajo, alto, aumento, g_aumento)
    datos_dv = DatosTSSI(dev, a_indice, bajo, alto)
    # Subconjunto fijo de train SIN aumento: mide cuánto memoriza el modelo.
    indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(semilla))
    datos_tr_limpio = DatosTSSI([train[i] for i in sorted(indices[:n_train_limpio].tolist())],
                                a_indice, bajo, alto)
    cargador_tr = DataLoader(datos_tr, batch_size=lote, shuffle=True, generator=g_orden)

    # --- Modelo ---
    torch.manual_seed(semilla)
    modelo = construir_densenet(len(clases))
    optim = torch.optim.Adam(modelo.parameters(), lr=lr)
    planificador = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", factor=0.5,
                                                              patience=2)
    criterio = nn.CrossEntropyLoss()

    historial, mejor_dev, mejores_pesos, epoca_mejor, primera = [], -1.0, None, 0, 1

    if ruta_ultimo.is_file():
        estado = _cargar(ruta_ultimo)
        assert estado["clases"] == clases, "Las clases del entrenamiento a reanudar no coinciden"
        if estado["config"] != config:
            raise ValueError("ultimo.pt se entrenó con otra configuración. Bórralo para empezar "
                             f"de cero.\n  guardada: {estado['config']}\n  actual:   {config}")
        modelo.load_state_dict(estado["model_state"])
        optim.load_state_dict(estado["optim_state"])
        planificador.load_state_dict(estado["sched_state"])
        historial, mejor_dev = estado["historial"], estado["mejor_dev"]
        mejores_pesos, epoca_mejor = estado["mejores_pesos"], estado["epoca_mejor"]
        torch.set_rng_state(estado["rng"]["global"])
        g_aumento.set_state(estado["rng"]["aumento"])
        g_orden.set_state(estado["rng"]["orden"])
        primera = estado["epoca"] + 1
        print(f"Reanudando desde la época {primera} (mejor dev hasta ahora: {mejor_dev:.1%}, "
              f"época {epoca_mejor})")
    else:
        print(f"Entrenando desde cero: hasta {epocas} épocas, parada temprana tras {paciencia} "
              f"sin mejorar.")
    print(f"  train {len(datos_tr)} (aumentado) | dev {len(datos_dv)} | "
          f"train sin aumento para control: {len(datos_tr_limpio)}\n", flush=True)

    dir_modelo.mkdir(parents=True, exist_ok=True)
    parar = bool(historial) and (historial[-1]["epoca"] - epoca_mejor >= paciencia)

    for epoca in range(primera, epocas + 1):
        if parar:
            break
        inicio = time.perf_counter()
        modelo.train()
        perdida_tr = aciertos_tr = vistos = 0
        for xb, yb in cargador_tr:
            optim.zero_grad()
            salida = modelo(xb)
            perdida = criterio(salida, yb)
            perdida.backward()
            optim.step()
            perdida_tr += perdida.item() * yb.numel()
            aciertos_tr += (salida.argmax(1) == yb).sum().item()
            vistos += yb.numel()

        loss_dev, acc_dev, _ = evaluar(modelo, datos_dv)
        loss_limpio, acc_limpio, _ = evaluar(modelo, datos_tr_limpio)
        planificador.step(acc_dev)

        marca = ""
        if acc_dev > mejor_dev:
            mejor_dev, epoca_mejor = acc_dev, epoca
            mejores_pesos = {k: v.clone() for k, v in modelo.state_dict().items()}
            marca = "  *"
        historial.append({"epoca": epoca,
                          "loss_train": perdida_tr / vistos,      # sobre datos aumentados
                          "acc_train": aciertos_tr / vistos,      # sobre datos aumentados
                          "loss_train_limpio": loss_limpio,       # train sin aumento
                          "acc_train_limpio": acc_limpio,         # train sin aumento
                          "loss_dev": loss_dev,
                          "acc_dev": acc_dev,
                          "lr": optim.param_groups[0]["lr"],
                          "minutos": (time.perf_counter() - inicio) / 60})
        print(f"época {epoca:2d}/{epocas}  pérdida {perdida_tr / vistos:.3f}  "
              f"train aum. {aciertos_tr / vistos:.1%}  train limpio {acc_limpio:.1%}  "
              f"dev {acc_dev:.1%}  lr {optim.param_groups[0]['lr']:.0e}  "
              f"({historial[-1]['minutos']:.1f} min){marca}", flush=True)

        _guardar_atomico({"epoca": epoca, "clases": clases, "config": config,
                          "model_state": modelo.state_dict(), "optim_state": optim.state_dict(),
                          "sched_state": planificador.state_dict(),
                          "mejores_pesos": mejores_pesos, "mejor_dev": mejor_dev,
                          "epoca_mejor": epoca_mejor, "historial": historial,
                          "rng": {"global": torch.get_rng_state(),
                                  "aumento": g_aumento.get_state(),
                                  "orden": g_orden.get_state()}},
                         ruta_ultimo)
        ruta_hist.write_text(json.dumps(historial, indent=2), encoding="utf-8")

        if epoca - epoca_mejor >= paciencia:
            print(f"\nParada temprana: {paciencia} épocas sin mejorar.")
            parar = True

    modelo.load_state_dict(mejores_pesos)
    modelo.eval()
    _guardar_atomico({"model_state": mejores_pesos, "clases": clases, "tamano": TAMANO,
                      "orden_tssi": ORDEN_TSSI, "semilla": semilla, "p_bajo": bajo,
                      "p_alto": alto, "acc_dev": mejor_dev, "epoca_mejor": epoca_mejor,
                      "historial": historial, "config": config},
                     ruta_best)
    ruta_hist.write_text(json.dumps(historial, indent=2), encoding="utf-8")
    ruta_ultimo.unlink(missing_ok=True)
    minutos = sum(h["minutos"] for h in historial)
    print(f"\nMejor dev: {mejor_dev:.1%} (época {epoca_mejor}) · {len(historial)} épocas · "
          f"{minutos / 60:.1f} h · guardado en {_mostrar(ruta_best)}")
    return modelo, historial


# ---------------------------------------------------------------------------
# Resultados y métricas
# ---------------------------------------------------------------------------

def metricas(reales, predichas, clases, grupo):
    """Mismo formato que sl_tssi_v2/resumen.json."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for r, p in zip(reales, predichas):
        if r == p:
            tp[r] += 1
        else:
            fp[p] += 1
            fn[r] += 1
    por_clase = {}
    for c in clases:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        por_clase[c] = {"precision": prec, "recall": rec, "f1": f1, "soporte": tp[c] + fn[c]}
    presentes = [c for c in clases if por_clase[c]["soporte"] > 0]

    cat = defaultdict(lambda: [0, 0])
    for r, p in zip(reales, predichas):
        cat[grupo[r]][0] += int(r == p)
        cat[grupo[r]][1] += 1

    n = len(reales)
    exactitud = sum(r == p for r, p in zip(reales, predichas)) / n
    return {
        "exactitud": exactitud,
        "margen_95": 1.96 * math.sqrt(exactitud * (1 - exactitud) / n),
        "f1_macro": sum(por_clase[c]["f1"] for c in presentes) / len(presentes),
        "recall_macro": sum(por_clase[c]["recall"] for c in presentes) / len(presentes),
        "precision_macro": sum(por_clase[c]["precision"] for c in presentes) / len(presentes),
        "por_categoria": {k: cat[k][0] / cat[k][1] for k in CATEGORIAS if cat[k][1]},
        "por_clase": por_clase,
    }


def predicciones_test(modelo, datos, dir_modelo: Path = DIR_AUG):
    """Predice test (o carga las predicciones guardadas) y escribe resumen.json.

    Devuelve (lista de glosas predichas, resumen).
    """
    dir_modelo = Path(dir_modelo)
    ruta_pred, ruta_res = dir_modelo / "predicciones_test.json", dir_modelo / "resumen.json"
    reales = [e["gloss"] for e in datos.test]
    nombres = [e["name"] for e in datos.test]

    if ruta_pred.is_file():
        guardado = json.loads(ruta_pred.read_text(encoding="utf-8"))
        assert guardado["nombres"] == nombres, "Las predicciones guardadas son de otro test"
        predichas = guardado["predichas"]
        print(f"Predicciones de prueba cargadas de {_mostrar(ruta_pred)}")
    else:
        guardado = _cargar(dir_modelo / "best.pt")
        test = DatosTSSI(datos.test, datos.a_indice, guardado["p_bajo"], guardado["p_alto"])
        _, _, pred = evaluar(modelo, test)
        predichas = [datos.clases[i] for i in pred.tolist()]
        ruta_pred.write_text(json.dumps({"clases": datos.clases, "nombres": nombres,
                                         "reales": reales, "predichas": predichas},
                                        ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Predicciones de prueba guardadas en {_mostrar(ruta_pred)}")

    resumen = metricas(reales, predichas, datos.clases, datos.grupo)
    ruta_res.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    return predichas, resumen


def leer_predicciones(dir_modelo: Path):
    d = json.loads((Path(dir_modelo) / "predicciones_test.json").read_text(encoding="utf-8"))
    return d["reales"], d["predichas"]


def leer_historial(dir_modelo: Path):
    return json.loads((Path(dir_modelo) / "historial.json").read_text(encoding="utf-8"))


def leer_transformer(carpeta: str):
    """Glosas reales y predichas que signjoey guardó para test."""
    archivos = sorted((MODELOS / carpeta).glob("best.IT_*.test_results.pkl"))
    if not archivos:
        return None
    with open(archivos[-1], "rb") as f:
        m = pickle.load(f)

    def texto(x):
        return " ".join(x) if isinstance(x, (list, tuple)) else str(x)

    return [texto(r) for r in m["gls_ref"]], [texto(h) for h in m["gls_hyp"]]


def mcnemar(reales, pred_a, pred_b):
    """Prueba exacta de McNemar entre dos modelos evaluados sobre los mismos videos.

    Devuelve (solo A acierta, solo B acierta, valor p bilateral).
    """
    solo_a = sum(a == r != b for r, a, b in zip(reales, pred_a, pred_b))
    solo_b = sum(b == r != a for r, a, b in zip(reales, pred_a, pred_b))
    n, k = solo_a + solo_b, min(solo_a, solo_b)
    if n == 0:
        return solo_a, solo_b, 1.0
    cola = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return solo_a, solo_b, min(1.0, 2 * cola)


def verificar_fidelidad_v2(datos):
    """Comprueba que este módulo reproduce el pipeline del v2.

    Carga sl_tssi_v2/best.pt, predice test con el código de este módulo y compara con
    las predicciones que el v2 guardó. También compara los percentiles de escalado.
    Devuelve la fracción de predicciones que coinciden.
    """
    guardado = _cargar(DIR_V2 / "best.pt")
    assert guardado["clases"] == datos.clases, "Las clases del v2 no coinciden"
    assert list(guardado["orden_tssi"]) == ORDEN_TSSI, "El orden TSSI del v2 no coincide"
    assert guardado["tamano"] == TAMANO, "El tamaño de entrada del v2 no coincide"

    bajo, alto = calcular_percentiles(datos.train)
    dif = max((bajo - guardado["p_bajo"]).abs().max().item(),
              (alto - guardado["p_alto"]).abs().max().item())
    print(f"Percentiles de escalado: diferencia máxima con los del v2 = {dif:.2e}")

    modelo = construir_densenet(len(datos.clases), imagenet=False)
    modelo.load_state_dict(guardado["model_state"])
    test = DatosTSSI(datos.test, datos.a_indice, bajo, alto)
    _, acc, pred = evaluar(modelo, test)
    nuevas = [datos.clases[i] for i in pred.tolist()]
    _, guardadas = leer_predicciones(DIR_V2)
    coincidencia = sum(a == b for a, b in zip(nuevas, guardadas)) / len(guardadas)
    print(f"Exactitud del v2 recalculada con este módulo: {acc:.1%}")
    print(f"Predicciones que coinciden con las guardadas del v2: {coincidencia:.1%}")
    return coincidencia
