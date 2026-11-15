"""Exporta el modelo SL-TSSI-DenseNet (con aumento) a ONNX para el demo web.

El archivo ONNX incluye TODO el preprocesamiento del cuaderno, de modo que el
navegador solo entrega la matriz cruda (T, 225) que produce MediaPipe:

    matriz cruda -> normalizar -> imagen TSSI -> escalar con p_bajo/p_alto
                 -> redimensionar a 128x128 -> normalización ImageNet -> DenseNet-121

Además evalúa cada seña en dos orientaciones (original y espejo horizontal). En el
dataset los colores se grabaron con la mano hacia un lado de la imagen y los números
y la cortesía hacia el otro; el demo se queda con la orientación más segura, lo que
lo hace independiente de la mano que use la persona y de si la cámara voltea la imagen.

Uso (desde la raíz del proyecto):

    python Demo/herramientas/exportar_onnx.py

Escribe Demo/web/modelos/lsc_tssi.onnx y Demo/web/modelos/clases.json, y verifica
que ONNX reproduce las predicciones de PyTorch sobre el conjunto de prueba.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

import tssi_aumento as ta  # noqa: E402

DESTINO = RAIZ / "Demo" / "web" / "modelos"
RUTA_MODELO = ta.DIR_AUG / "best.pt"

# Cómo se muestra cada clase en la página (el modelo usa los nombres de las carpetas).
ETIQUETAS = {
    "Baño": "Baño", "Buenas Noches": "Buenas noches", "Buenos dias": "Buenos días",
    "Como esta": "¿Cómo está?", "Denada": "De nada", "Maso Menos": "Más o menos",
    "Muchas Gracias": "Muchas gracias", "Muy Bien": "Muy bien", "Perdon": "Perdón",
    "Si": "Sí", "cafe": "Café", "dies": "Diez",
}

# Pares izquierda/derecha de MediaPipe Pose que se intercambian al reflejar.
PARES_POSE = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
              (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]


def permutacion_espejo() -> list[int]:
    pose = list(range(33))
    for a, b in PARES_POSE:
        pose[a], pose[b] = b, a
    # La mano izquierda pasa a ser la derecha y viceversa.
    return pose + list(range(54, 75)) + list(range(33, 54))


class ModeloDemo(nn.Module):
    """(T, 225) cruda -> (probabilidades (2, 54), imagen TSSI (3, 128, 128))."""

    def __init__(self, red: nn.Module, bajo: torch.Tensor, alto: torch.Tensor):
        super().__init__()
        self.red = red
        self.register_buffer("bajo", bajo.clone().view(1, 3, 1, 1))
        self.register_buffer("alto", alto.clone().view(1, 3, 1, 1))
        self.register_buffer("media", ta.MEDIA_IMAGENET.clone().view(1, 3, 1, 1))
        self.register_buffer("desv", ta.DESV_IMAGENET.clone().view(1, 3, 1, 1))
        self.register_buffer("orden", torch.tensor(ta.ORDEN_TSSI, dtype=torch.long))
        self.register_buffer("perm", torch.tensor(permutacion_espejo(), dtype=torch.long))

    def espejo(self, s: torch.Tensor) -> torch.Tensor:
        p = s.reshape(-1, 75, 3)
        presente = p.abs().sum(dim=2) > 0          # los puntos ausentes son ceros y siguen en cero
        x = torch.where(presente, 1.0 - p[:, :, 0], p[:, :, 0])
        p = torch.stack([x, p[:, :, 1], p[:, :, 2]], dim=2)
        return p.index_select(1, self.perm).reshape(-1, 225)

    @staticmethod
    def normalizar(s: torch.Tensor) -> torch.Tensor:
        """Misma operación que tssi_aumento.normalizar, escrita para exportar."""
        pose = s[:, 0:99].reshape(-1, 33, 3)
        izquierda = s[:, 99:162].reshape(-1, 21, 3)
        derecha = s[:, 162:225].reshape(-1, 21, 3)

        hombro_izq, hombro_der = pose[:, 11, :], pose[:, 12, :]
        centro = ((hombro_izq + hombro_der) / 2).unsqueeze(1)
        ancho = torch.sqrt(((hombro_izq[:, :2] - hombro_der[:, :2]) ** 2).sum(dim=1))
        ancho = torch.clamp(ancho, min=1e-3).view(-1, 1, 1)
        hay_pose = (pose.abs().sum(dim=2).sum(dim=1) > 0).view(-1, 1, 1)
        pose = torch.where(hay_pose, (pose - centro) / ancho, pose)

        def mano_normalizada(mano):
            hay = (mano.abs().sum(dim=2).sum(dim=1) > 0).view(-1, 1, 1)
            centrada = mano - mano[:, 0:1, :]
            escala = torch.sqrt((centrada ** 2).sum(dim=2)).max(dim=1).values
            escala = torch.clamp(escala, min=1e-3).view(-1, 1, 1)
            return torch.where(hay, centrada / escala, mano)

        return torch.cat([pose, mano_normalizada(izquierda), mano_normalizada(derecha)], dim=1)

    def tssi(self, s: torch.Tensor) -> torch.Tensor:
        puntos = self.normalizar(s)                                 # (T, 75, 3)
        return puntos.index_select(1, self.orden).permute(2, 0, 1)  # (3, T, 100)

    def forward(self, secuencia: torch.Tensor):
        lote = torch.stack([self.tssi(secuencia), self.tssi(self.espejo(secuencia))])
        imagen = ((lote - self.bajo) / (self.alto - self.bajo)).clamp(0, 1)
        imagen = F.interpolate(imagen, size=(ta.TAMANO, ta.TAMANO), mode="bilinear",
                               align_corners=False)
        logits = self.red((imagen - self.media) / self.desv)
        return torch.softmax(logits, dim=1), imagen[0]


def construir() -> tuple[ModeloDemo, dict]:
    guardado = torch.load(RUTA_MODELO, map_location="cpu", weights_only=False)
    assert list(guardado["orden_tssi"]) == ta.ORDEN_TSSI
    assert guardado["tamano"] == ta.TAMANO
    red = ta.construir_densenet(len(guardado["clases"]), imagenet=False)
    red.load_state_dict(guardado["model_state"])
    return ModeloDemo(red, guardado["p_bajo"], guardado["p_alto"]).eval(), guardado


def exportar(modelo: ModeloDemo, ruta: Path) -> None:
    ejemplo = torch.rand(37, 225)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        modelo, (ejemplo,), str(ruta),
        input_names=["secuencia"], output_names=["probabilidades", "imagen"],
        dynamic_axes={"secuencia": {0: "T"}}, opset_version=17, dynamo=False,
    )


def escribir_clases(clases: list[str], grupo: dict[str, str], guardado: dict) -> None:
    datos = {
        "modelo": "SL-TSSI-DenseNet-121 con aumento de datos",
        "exactitud_validacion": round(float(guardado["acc_dev"]), 4),
        "clases": [{"id": c, "etiqueta": ETIQUETAS.get(c, c[:1].upper() + c[1:]),
                    "categoria": grupo[c]} for c in clases],
    }
    (DESTINO / "clases.json").write_text(json.dumps(datos, ensure_ascii=False, indent=1),
                                         encoding="utf-8")


@torch.no_grad()
def verificar(modelo: ModeloDemo, ruta: Path, datos) -> None:
    """Compara ONNX con el pipeline original de tssi_aumento sobre todo el test."""
    import onnxruntime as ort

    sesion = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
    guardado = torch.load(RUTA_MODELO, map_location="cpu", weights_only=False)
    bajo, alto = guardado["p_bajo"], guardado["p_alto"]

    reales = [e["gloss"] for e in datos.test]
    guardadas = json.loads((ta.DIR_AUG / "predicciones_test.json").read_text(encoding="utf-8"))
    dif_max, iguales, aciertos, aciertos_doble = 0.0, 0, 0, 0
    confianzas = []
    for e, real, previa in zip(datos.test, reales, guardadas["predichas"]):
        s = e["sign"].float()
        # Referencia: el código con el que se entrenó.
        entrada = ta.a_entrada_red(ta.escalar_a_imagen(ta.a_imagen_tssi(s), bajo, alto))
        ref = torch.softmax(modelo.red(entrada.unsqueeze(0)), 1)[0].numpy()
        prob, _ = sesion.run(None, {"secuencia": s.numpy()})
        dif_max = max(dif_max, float(np.abs(prob[0] - ref).max()))
        directa = datos.clases[int(prob[0].argmax())]
        iguales += directa == previa
        aciertos += directa == real
        mejor = np.maximum(prob[0], prob[1])
        acierto = datos.clases[int(mejor.argmax())] == real
        aciertos_doble += acierto
        confianzas.append((float(mejor.max()), acierto))

    n = len(reales)
    print(f"Videos de prueba: {n}")
    print(f"Diferencia máxima de probabilidad ONNX vs PyTorch: {dif_max:.2e}")
    print(f"Predicciones iguales a las guardadas en predicciones_test.json: {iguales}/{n}")
    print(f"Exactitud, una orientación (como el cuaderno): {aciertos / n:.1%}")
    print(f"Exactitud, mejor de dos orientaciones (como el demo): {aciertos_doble / n:.1%}")
    print("\nUmbral de confianza -> señas aceptadas / exactitud entre las aceptadas")
    for umbral in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        dentro = [a for c, a in confianzas if c >= umbral]
        print(f"  {umbral:.1f}: {len(dentro) / n:6.1%} / {sum(dentro) / max(1, len(dentro)):.1%}")


def main() -> None:
    modelo, guardado = construir()
    datos = ta.cargar_datos()
    assert guardado["clases"] == datos.clases
    ruta = DESTINO / "lsc_tssi.onnx"
    exportar(modelo, ruta)
    escribir_clases(datos.clases, datos.grupo, guardado)
    print(f"Modelo exportado: {ruta.relative_to(RAIZ)} ({ruta.stat().st_size / 1e6:.1f} MB)\n")
    verificar(modelo, ruta, datos)


if __name__ == "__main__":
    main()
