"""Prepara los datos que usan las pruebas del demo.

    python Demo/herramientas/pruebas/preparar_datos.py <carpeta_salida> [video]

Escribe en <carpeta_salida>:

- flujo_test.npy / flujo_test.json: todos los videos del conjunto de prueba pegados uno
  tras otro, como si fueran una sola grabación continua, con la glosa de cada tramo.
- video.npy (solo si se pasa un video): los puntos de MediaPipe (Python) de ese video,
  extraídos con el mismo código que construyó el dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ))


def main() -> None:
    salida = Path(sys.argv[1])
    salida.mkdir(parents=True, exist_ok=True)

    import tssi_aumento as ta

    datos = ta.cargar_datos()
    tramos, matrices = [], []
    for e in datos.test:
        s = e["sign"].numpy().astype(np.float32)
        tramos.append({"glosa": e["gloss"], "cuadros": int(s.shape[0]), "nombre": e["name"]})
        matrices.append(s)
    np.save(salida / "flujo_test.npy", np.concatenate(matrices))
    (salida / "flujo_test.json").write_text(
        json.dumps({"clases": datos.clases, "tramos": tramos}, ensure_ascii=False), encoding="utf-8")
    print(f"flujo_test: {len(tramos)} videos, {sum(t['cuadros'] for t in tramos)} cuadros")

    if len(sys.argv) > 2:
        import lsc_pipeline as lp

        puntos = lp.extract_features(Path(sys.argv[2]))
        np.save(salida / "video.npy", puntos)
        print(f"video: {puntos.shape[0]} cuadros")


if __name__ == "__main__":
    main()
