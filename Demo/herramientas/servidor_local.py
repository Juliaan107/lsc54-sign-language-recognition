"""Servidor local para probar el demo en este computador.

    python Demo/herramientas/servidor_local.py            # http://localhost:8000
    python Demo/herramientas/servidor_local.py --puerto 9000
    python Demo/herramientas/servidor_local.py --video Demo/video_ejemplo.mp4

El navegador solo permite usar la cámara en https o en localhost, así que la página
debe abrirse como http://localhost:8000 (no con doble clic sobre index.html).

Con --video, el archivo queda disponible en /_prueba/video y la página puede usarlo en
lugar de la cámara: http://localhost:8000/?video=/_prueba/video
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import re
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"

mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/octet-stream", ".task")
mimetypes.add_type("application/octet-stream", ".onnx")
mimetypes.add_type("font/woff2", ".woff2")


class Manejador(SimpleHTTPRequestHandler):
    video: Path | None = None
    aislar = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def translate_path(self, path: str) -> str:
        if self.video is not None and path.split("?")[0] == "/_prueba/video":
            return str(self.video)
        return super().translate_path(path)

    def end_headers(self):
        if self.aislar:
            # Permiten que el clasificador use varios hilos (igual que en el sitio publicado).
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def send_head(self):
        # Los videos necesitan peticiones por rangos; http.server no las trae.
        rango = self.headers.get("Range")
        ruta = Path(self.translate_path(self.path))
        if not rango or not ruta.is_file():
            return super().send_head()
        m = re.match(r"bytes=(\d*)-(\d*)$", rango.strip())
        if not m:
            return super().send_head()
        total = ruta.stat().st_size
        inicio = int(m.group(1)) if m.group(1) else max(0, total - int(m.group(2) or 0))
        fin = int(m.group(2)) if m.group(1) and m.group(2) else total - 1
        fin = min(fin, total - 1)
        if inicio > fin:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return None
        archivo = open(ruta, "rb")
        archivo.seek(inicio)
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        tipo = "video/mp4" if ruta == self.video else self.guess_type(str(ruta))
        self.send_header("Content-Type", tipo)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {inicio}-{fin}/{total}")
        self.send_header("Content-Length", str(fin - inicio + 1))
        self.end_headers()
        self._restante = fin - inicio + 1
        return archivo

    def copyfile(self, source, outputfile):
        restante = getattr(self, "_restante", None)
        if restante is None:
            return super().copyfile(source, outputfile)
        self._restante = None
        while restante > 0:
            bloque = source.read(min(64 * 1024, restante))
            if not bloque:
                break
            outputfile.write(bloque)
            restante -= len(bloque)

    def guess_type(self, path):
        if self.video is not None and Path(path) == self.video:
            return "video/mp4"
        return super().guess_type(path)

    def log_message(self, formato, *args):
        if os.environ.get("DEMO_SILENCIO"):
            return
        super().log_message(formato, *args)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--puerto", type=int, default=8000)
    p.add_argument("--video", type=Path, default=None, help="video que reemplaza a la cámara en /_prueba/video")
    p.add_argument("--sin-aislamiento", action="store_true",
                   help="no enviar las cabeceras COOP/COEP (el clasificador usa un solo hilo)")
    args = p.parse_args()

    Manejador.video = args.video.resolve() if args.video else None
    Manejador.aislar = not args.sin_aislamiento
    servidor = ThreadingHTTPServer(("127.0.0.1", args.puerto), Manejador)
    print(f"Demo disponible en http://localhost:{args.puerto}")
    if Manejador.video:
        print(f"Con el video de prueba: http://localhost:{args.puerto}/?video=/_prueba/video")
    print("Ctrl+C para detener.")
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
