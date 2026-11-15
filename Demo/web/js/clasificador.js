// Interfaz del hilo principal con el worker que corre el modelo.

export class Clasificador {
  constructor() {
    this.worker = new Worker(new URL('./clasificador.worker.js', import.meta.url), { type: 'module' });
    this.pendientes = new Map();
    this.siguiente = 1;
    this.hilos = 1;
    this.listo = new Promise((resolver, rechazar) => {
      this._resolverListo = resolver;
      this._rechazarListo = rechazar;
    });
    this.worker.onmessage = (mensaje) => this.recibir(mensaje.data);
    this.worker.onerror = (e) => this._rechazarListo(new Error(e.message || 'No se pudo iniciar el clasificador'));
  }

  recibir(m) {
    if (m.tipo === 'listo') {
      this.hilos = m.hilos;
      this._resolverListo(m);
    } else if (m.tipo === 'resultado') {
      const p = this.pendientes.get(m.id);
      this.pendientes.delete(m.id);
      if (p) p.resolver(m);
    } else if (m.tipo === 'error') {
      const p = this.pendientes.get(m.id);
      if (p) {
        this.pendientes.delete(m.id);
        p.rechazar(new Error(m.mensaje));
      } else {
        this._rechazarListo(new Error(m.mensaje));
      }
    }
  }

  /** @param {ArrayBuffer} modelo contenido de lsc_tssi.onnx */
  cargar(modelo) {
    this.worker.postMessage({ tipo: 'cargar', modelo }, [modelo]);
    return this.listo;
  }

  /**
   * @param {Float32Array} datos matriz cruda (T x 225)
   * @returns {Promise<{probabilidades: Float32Array, imagen: Float32Array, ms: number}>}
   */
  clasificar(datos, T) {
    const id = this.siguiente++;
    return new Promise((resolver, rechazar) => {
      this.pendientes.set(id, { resolver, rechazar });
      this.worker.postMessage({ tipo: 'clasificar', id, datos, T }, [datos.buffer]);
    });
  }
}
