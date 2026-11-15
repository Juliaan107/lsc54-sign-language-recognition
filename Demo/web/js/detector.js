// MediaPipe en el navegador: los mismos dos modelos con los que se construyó el
// dataset (pose_landmarker_full y hand_landmarker), en modo video.
//
// Modo normal ("principal"): los modelos corren en el hilo de la página. En equipos con
// gráfica integrada es alrededor de un 20 % más rápido que usar un worker, y analizar más
// cuadros por segundo mejora el reconocimiento. El video no se traba porque el navegador
// lo muestra por su cuenta.
//
// Modo "hilo" (opcional, ?hilo en la dirección): los modelos corren en un worker y la
// página queda libre mientras se analiza. Conviene en equipos con buena tarjeta gráfica.

import { vectorDeCuadro } from './puntos.js';

const LADO_ANALISIS = 640;         // los videos de entrenamiento medían 640 px de ancho
const RUTA_WASM = new URL('../vendor/mediapipe/wasm', import.meta.url).href;

// Encuadre del zoom, en anchos de hombros. En los videos de entrenamiento el cuadro mide
// unos 4.8 anchos de hombros a lo ancho y 2.7 a lo alto; aquí se deja algo más de margen
// para que las manos no se salgan.
const ZOOM_ANCHO = 5.6, ZOOM_ARRIBA = 1.8, ZOOM_ABAJO = 1.7;
const CAJA_COMPLETA = { x: 0, y: 0, w: 1, h: 1 };

/** Devuelve el control al navegador un instante, para que pueda pintar la pantalla. */
const canal = new MessageChannel();
const enEspera = [];
canal.port1.onmessage = () => { const seguir = enEspera.shift(); if (seguir) seguir(); };
function ceder() {
  return new Promise((resolver) => { enEspera.push(resolver); canal.port2.postMessage(0); });
}

/** Convierte números seguidos (x, y, z, visibilidad) en listas de puntos. */
function desempaquetar(datos, cantidad) {
  if (!cantidad) return [];
  const porLista = datos.length / 4 / cantidad;
  const listas = [];
  for (let l = 0; l < cantidad; l++) {
    const lista = new Array(porLista);
    for (let i = 0; i < porLista; i++) {
      const k = (l * porLista + i) * 4;
      lista[i] = { x: datos[k], y: datos[k + 1], z: datos[k + 2], visibility: datos[k + 3] };
    }
    listas.push(lista);
  }
  return listas;
}

/** Pose como 33 x (x, y, z, visibilidad), el formato que usa el dibujo. */
function posePlana(pose) {
  if (!pose) return null;
  const plano = new Float32Array(33 * 4);
  for (let i = 0; i < 33; i++) {
    plano[i * 4] = pose[i].x;
    plano[i * 4 + 1] = pose[i].y;
    plano[i * 4 + 2] = pose[i].z;
    plano[i * 4 + 3] = pose[i].visibility === undefined ? 1 : pose[i].visibility;
  }
  return plano;
}

class Hilo {
  constructor() {
    this.worker = new Worker(new URL('./detector.worker.js', import.meta.url), { type: 'module' });
    this.pendientes = new Map();
    this.siguiente = 1;
    this.alCargar = null;
    this.worker.onmessage = (mensaje) => {
      const m = mensaje.data;
      if (m.tipo === 'listo') { this.alCargar.resolver(); return; }
      const p = this.pendientes.get(m.id);
      if (m.tipo === 'error' && !p) { if (this.alCargar) this.alCargar.rechazar(new Error(m.mensaje)); return; }
      this.pendientes.delete(m.id);
      if (m.tipo === 'error') p.rechazar(new Error(m.mensaje)); else p.resolver(m);
    };
    this.worker.onerror = (e) => {
      const error = new Error(e.message || 'El worker de MediaPipe falló');
      if (this.alCargar) this.alCargar.rechazar(error);
      for (const p of this.pendientes.values()) p.rechazar(error);
      this.pendientes.clear();
    };
  }

  cargar(modeloPose, modeloManos, delegado) {
    return new Promise((resolver, rechazar) => {
      this.alCargar = { resolver, rechazar };
      this.worker.postMessage({ tipo: 'cargar', modeloPose, modeloManos, delegado });   // se envían copias
    });
  }

  detectar(imagen, t) {
    const id = this.siguiente++;
    return new Promise((resolver, rechazar) => {
      this.pendientes.set(id, { resolver, rechazar });
      this.worker.postMessage({ tipo: 'detectar', id, imagen, t }, [imagen]);
    });
  }

  cerrar() { this.worker.terminate(); }
}

export class Detector {
  constructor() {
    this.modo = null;          // 'hilo' | 'principal'
    this.delegado = null;      // 'GPU' | 'CPU'
    this.hilo = null;
    this.pose = null;
    this.manos = null;
    this.ultimoTiempo = 0;
    this.lienzo = null;        // copia reducida del cuadro que se analiza
    this.ctx = null;
    this.reducir = true;       // analizar una copia reducida (false: el video tal cual)
    this.zoom = true;          // acercar el análisis a la persona cuando se ve pequeña
    this.cajaFija = false;     // la página lo activa durante una seña: ahí no se reencuadra
    this.caja = { ...CAJA_COMPLETA };   // parte del cuadro que se analiza (fracciones del cuadro)
    this.sinPersona = 0;
  }

  /**
   * @param {Uint8Array} modeloPose   contenido de pose_landmarker_full.task
   * @param {Uint8Array} modeloManos  contenido de hand_landmarker.task
   * @param {{delegado?: 'GPU'|'CPU', enHilo?: boolean}} opciones
   */
  async cargar(modeloPose, modeloManos, opciones = {}) {
    const delegados = opciones.delegado === 'CPU' ? ['CPU'] : ['GPU', 'CPU'];
    const admiteHilos = !!opciones.enHilo && typeof Worker !== 'undefined' &&
      typeof OffscreenCanvas !== 'undefined' && typeof createImageBitmap === 'function';

    let error;
    for (const delegado of delegados) {
      if (admiteHilos) {
        try {
          await this.cargarHilo(modeloPose, modeloManos, delegado);
          this.modo = 'hilo';
          this.delegado = delegado;
          return;
        } catch (e) {
          error = e;
          console.warn('MediaPipe no pudo iniciar en un hilo aparte; se usa el hilo de la página.', e);
          this.cerrarHilo();
        }
      }
      try {
        await this.cargarPrincipal(modeloPose, modeloManos, delegado);
        this.modo = 'principal';
        this.delegado = delegado;
        return;
      } catch (e) {
        error = e;
      }
    }
    throw error;
  }

  async cargarHilo(modeloPose, modeloManos, delegado) {
    this.hilo = new Hilo();
    const limite = new Promise((_, rechazar) => setTimeout(() => rechazar(new Error('MediaPipe tardó demasiado en iniciar')), 60000));
    await Promise.race([this.hilo.cargar(modeloPose.buffer, modeloManos.buffer, delegado), limite]);
  }

  cerrarHilo() {
    if (this.hilo) this.hilo.cerrar();
    this.hilo = null;
  }

  async cargarPrincipal(modeloPose, modeloManos, delegado) {
    const { FilesetResolver, PoseLandmarker, HandLandmarker } = await import('../vendor/mediapipe/vision_bundle.mjs');
    const archivos = await FilesetResolver.forVisionTasks(RUTA_WASM);
    this.pose = await PoseLandmarker.createFromOptions(archivos, {
      baseOptions: { modelAssetBuffer: modeloPose, delegate: delegado },
      runningMode: 'VIDEO',
      numPoses: 1,
    });
    this.manos = await HandLandmarker.createFromOptions(archivos, {
      baseOptions: { modelAssetBuffer: modeloManos, delegate: delegado },
      runningMode: 'VIDEO',
      numHands: 2,
    });
    // El primer análisis siempre tarda (compila en la tarjeta gráfica); se hace por adelantado.
    const lienzo = document.createElement('canvas');
    lienzo.width = 640;
    lienzo.height = 360;
    const ctx = lienzo.getContext('2d');
    ctx.fillStyle = '#777';
    ctx.fillRect(0, 0, lienzo.width, lienzo.height);
    for (let i = 0; i < 3; i++) {
      this.pose.detectForVideo(lienzo, ++this.ultimoTiempo);
      this.manos.detectForVideo(lienzo, this.ultimoTiempo);
    }
  }

  /** Recuadro que se analiza, en pixeles del video: [x, y, ancho, alto]. */
  recuadro(video) {
    const W = video.videoWidth, H = video.videoHeight, c = this.caja;
    return [Math.round(c.x * W), Math.round(c.y * H), Math.max(2, Math.round(c.w * W)), Math.max(2, Math.round(c.h * H))];
  }

  /** Tamaño con que se analiza el recuadro: su lado mayor se reduce a LADO_ANALISIS. */
  tamano(ancho, alto) {
    const factor = Math.min(1, LADO_ANALISIS / Math.max(ancho, alto));
    return [Math.max(2, Math.round(ancho * factor)), Math.max(2, Math.round(alto * factor))];
  }

  /** Copia del recuadro actual para el worker. */
  copiar(video) {
    const [x, y, w, h] = this.recuadro(video);
    const [ancho, alto] = this.tamano(w, h);
    return createImageBitmap(video, x, y, w, h, { resizeWidth: ancho, resizeHeight: alto, resizeQuality: 'low' })
      .catch(() => { this.caja = { ...CAJA_COMPLETA }; return createImageBitmap(video); });
  }

  /**
   * Congela el recuadro actual en un lienzo pequeño. Así los dos modelos analizan
   * exactamente el mismo cuadro (aunque entre uno y otro se le ceda el turno al
   * navegador) y trabajan con el tamaño de los videos de entrenamiento.
   */
  congelar(video) {
    const [x, y, w, h] = this.recuadro(video);
    const [ancho, alto] = this.tamano(w, h);
    if (!this.lienzo) {
      this.lienzo = document.createElement('canvas');
      this.ctx = this.lienzo.getContext('2d', { alpha: false });
    }
    if (this.lienzo.width !== ancho || this.lienzo.height !== alto) {
      this.lienzo.width = ancho;
      this.lienzo.height = alto;
    }
    this.ctx.drawImage(video, x, y, w, h, 0, 0, ancho, alto);
    return this.lienzo;
  }

  /** Lleva los puntos del recuadro analizado a coordenadas del cuadro completo. */
  aCuadroCompleto(puntos, c) {
    if (!puntos || (c.x === 0 && c.y === 0 && c.w === 1 && c.h === 1)) return puntos;
    return puntos.map((p) => ({ x: c.x + p.x * c.w, y: c.y + p.y * c.h, z: p.z * c.w, visibility: p.visibility }));
  }

  /**
   * Decide qué parte del cuadro analizar la próxima vez.
   *
   * Los videos de entrenamiento muestran a la persona de la cintura hacia arriba. Si en
   * la cámara aparece pequeña (está lejos, o el celular está en vertical), sus manos son
   * muy chicas para MediaPipe y las pierde. Por eso se analiza solo un recuadro alrededor
   * de la persona, con un encuadre parecido al de los videos de entrenamiento.
   *
   * El recuadro se deja quieto mientras la persona no se mueva de su sitio: cambiarlo
   * desorienta por un instante el seguimiento de MediaPipe.
   */
  reencuadrar(pose, video) {
    if (!this.zoom) return;
    if (!pose) {
      if (++this.sinPersona >= 8) this.caja = { ...CAJA_COMPLETA };
      return;
    }
    this.sinPersona = 0;
    if (this.cajaFija) return;
    const izq = pose[11], der = pose[12];
    if (!(izq.visibility > 0.5 && der.visibility > 0.5)) return;

    const W = video.videoWidth, H = video.videoHeight;
    const hombros = Math.hypot((izq.x - der.x) * W, (izq.y - der.y) * H);       // en pixeles
    if (!(hombros > 8)) return;
    const cx = ((izq.x + der.x) / 2) * W, cy = ((izq.y + der.y) / 2) * H;
    const x0 = Math.max(0, cx - (ZOOM_ANCHO / 2) * hombros), x1 = Math.min(W, cx + (ZOOM_ANCHO / 2) * hombros);
    const y0 = Math.max(0, cy - ZOOM_ARRIBA * hombros), y1 = Math.min(H, cy + ZOOM_ABAJO * hombros);
    let deseada = { x: x0 / W, y: y0 / H, w: (x1 - x0) / W, h: (y1 - y0) / H };
    if (deseada.w > 0.85 && deseada.h > 0.85) deseada = { ...CAJA_COMPLETA };   // casi todo: mejor todo
    if (!(deseada.w > 0.05 && deseada.h > 0.05)) return;

    const c = this.caja;
    const corrida = Math.max(
      Math.abs(deseada.x + deseada.w / 2 - c.x - c.w / 2) / c.w,
      Math.abs(deseada.y + deseada.h / 2 - c.y - c.h / 2) / c.h);
    const escala = Math.max(deseada.w / c.w, c.w / deseada.w, deseada.h / c.h, c.h / deseada.h);
    if (corrida > 0.12 || escala > 1.25) this.caja = deseada;
  }

  /**
   * Analiza el cuadro actual del video.
   * @param {(pose: Float32Array|null) => void} [alTenerPose]  se llama en cuanto está el
   *   cuerpo, antes de buscar las manos (para ir moviendo el esqueleto)
   * @returns {Promise<{ pose: Float32Array|null, vector: Float32Array }>}
   *   pose: 33 x (x, y, z, visibilidad); vector: los 225 números que usa el modelo.
   *   Siempre en coordenadas del cuadro completo.
   */
  async detectar(video, ahora, alTenerPose) {
    // MediaPipe exige marcas de tiempo estrictamente crecientes.
    const t = Math.max(ahora, this.ultimoTiempo + 1);
    this.ultimoTiempo = t;

    let pose, manos, plana;
    if (this.modo === 'hilo') {
      const caja = this.caja;
      const r = await this.hilo.detectar(await this.copiar(video), t);
      pose = this.aCuadroCompleto(r.cantidadPose ? desempaquetar(r.pose, r.cantidadPose)[0] : null, caja);
      manos = desempaquetar(r.manos, r.cantidadManos).map((m) => this.aCuadroCompleto(m, caja));
      plana = posePlana(pose);
    } else {
      if (!this.reducir) this.caja = { ...CAJA_COMPLETA };
      const caja = this.caja;
      const fuente = this.reducir ? this.congelar(video) : video;
      const rPose = this.pose.detectForVideo(fuente, t);
      pose = this.aCuadroCompleto(rPose.landmarks && rPose.landmarks.length ? rPose.landmarks[0] : null, caja);
      plana = posePlana(pose);
      if (alTenerPose) alTenerPose(plana);
      if (this.reducir) await ceder();              // deja pintar antes de seguir con las manos
      const rManos = this.manos.detectForVideo(fuente, t);
      manos = (rManos.landmarks || []).map((m) => this.aCuadroCompleto(m, caja));
    }
    this.reencuadrar(pose, video);
    return { pose: plana, vector: vectorDeCuadro(pose, manos) };
  }
}
