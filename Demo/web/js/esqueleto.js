// Dibuja sobre el video los puntos que detecta MediaPipe: el cuerpo y las dos manos.
// Las manos se dibujan a partir del mismo vector que recibe el modelo, así que lo que
// se ve en pantalla es exactamente lo que se clasifica.

const HUESOS_CUERPO = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],     // hombros y brazos
  [11, 23], [12, 24], [23, 24],                          // tronco
  [23, 25], [25, 27], [24, 26], [26, 28],                // piernas (si se ven)
];
const PUNTOS_CUERPO = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
const PUNTOS_CARA = [0, 2, 5, 7, 8, 9, 10];
const HUESOS_CARA = [[0, 2], [2, 7], [0, 5], [5, 8], [9, 10]];

const HUESOS_MANO = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const MANO_IZQ = 99, MANO_DER = 162;

const COLORES = {
  cuerpo: '#f4efe6',
  izquierda: '#ff8a5c',
  derecha: '#5ee6c8',
  borde: 'rgba(8, 10, 14, 0.55)',
  halo: 'rgba(255, 193, 94, 0.45)',
};

function hayMano(vector, desde) {
  for (let i = desde; i < desde + 63; i++) if (vector[i] !== 0) return true;
  return false;
}

export class Esqueleto {
  /** @param {HTMLCanvasElement} lienzo */
  constructor(lienzo) {
    this.lienzo = lienzo;
    this.ctx = lienzo.getContext('2d');
    this.ancho = 0;
    this.alto = 0;
    this.escala = 1;
  }

  /** Ajusta el lienzo al tamaño con que se muestra en pantalla. */
  ajustar() {
    const caja = this.lienzo.getBoundingClientRect();
    const densidad = Math.min(window.devicePixelRatio || 1, 2);
    const ancho = Math.max(1, Math.round(caja.width * densidad));
    const alto = Math.max(1, Math.round(caja.height * densidad));
    if (ancho !== this.lienzo.width || alto !== this.lienzo.height) {
      this.lienzo.width = ancho;
      this.lienzo.height = alto;
    }
    this.ancho = ancho;
    this.alto = alto;
    this.escala = densidad;
  }

  limpiar() {
    this.ctx.clearRect(0, 0, this.lienzo.width, this.lienzo.height);
  }

  /**
   * @param {{pose: Float32Array|null, vector: Float32Array}|null} puntos
   *        pose: 33 x (x, y, z, visibilidad); vector: los 225 números del modelo
   * @param {number} anchoVideo   tamaño real del video, para calcular el recorte
   * @param {number} altoVideo
   * @param {boolean} capturando  resalta el esqueleto mientras se graba una seña
   */
  dibujar(puntos, anchoVideo, altoVideo, capturando) {
    const ctx = this.ctx;
    this.limpiar();
    if (!puntos || !anchoVideo || !altoVideo) return;

    // El video se muestra con object-fit: cover; se replica el mismo recorte.
    const factor = Math.max(this.ancho / anchoVideo, this.alto / altoVideo);
    const anchoFinal = anchoVideo * factor, altoFinal = altoVideo * factor;
    const dx = (this.ancho - anchoFinal) / 2;
    const dy = (this.alto - altoFinal) / 2;
    const X = (x) => dx + x * anchoFinal;
    const Y = (y) => dy + y * altoFinal;

    const base = Math.max(this.ancho, this.alto);
    const grosor = Math.max(2.5 * this.escala, base / 230);
    const radio = grosor * 1.15;
    const borde = capturando ? COLORES.halo : COLORES.borde;
    const extra = capturando ? grosor * 2.2 : grosor * 0.9;      // un halo en vez de sombras (más liviano)

    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    /** Traza varios segmentos de una vez: primero el borde o halo, luego el color. */
    const trazar = (segmentos, color, ancho) => {
      if (!segmentos.length) return;
      const camino = new Path2D();
      for (const [x1, y1, x2, y2] of segmentos) { camino.moveTo(x1, y1); camino.lineTo(x2, y2); }
      ctx.strokeStyle = borde;
      ctx.lineWidth = ancho + extra;
      ctx.stroke(camino);
      ctx.strokeStyle = color;
      ctx.lineWidth = ancho;
      ctx.stroke(camino);
    };
    const puntear = (centros, color, r) => {
      if (!centros.length) return;
      const fondo = new Path2D(), frente = new Path2D();
      for (const [x, y] of centros) {
        fondo.moveTo(x + r * 1.45, y); fondo.arc(x, y, r * 1.45, 0, Math.PI * 2);
        frente.moveTo(x + r, y); frente.arc(x, y, r, 0, Math.PI * 2);
      }
      ctx.fillStyle = '#0c0e13';
      ctx.fill(fondo);
      ctx.fillStyle = color;
      ctx.fill(frente);
    };

    // --- Cuerpo ---
    const pose = puntos.pose;
    if (pose) {
      const px = (i) => pose[i * 4], py = (i) => pose[i * 4 + 1];
      // Un hueso se dibuja si MediaPipe lo ubica con cierta confianza, aunque uno de sus
      // extremos quede fuera del cuadro (por ejemplo, el brazo cuando la mano está abajo).
      const fiable = (i, minimo) => pose[i * 4 + 3] > minimo;
      const dentro = (i) => py(i) < 1.02 && py(i) > -0.02 && px(i) > -0.02 && px(i) < 1.02;
      const segmento = (a, b) => [X(px(a)), Y(py(a)), X(px(b)), Y(py(b))];

      trazar(HUESOS_CUERPO.filter(([a, b]) => fiable(a, 0.25) && fiable(b, 0.25) && (dentro(a) || dentro(b)))
        .map(([a, b]) => segmento(a, b)), COLORES.cuerpo, grosor);
      trazar(HUESOS_CARA.filter(([a, b]) => fiable(a, 0.5) && fiable(b, 0.5)).map(([a, b]) => segmento(a, b)),
        COLORES.cuerpo, grosor * 0.5);
      puntear(PUNTOS_CUERPO.filter((i) => fiable(i, 0.25) && dentro(i)).map((i) => [X(px(i)), Y(py(i))]),
        COLORES.cuerpo, radio * 1.05);
      puntear(PUNTOS_CARA.filter((i) => fiable(i, 0.5) && dentro(i)).map((i) => [X(px(i)), Y(py(i))]),
        COLORES.cuerpo, radio * 0.55);
    }

    // --- Manos ---
    const vector = puntos.vector;
    for (const [desde, color] of [[MANO_IZQ, COLORES.izquierda], [MANO_DER, COLORES.derecha]]) {
      if (!hayMano(vector, desde)) continue;
      const px = (i) => X(vector[desde + i * 3]);
      const py = (i) => Y(vector[desde + i * 3 + 1]);
      trazar(HUESOS_MANO.map(([a, b]) => [px(a), py(a), px(b), py(b)]), color, grosor * 0.7);
      const centros = [];
      for (let i = 0; i < 21; i++) centros.push([px(i), py(i)]);
      puntear(centros, color, radio * 0.6);
    }
  }
}

// ---------------------------------------------------------------------------------
// Movimiento del esqueleto entre un análisis y el siguiente.
//
// MediaPipe entrega puntos unas 10 a 30 veces por segundo y siempre con algo de atraso
// (el cuadro que analizó ya pasó). Para que el esqueleto no se vea a saltos ni rezagado:
//   1. se estima hacia dónde va cada punto con su velocidad reciente (un poco, no todo), y
//   2. lo que se dibuja se acerca con suavidad a esa estimación en cada pintada.

const SUAVIZADO_MS = 40;          // qué tan rápido alcanza el dibujo a la estimación
const ADELANTO_MAX_MS = 110;      // cuánto se adelanta como mucho
const GANANCIA = 0.7;             // fracción del adelanto que se aplica
const SALTO_MAX = 0.06;           // adelanto máximo por coordenada (fracción del cuadro)
const VIGENCIA_MS = 280;          // una velocidad más vieja que esto no se usa

class Pista {
  /** @param {number} paso  números por punto (4 en el cuerpo, 3 en las manos) */
  constructor(paso) {
    this.paso = paso;
    this.actual = null; this.tActual = 0;
    this.previa = null; this.tPrevia = 0;
    this.mostrada = null;
  }

  poner(datos, t) {
    this.previa = this.actual; this.tPrevia = this.tActual;
    this.actual = datos; this.tActual = t;
    if (!datos) this.mostrada = null;
  }

  /** Avanza el dibujo. Devuelve los puntos a mostrar (o null). */
  avanzar(ahora, dt) {
    const a = this.actual;
    if (!a) return null;
    const p = this.previa;
    const intervalo = this.tActual - this.tPrevia;
    const conVelocidad = p && intervalo > 0 && intervalo < VIGENCIA_MS;
    const adelanto = conVelocidad ? (GANANCIA * Math.min(ADELANTO_MAX_MS, Math.max(0, ahora - this.tActual))) / intervalo : 0;
    const f = this.mostrada ? 1 - Math.exp(-dt / SUAVIZADO_MS) : 1;
    const m = this.mostrada || new Float32Array(a);
    for (let i = 0; i < a.length; i += this.paso) {
      for (let k = 0; k < 2; k++) {
        let meta = a[i + k];
        if (adelanto) meta += Math.max(-SALTO_MAX, Math.min(SALTO_MAX, (a[i + k] - p[i + k]) * adelanto));
        m[i + k] += (meta - m[i + k]) * f;
      }
      for (let k = 2; k < this.paso; k++) m[i + k] = a[i + k];
    }
    this.mostrada = m;
    return m;
  }
}

export class Seguidor {
  constructor() {
    this.cuerpo = new Pista(4);
    this.manos = [new Pista(3), new Pista(3)];
    this.vector = new Float32Array(MANO_DER + 63);
  }

  /** @param {Float32Array|null} pose  33 x (x, y, z, visibilidad) */
  ponerCuerpo(pose, t) { this.cuerpo.poner(pose, t); }

  /** @param {Float32Array} vector  los 225 números; de aquí salen las dos manos */
  ponerManos(vector, t) {
    [MANO_IZQ, MANO_DER].forEach((desde, i) => {
      this.manos[i].poner(hayMano(vector, desde) ? vector.slice(desde, desde + 63) : null, t);
    });
  }

  reiniciar() {
    this.cuerpo = new Pista(4);
    this.manos = [new Pista(3), new Pista(3)];
  }

  /** @returns {{pose: Float32Array|null, vector: Float32Array}} lo que hay que dibujar ahora */
  avanzar(ahora, dt) {
    const pose = this.cuerpo.avanzar(ahora, dt);
    [MANO_IZQ, MANO_DER].forEach((desde, i) => {
      const mano = this.manos[i].avanzar(ahora, dt);
      if (mano) this.vector.set(mano, desde); else this.vector.fill(0, desde, desde + 63);
    });
    return { pose, vector: this.vector };
  }
}
