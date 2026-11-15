// Cinta de fotogramas: algunos instantes de la seña, en orden, dibujados solo con el
// esqueleto (sin el video). Se pinta una vez por seña, así que no afecta el rendimiento.

import { DIMENSION } from './puntos.js';

const HUESOS_CUERPO = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24]];
const HUESOS_MANO = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const MANOS = [[99, '#ff8a5c'], [162, '#5ee6c8']];
const COLOR_CUERPO = '#f4efe6';
const PROPORCION_CELDA = 4 / 5;          // ancho / alto de cada fotograma
const SEPARACION = 6;                    // px entre fotogramas
// Los puntos vienen en un cuadro 16:9: x e y son fracciones de lados distintos.
const EX = 16, EY = 9;

function hayMano(v, desde) {
  for (let i = desde; i < desde + 63; i++) if (v[i] !== 0) return true;
  return false;
}

/**
 * @param {HTMLCanvasElement} lienzo
 * @param {Float32Array} datos   T x 225, la seña tal como salió del segmentador
 * @param {number} T
 */
export function pintarCinta(lienzo, datos, T) {
  const anchoCss = lienzo.clientWidth || lienzo.parentElement.clientWidth;
  if (!anchoCss || !T) return;
  const cantidad = Math.max(4, Math.min(7, Math.round(anchoCss / 88)));
  const anchoCelda = (anchoCss - SEPARACION * (cantidad - 1)) / cantidad;
  const altoCss = anchoCelda / PROPORCION_CELDA;
  const densidad = Math.min(window.devicePixelRatio || 1, 2);
  lienzo.style.height = `${altoCss}px`;
  lienzo.width = Math.round(anchoCss * densidad);
  lienzo.height = Math.round(altoCss * densidad);
  const ctx = lienzo.getContext('2d');
  ctx.setTransform(densidad, 0, 0, densidad, 0, 0);
  ctx.clearRect(0, 0, anchoCss, altoCss);

  // Fotogramas repartidos por la seña (el recorte trae reposo al principio y al final).
  const cuadros = [];
  for (let i = 0; i < cantidad; i++) {
    const fila = Math.round((0.1 + (0.8 * i) / (cantidad - 1)) * (T - 1));
    cuadros.push(datos.subarray(fila * DIMENSION, (fila + 1) * DIMENSION));
  }

  // Un mismo encuadre para todos, para que el cuerpo no salte de un fotograma a otro.
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity, hombros = 0;
  const incluir = (x, y) => {
    x0 = Math.min(x0, x * EX); x1 = Math.max(x1, x * EX);
    y0 = Math.min(y0, y * EY); y1 = Math.max(y1, y * EY);
  };
  for (const v of cuadros) {
    if (v[33] === 0 && v[36] === 0) continue;
    hombros = Math.max(hombros, Math.hypot((v[33] - v[36]) * EX, (v[34] - v[37]) * EY));
    const tope = Math.max(v[34], v[37]) + (1.45 * hombros) / EY;     // hasta cerca de la cintura
    for (const i of [0, 7, 8, 11, 12, 13, 14, 15, 16]) incluir(v[i * 3], Math.min(v[i * 3 + 1], tope));
    for (const [desde] of MANOS) {
      if (!hayMano(v, desde)) continue;
      for (let i = 0; i < 21; i++) incluir(v[desde + i * 3], Math.min(v[desde + i * 3 + 1], tope));
    }
  }
  if (!hombros) return;
  y0 -= hombros * 0.5;                                  // espacio para la cabeza
  const margen = hombros * 0.12;
  x0 -= margen; x1 += margen; y1 += margen;
  const escala = Math.min(anchoCelda / (x1 - x0), altoCss / (y1 - y0));
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const grosor = Math.max(1.6, anchoCelda / 30);

  cuadros.forEach((v, n) => {
    const izquierda = n * (anchoCelda + SEPARACION);
    const X = (x) => izquierda + anchoCelda / 2 + (x * EX - cx) * escala;
    const Y = (y) => altoCss / 2 + (y * EY - cy) * escala;

    ctx.save();
    ctx.beginPath();
    ctx.roundRect(izquierda, 0, anchoCelda, altoCss, 9);
    ctx.clip();
    const fondo = ctx.createLinearGradient(0, 0, 0, altoCss);
    fondo.addColorStop(0, '#1d2140');
    fondo.addColorStop(1, '#121527');
    ctx.fillStyle = fondo;
    ctx.fillRect(izquierda, 0, anchoCelda, altoCss);

    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    if (v[33] !== 0 || v[36] !== 0) {
      // Cuerpo
      ctx.strokeStyle = COLOR_CUERPO;
      ctx.globalAlpha = 0.85;
      ctx.lineWidth = grosor;
      ctx.beginPath();
      for (const [a, b] of HUESOS_CUERPO) {
        ctx.moveTo(X(v[a * 3]), Y(v[a * 3 + 1]));
        ctx.lineTo(X(v[b * 3]), Y(v[b * 3 + 1]));
      }
      ctx.stroke();
      // Cabeza: un círculo alrededor de la nariz, del tamaño que marcan las orejas
      const radio = Math.max(Math.hypot((v[21] - v[24]) * EX, (v[22] - v[25]) * EY) * 0.6, hombros * 0.2) * escala;
      ctx.beginPath();
      ctx.arc(X(v[0]), Y(v[1]), radio, 0, Math.PI * 2);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = COLOR_CUERPO;
      ctx.beginPath();
      for (const i of [11, 12, 13, 14, 15, 16]) {
        ctx.moveTo(X(v[i * 3]) + grosor, Y(v[i * 3 + 1]));
        ctx.arc(X(v[i * 3]), Y(v[i * 3 + 1]), grosor, 0, Math.PI * 2);
      }
      ctx.fill();
    }
    // Manos
    for (const [desde, color] of MANOS) {
      if (!hayMano(v, desde)) continue;
      const px = (i) => X(v[desde + i * 3]), py = (i) => Y(v[desde + i * 3 + 1]);
      ctx.strokeStyle = color;
      ctx.lineWidth = grosor * 0.8;
      ctx.beginPath();
      for (const [a, b] of HUESOS_MANO) { ctx.moveTo(px(a), py(a)); ctx.lineTo(px(b), py(b)); }
      ctx.stroke();
    }
    ctx.restore();

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(izquierda + 0.5, 0.5, anchoCelda - 1, altoCss - 1, 9);
    ctx.stroke();
  });
}
