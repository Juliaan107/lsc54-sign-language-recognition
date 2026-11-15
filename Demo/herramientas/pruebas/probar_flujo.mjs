// Prueba, fuera del navegador, la misma lógica que usa la página: el segmentador
// (js/segmentador.js), la decisión (js/decision.js) y el modelo ONNX.
//
//   node probar_flujo.mjs video <video.npy> [fps]
//       Recorre los puntos de un video como si llegaran de la cámara y muestra
//       cada seña detectada.
//
//   node probar_flujo.mjs test <flujo_test.npy> <flujo_test.json> [fps]
//       Reproduce cada video del conjunto de prueba como si llegara de la cámara y
//       mide cuántas señas se detectan solas y cuántas se reconocen bien.
//
// Necesita onnxruntime-node. Si no está instalado junto a este archivo, indica dónde
// con la variable de entorno MODULOS (una carpeta que contenga node_modules).

import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';

const aqui = path.dirname(fileURLToPath(import.meta.url));
const web = path.resolve(aqui, '..', '..', 'web');
const requerir = createRequire(process.env.MODULOS ? path.join(process.env.MODULOS, 'x.js') : import.meta.url);
const ort = requerir('onnxruntime-node');

const { Segmentador } = await import(pathToFileURL(path.join(web, 'js', 'segmentador.js')));
const { decidir, creibleSinManos } = await import(pathToFileURL(path.join(web, 'js', 'decision.js')));
const { prepararSecuencia } = await import(pathToFileURL(path.join(web, 'js', 'secuencia.js')));
const { clases } = JSON.parse(readFileSync(path.join(web, 'modelos', 'clases.json'), 'utf8'));
const sesion = await ort.InferenceSession.create(path.join(web, 'modelos', 'lsc_tssi.onnx'));

/** Lee una matriz float32 de dos dimensiones guardada con numpy. */
function leerNpy(ruta) {
  const b = readFileSync(ruta);
  const largo = b.readUInt16LE(8);
  const cabecera = b.subarray(10, 10 + largo).toString('latin1');
  const [, filas, columnas] = cabecera.match(/\((\d+),\s*(\d+)\)/).map(Number);
  if (!cabecera.includes("'<f4'")) throw new Error('Se esperaba float32: ' + cabecera);
  const datos = new Float32Array(b.buffer.slice(b.byteOffset + 10 + largo, b.byteOffset + 10 + largo + filas * columnas * 4));
  return { filas, columnas, datos };
}

// --cruda: entrega la seña al modelo tal como sale del segmentador, sin rellenar huecos
// de las manos ni remuestrear (para comparar).
const CRUDA = process.argv.includes('--cruda');

async function clasificar(evento) {
  const s = CRUDA ? evento : prepararSecuencia(evento.datos, evento.tiempos);
  const entrada = new ort.Tensor('float32', s.datos, [s.T, 225]);
  const { probabilidades } = await sesion.run({ secuencia: entrada });
  const decision = decidir(probabilidades.data, clases.length);
  // Igual que la página: sin manos solo se aceptan las señas que se hacen sobre la cara.
  if (evento.pocasManos && !creibleSinManos(clases[decision.indice].id)) return null;
  return decision;
}

const [modo, ...resto] = process.argv.slice(2);

if (modo === 'video') {
  const { filas, datos } = leerNpy(resto[0]);
  const fps = Number(resto[1] ?? 30);
  const seg = new Segmentador();
  for (let i = 0; i < filas; i++) {
    const t = (i * 1000) / fps;
    const evento = seg.procesar(datos.subarray(i * 225, (i + 1) * 225), t);
    if (!evento || evento.tipo === 'inicio') continue;
    if (evento.tipo === 'descartada') {
      console.log(`${(t / 1000).toFixed(1).padStart(5)} s  (descartada: ${evento.motivo})`);
      continue;
    }
    const d = await clasificar(evento);
    if (!d) { console.log(`${(t / 1000).toFixed(1).padStart(5)} s  (descartada: sin-manos)`); continue; }
    const texto = d.mejores.map((m) => `${clases[m.indice].etiqueta} ${(m.confianza * 100).toFixed(0)}%`).join(' | ');
    console.log(`${(evento.tInicio / 1000).toFixed(1).padStart(5)}–${(evento.tFin / 1000).toFixed(1)} s  ` +
      `${d.reconocida ? 'OK ' : '?? '} ${texto}   [${evento.T} cuadros, lista ${((t - evento.tFin) / 1000).toFixed(2)} s después]`);
  }
} else if (modo === 'test') {
  const { datos } = leerNpy(resto[0]);
  const { tramos } = JSON.parse(readFileSync(resto[1], 'utf8'));
  const fps = Number(resto[2] && !resto[2].startsWith('--') ? resto[2] : 18);
  // Opciones para poner a prueba el segmentador en condiciones más difíciles:
  //   --salto=2      usa uno de cada 2 cuadros (equipo lento)
  //   --mesa=1.3     las manos nunca bajan de esa altura (apoyadas en una mesa)
  //   --previo=0     segundos de reposo antes de cada video (por omisión 1)
  //   --porclase     muestra el resultado de cada seña y con cuáles se confunde
  const opcion = (nombre, porOmision) => {
    const a = resto.find((x) => x.startsWith(`--${nombre}=`));
    return a ? Number(a.split('=')[1]) : porOmision;
  };
  const salto = opcion('salto', 1), mesa = opcion('mesa', 0), previo = opcion('previo', 1);
  const indiceDe = new Map(clases.map((c, i) => [c.id, i]));

  const enMesa = (v) => {
    if (!mesa) return v;
    const ancho = Math.hypot(v[33] - v[36], v[34] - v[37]);
    if (!(ancho > 0.03)) return v;
    const tope = (v[34] + v[37]) / 2 + mesa * ancho;
    const w = new Float32Array(v);
    for (const i of [15, 16]) if (w[i * 3 + 1] > tope || w[i * 3 + 1] >= 1) w[i * 3 + 1] = tope;
    return w;
  };

  // Cada video se reproduce como si llegara de la cámara: antes se sostiene su primer
  // cuadro (la persona está quieta) y al terminar su último cuadro 1.4 s, para que la
  // seña pueda cerrarse.
  const limites = [];
  let cuadro = 0, descartadas = 0;
  for (const tr of tramos) {
    const seg = new Segmentador();
    const l = { glosa: tr.glosa, detecciones: [] };
    const antes = Math.ceil(previo * fps), extra = Math.ceil(1.4 * fps);
    for (let i = -antes; i < tr.cuadros + extra; i++) {
      if (((i % salto) + salto) % salto !== 0) continue;
      const k = cuadro + Math.max(0, Math.min(i, tr.cuadros - 1));
      const evento = seg.procesar(enMesa(datos.subarray(k * 225, (k + 1) * 225)), ((i + antes) * 1000) / fps);
      if (!evento || evento.tipo === 'inicio') continue;
      if (evento.tipo === 'descartada') { descartadas++; continue; }
      const d = await clasificar(evento);
      if (d) l.detecciones.push(d); else descartadas++;
    }
    limites.push(l);
    cuadro += tr.cuadros;
  }

  let unaSola = 0, ninguna = 0, varias = 0, aciertos = 0, aceptadas = 0, aciertosAceptadas = 0;
  for (const l of limites) {
    if (l.detecciones.length === 0) { ninguna++; continue; }
    if (l.detecciones.length > 1) varias++; else unaSola++;
    const d = l.detecciones[0];
    const bien = d.indice === indiceDe.get(l.glosa);
    aciertos += bien;
    if (d.reconocida) { aceptadas++; aciertosAceptadas += bien; }
  }
  const n = limites.length, pct = (x, y) => `${((100 * x) / y).toFixed(1)}%`;
  console.log(`Videos de prueba: ${n} (a ${fps} cuadros por segundo)`);
  console.log(`Detectados como una sola seña: ${unaSola} (${pct(unaSola, n)}) · partidos en varias: ${varias} · no detectados: ${ninguna}`);
  console.log(`Movimientos descartados por cortos o bajos: ${descartadas}`);
  console.log(`Exactitud sobre todos los videos (no detectado = error): ${pct(aciertos, n)}`);
  console.log(`Con el umbral de confianza: se muestran ${pct(aceptadas, n)} y de esas acierta ${pct(aciertosAceptadas, aceptadas)}`);

  // --porclase: las señas con peor resultado y con qué se confunden.
  if (resto.includes('--porclase')) {
    const porClase = new Map();
    for (const l of limites) {
      const c = porClase.get(l.glosa) ?? { n: 0, sin: 0, bien: 0, errores: new Map() };
      c.n++;
      const d = l.detecciones[0];
      if (!d) c.sin++;
      else if (d.indice === indiceDe.get(l.glosa)) c.bien++;
      else c.errores.set(clases[d.indice].etiqueta, (c.errores.get(clases[d.indice].etiqueta) ?? 0) + 1);
      porClase.set(l.glosa, c);
    }
    console.log('\nPor seña (de peor a mejor): aciertos / videos · no detectados · confusiones');
    for (const [glosa, c] of [...porClase].sort((x, y) => x[1].bien / x[1].n - y[1].bien / y[1].n)) {
      const errores = [...c.errores].sort((x, y) => y[1] - x[1]).map(([e, k]) => `${e} ${k}`).join(', ');
      console.log(`${glosa.padEnd(16)} ${String(c.bien).padStart(2)} / ${String(c.n).padEnd(3)} · ${c.sin} · ${errores}`);
    }
  }
} else {
  console.log('Uso: node probar_flujo.mjs video <video.npy> [fps]  |  test <flujo_test.npy> <flujo_test.json> [fps]');
}
