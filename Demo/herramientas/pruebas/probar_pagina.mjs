// Prueba la página completa en un navegador real (Edge o Chrome ya instalados),
// usando un video en lugar de la cámara.
//
//   1. En otra terminal:  python Demo/herramientas/servidor_local.py --video Demo/video_ejemplo.mp4
//   2. node probar_pagina.mjs [opciones]
//
//        --url http://localhost:8000     dirección del servidor local
//        --capturas <carpeta>            guarda capturas de pantalla
//        --navegador msedge|chrome
//        --movil                         tamaño de celular
//        --cpu                           MediaPipe sin aceleración gráfica
//        --hilo                          MediaPipe en un hilo aparte
//        --directo                       analiza el video a tamaño completo
//        --sinzoom                       sin acercar el análisis a la persona
//        --gpu                           pide la tarjeta gráfica real aunque no haya ventana
//        --visible                       muestra la ventana del navegador
//
// Muestra las señas que la página reconoció en el video.
// Necesita playwright-core (ver MODULOS en probar_flujo.mjs).

import { createRequire } from 'node:module';
import { mkdirSync } from 'node:fs';
import path from 'node:path';

const requerir = createRequire(process.env.MODULOS ? path.join(process.env.MODULOS, 'x.js') : import.meta.url);
const { chromium } = requerir('playwright-core');

const args = process.argv.slice(2);
const tiene = (nombre) => args.includes(`--${nombre}`);
const opcion = (nombre, porDefecto) => {
  const i = args.indexOf(`--${nombre}`);
  return i >= 0 && args[i + 1] && !args[i + 1].startsWith('--') ? args[i + 1] : porDefecto;
};
const url = opcion('url', 'http://localhost:8000');
const capturas = opcion('capturas', null);
const movil = tiene('movil');
const prefijo = movil ? 'movil-' : '';
if (capturas) mkdirSync(capturas, { recursive: true });

const banderas = ['--autoplay-policy=no-user-gesture-required', '--use-fake-ui-for-media-stream', '--ignore-gpu-blocklist'];
banderas.push(...(tiene('gpu') ? ['--enable-gpu', '--use-angle=d3d11'] : ['--enable-unsafe-swiftshader']));

const navegador = await chromium.launch({ channel: opcion('navegador', 'msedge'), headless: !tiene('visible'), args: banderas });
const contexto = await navegador.newContext(movil
  ? { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true }
  : { viewport: { width: Number(opcion('ancho', 1440)), height: Number(opcion('alto', 900)) }, deviceScaleFactor: 1 });
const pagina = await contexto.newPage();
pagina.on('console', (m) => { if (m.type() === 'error') console.log('  [consola]', m.text().slice(0, 300)); });
pagina.on('pageerror', (e) => console.log('  [error]', String(e).slice(0, 300)));

const foto = async (nombre, completa = false) => {
  if (capturas) await pagina.screenshot({ path: path.join(capturas, `${prefijo}${nombre}.png`), fullPage: completa });
};
const listo = () => pagina.waitForFunction(() => window.__demo && window.__demo.listo, null, { timeout: 180000 });

// 1. Pantalla de bienvenida y diálogo
await pagina.goto(url, { waitUntil: 'load' });
await listo();
await pagina.waitForTimeout(600);
await foto('1-bienvenida', movil);
await pagina.click('#btn-como');
await pagina.waitForTimeout(600);
await foto('1-dialogo');

// 2. Con el video de prueba
await pagina.goto(`${url}/?video=/_prueba/video&debug${tiene('cpu') ? '&cpu' : ''}${tiene('hilo') ? '&hilo' : ''}${tiene('directo') ? '&directo' : ''}${tiene('sinzoom') ? '&sinzoom' : ''}`, { waitUntil: 'load' });
await listo();
const inicio = Date.now();
let vistos = 0, fotoCaptura = false;
for (;;) {
  const estado = await pagina.evaluate(() => ({
    n: window.__demo.eventos.length, fin: window.__demo.terminado,
    estado: document.getElementById('escenario').dataset.estado,
  }));
  if (estado.estado === 'capturando' && !fotoCaptura) { await foto('2-capturando'); fotoCaptura = true; }
  if (estado.n > vistos) {
    vistos = estado.n;
    await pagina.waitForTimeout(350);
    await foto(`3-resultado-${vistos}`);
  }
  if (estado.fin || Date.now() - inicio > 240000) break;
  await pagina.waitForTimeout(60);
}
await pagina.waitForTimeout(400);
await foto('4-final', movil);

const datos = await pagina.evaluate(() => ({
  eventos: window.__demo.eventos, info: window.__demo.info, registro: window.__demo.registro, traza: window.__demo.traza,
  rendimiento: document.getElementById('depuracion').textContent,
  aislada: window.crossOriginIsolated,
  desplazamiento: Math.round(window.scrollY),
  grafica: (() => {
    try {
      const gl = document.createElement('canvas').getContext('webgl');
      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : 'desconocida';
    } catch { return 'sin WebGL'; }
  })(),
}));
console.log(`Entorno: MediaPipe ${datos.info.delegado} (${datos.info.modo}) · clasificador con ${datos.info.hilos} hilo(s) · página aislada: ${datos.aislada}`);
console.log(`Tarjeta gráfica: ${datos.grafica}`);
console.log(datos.rendimiento.split('\n')[0]);
console.log(`Desplazamiento de la página al terminar: ${datos.desplazamiento} px (debe ser 0)`);
console.log(`Señas detectadas: ${datos.eventos.length}`);
for (const e of datos.eventos) {
  console.log(`  ${(e.inicio / 1000).toFixed(1).padStart(5)}–${(e.fin / 1000).toFixed(1)} s  ${e.reconocida ? 'OK' : '??'}  ` +
    e.mejores.map(([n, c]) => `${n} ${(c * 100).toFixed(0)}%`).join(' | ') + `   [${e.cuadros} cuadros, ${Math.round(e.ms)} ms]`);
}
const descartes = datos.registro.filter((r) => r.includes('descartada'));
if (descartes.length) console.log('Descartadas:', descartes.join(', '));
// --traza <archivo>: guarda lo que midió el segmentador en cada cuadro
// [ms, estado, altura izq, altura der, reposo izq, reposo der, borde, mano izq, mano der, área analizada]
if (opcion('traza', '')) (await import('node:fs')).writeFileSync(opcion('traza', ''), JSON.stringify(datos.traza));
await navegador.close();
