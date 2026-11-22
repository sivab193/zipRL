import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { Eye, EyeOff, RotateCcw } from 'lucide-react';
import { NetworkVisualSnapshot } from '../rl/types';

interface Network3DViewProps {
  network: NetworkVisualSnapshot;
  heightClass?: string;
}

type CameraPreset = 'isometric' | 'side' | 'output' | 'top';

const CAMERA_PRESETS: Record<CameraPreset, { pos: [number, number, number]; target: [number, number, number] }> = {
  isometric: { pos: [8.8, 5.2, 11.4], target: [0, 1.8, 0] },
  side: { pos: [0, 2.2, 13.8], target: [0, 1.8, 0] },
  output: { pos: [9.6, 2.4, 4.2], target: [4.2, 1.8, 0] },
  top: { pos: [0.2, 14.2, 3.8], target: [0, 1.2, 0] },
};

function createFloorGridTexture(maxAnisotropy: number): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d')!;

  ctx.fillStyle = '#0D1422';
  ctx.fillRect(0, 0, 512, 512);

  // Fine technical coordinate lines
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.12)';
  ctx.lineWidth = 1;
  const step = 32;
  for (let i = 0; i <= 512; i += step) {
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i, 512);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i);
    ctx.lineTo(512, i);
    ctx.stroke();
  }

  // Major crosshairs
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.22)';
  ctx.lineWidth = 2;
  for (let i = 0; i <= 512; i += 128) {
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i, 512);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i);
    ctx.lineTo(512, i);
    ctx.stroke();
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(4, 4);
  tex.anisotropy = maxAnisotropy;
  return tex;
}

function computeLayerPositions(): {
  input: THREE.Vector3[];
  h1: THREE.Vector3[];
  h2: THREE.Vector3[];
  output: THREE.Vector3[];
} {
  const yCenter = 2.25;

  // Input Layer: 5x5 grid at X = -5.4
  const input: THREE.Vector3[] = [];
  for (let r = 0; r < 5; r++) {
    for (let c = 0; c < 5; c++) {
      input.push(new THREE.Vector3(-5.4, yCenter + (2 - r) * 0.68, (c - 2) * 0.68));
    }
  }

  // Hidden Layer 1: 6x4 matrix (representing 128 units) at X = -1.7
  const h1: THREE.Vector3[] = [];
  for (let r = 0; r < 6; r++) {
    for (let c = 0; c < 4; c++) {
      h1.push(new THREE.Vector3(-1.7, yCenter + (2.5 - r) * 0.64, (c - 1.5) * 0.85));
    }
  }

  // Hidden Layer 2: 4x4 matrix (representing 64 units) at X = 1.9
  const h2: THREE.Vector3[] = [];
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      h2.push(new THREE.Vector3(1.9, yCenter + (1.5 - r) * 0.78, (c - 1.5) * 0.85));
    }
  }

  // Output Q-Head: 4 directional compass nodes (Up, Right, Down, Left) at X = 5.4
  const output: THREE.Vector3[] = [
    new THREE.Vector3(5.4, yCenter + 1.15, 0),    // 0: Up
    new THREE.Vector3(5.4, yCenter, 1.15),        // 1: Right
    new THREE.Vector3(5.4, yCenter - 1.15, 0),    // 2: Down
    new THREE.Vector3(5.4, yCenter, -1.15),       // 3: Left
  ];

  return { input, h1, h2, output };
}

export const Network3DView: React.FC<Network3DViewProps> = ({
  network,
  heightClass = 'h-[420px]',
}) => {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const networkRef = useRef<NetworkVisualSnapshot>(network);
  const requestRenderRef = useRef<(() => void) | null>(null);
  const applyPresetRef = useRef<((preset: CameraPreset) => void) | null>(null);

  const [activePreset, setActivePreset] = useState<CameraPreset>('isometric');
  const [hudExpanded, setHudExpanded] = useState(true);

  useEffect(() => {
    networkRef.current = network;
    requestRenderRef.current?.();
  }, [network]);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    // Detect software WebGL from GPU renderer string (never user agent)
    const isSoftwareGl = (() => {
      const probeCanvas = document.createElement('canvas');
      const gl = probeCanvas.getContext('webgl2');
      if (!gl) return true;
      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      const param = ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER;
      const gpu = String(gl.getParameter(param));
      gl.getExtension('WEBGL_lose_context')?.loseContext();
      return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
    })();

    const width = Math.max(300, container.clientWidth);
    const height = Math.max(260, container.clientHeight);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b101a);
    scene.fog = new THREE.FogExp2(0x0b101a, 0.024);

    const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);
    camera.position.set(...CAMERA_PRESETS.isometric.pos);

    const renderer = new THREE.WebGLRenderer({
      antialias: !isSoftwareGl,
      powerPreference: 'high-performance',
    });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.12;
    renderer.setPixelRatio(isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = !isSoftwareGl;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    // Procedural Studio Environment Map (RoomEnvironment)
    const pmrem = new THREE.PMREMGenerator(renderer);
    const envTarget = pmrem.fromScene(new RoomEnvironment(), 0.04);
    scene.environment = envTarget.texture;
    pmrem.dispose();

    // Warm/cool studio lighting rig
    const hemiLight = new THREE.HemisphereLight(0xb8d8ff, 0x162033, 0.95);
    scene.add(hemiLight);

    const keyLight = new THREE.DirectionalLight(0xfff1dc, 1.8);
    keyLight.position.set(8, 12, 9);
    if (!isSoftwareGl) {
      keyLight.castShadow = true;
      keyLight.shadow.mapSize.set(1024, 1024);
      keyLight.shadow.camera.left = -9;
      keyLight.shadow.camera.right = 9;
      keyLight.shadow.camera.top = 7;
      keyLight.shadow.camera.bottom = -7;
      keyLight.shadow.bias = -0.0005;
      keyLight.shadow.normalBias = 0.02;
    }
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x38bdf8, 0.75);
    rimLight.position.set(-9, 6, -8);
    scene.add(rimLight);

    // Grounded architectural pedestal platform
    const floorTex = createFloorGridTexture(renderer.capabilities.getMaxAnisotropy());
    const pedestalGeo = new THREE.BoxGeometry(16.5, 0.24, 9.2);
    const pedestalMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      map: floorTex,
      roughness: 0.62,
      metalness: 0.22,
    });
    const pedestal = new THREE.Mesh(pedestalGeo, pedestalMat);
    pedestal.position.set(0, -0.12, 0);
    pedestal.receiveShadow = !isSoftwareGl;
    scene.add(pedestal);

    // Subtle metallic trim bezel around pedestal
    const bezelGeo = new THREE.BoxGeometry(16.7, 0.08, 9.4);
    const bezelMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      roughness: 0.3,
      metalness: 0.8,
    });
    const bezel = new THREE.Mesh(bezelGeo, bezelMat);
    bezel.position.set(0, -0.22, 0);
    scene.add(bezel);

    // Layer positions & architectural mounting rails (grounded flush to y=0)
    const layers = computeLayerPositions();
    const layerXs = [-5.4, -1.7, 1.9, 5.4];
    const railGeo = new THREE.BoxGeometry(0.18, 0.14, 4.4);
    const railMat = new THREE.MeshStandardMaterial({
      color: 0x334155,
      roughness: 0.35,
      metalness: 0.75,
    });
    const pillarGeo = new THREE.CylinderGeometry(0.05, 0.06, 0.55, 12);
    const createdDisposables: Array<{ dispose: () => void }> = [
      floorTex,
      pedestalGeo,
      pedestalMat,
      bezelGeo,
      bezelMat,
      railGeo,
      railMat,
      pillarGeo,
      envTarget,
    ];

    for (const lx of layerXs) {
      const rail = new THREE.Mesh(railGeo, railMat);
      rail.position.set(lx, 0.07, 0);
      scene.add(rail);
      for (const pz of [-1.75, 1.75]) {
        const pillar = new THREE.Mesh(pillarGeo, railMat);
        pillar.position.set(lx, 0.28, pz);
        scene.add(pillar);
      }
    }

    // Composite Neuron InstancedMeshes:
    // Part 1: Glossy core sphere
    // Part 2: Brushed metallic equatorial collar ring
    const allNodes: THREE.Vector3[] = [
      ...layers.input,
      ...layers.h1,
      ...layers.h2,
      ...layers.output,
    ];
    const nodeCount = allNodes.length;

    const coreGeo = new THREE.SphereGeometry(0.15, 16, 12);
    const coreMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 0.22,
      metalness: 0.18,
    });
    const coreMesh = new THREE.InstancedMesh(coreGeo, coreMat, nodeCount);
    coreMesh.frustumCulled = false;

    const collarGeo = new THREE.TorusGeometry(0.185, 0.024, 8, 18);
    const collarMat = new THREE.MeshStandardMaterial({
      color: 0x94a3b8,
      roughness: 0.28,
      metalness: 0.85,
    });
    const collarMesh = new THREE.InstancedMesh(collarGeo, collarMat, nodeCount);
    collarMesh.frustumCulled = false;

    const dummy = new THREE.Object3D();
    const baseColor = new THREE.Color(0x3b82f6);
    for (let i = 0; i < nodeCount; i++) {
      const pos = allNodes[i];
      const isOutput = i >= nodeCount - 4;
      const scale = isOutput ? 1.45 : 1.0;

      dummy.position.copy(pos);
      dummy.rotation.set(0, Math.PI / 2, 0);
      dummy.scale.setScalar(scale);
      dummy.updateMatrix();
      coreMesh.setMatrixAt(i, dummy.matrix);
      coreMesh.setColorAt(i, baseColor);

      collarMesh.setMatrixAt(i, dummy.matrix);
    }
    coreMesh.instanceMatrix.needsUpdate = true;
    collarMesh.instanceMatrix.needsUpdate = true;
    if (coreMesh.instanceColor) coreMesh.instanceColor.needsUpdate = true;
    coreMesh.computeBoundingSphere();
    collarMesh.computeBoundingSphere();
    scene.add(coreMesh);
    scene.add(collarMesh);

    createdDisposables.push(coreGeo, coreMat, collarGeo, collarMat);

    // Synaptic Connections (LineSegments with per-vertex RGB driven by real weight-update magnitudes)
    // Build a clean deterministic topology covering all layers without opaque visual clutter:
    // - Input (25) -> H1 (24): connect each H1 node to 6 strided input nodes = 144 lines
    // - H1 (24) -> H2 (16): connect each H2 node to 8 strided H1 nodes = 128 lines
    // - H2 (16) -> Output (4): full 16 * 4 = 64 lines
    interface SynapseEdge {
      from: THREE.Vector3;
      to: THREE.Vector3;
      layerIdx: 0 | 1 | 2;
      deltaIdx: number;
    }

    const edges: SynapseEdge[] = [];
    for (let o = 0; o < layers.h1.length; o++) {
      for (let k = 0; k < 6; k++) {
        const i = (o * 5 + k * 4) % layers.input.length;
        edges.push({
          from: layers.input[i],
          to: layers.h1[o],
          layerIdx: 0,
          deltaIdx: o * layers.input.length + i,
        });
      }
    }
    for (let o = 0; o < layers.h2.length; o++) {
      for (let k = 0; k < 8; k++) {
        const i = (o * 3 + k * 3) % layers.h1.length;
        edges.push({
          from: layers.h1[i],
          to: layers.h2[o],
          layerIdx: 1,
          deltaIdx: o * layers.h1.length + i,
        });
      }
    }
    for (let o = 0; o < layers.output.length; o++) {
      for (let i = 0; i < layers.h2.length; i++) {
        edges.push({
          from: layers.h2[i],
          to: layers.output[o],
          layerIdx: 2,
          deltaIdx: o * layers.h2.length + i,
        });
      }
    }

    const linePositions = new Float32Array(edges.length * 6);
    const lineColors = new Float32Array(edges.length * 6);
    for (let e = 0; e < edges.length; e++) {
      const edge = edges[e];
      linePositions[e * 6] = edge.from.x;
      linePositions[e * 6 + 1] = edge.from.y;
      linePositions[e * 6 + 2] = edge.from.z;
      linePositions[e * 6 + 3] = edge.to.x;
      linePositions[e * 6 + 4] = edge.to.y;
      linePositions[e * 6 + 5] = edge.to.z;
    }

    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
    const colorAttr = new THREE.BufferAttribute(lineColors, 3);
    lineGeo.setAttribute('color', colorAttr);

    const lineMat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.78,
    });
    const synapseLines = new THREE.LineSegments(lineGeo, lineMat);
    synapseLines.frustumCulled = false;
    scene.add(synapseLines);
    createdDisposables.push(lineGeo, lineMat);

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(...CAMERA_PRESETS.isometric.target);
    controls.enableDamping = !isSoftwareGl;
    controls.dampingFactor = 0.08;
    controls.minDistance = 4.5;
    controls.maxDistance = 24;
    controls.maxPolarAngle = Math.PI * 0.48;
    controls.update();

    // Scratch colors (zero GC in render)
    const _cLow = new THREE.Color(0x1e293b);
    const _cMid = new THREE.Color(0x38bdf8);
    const _cHigh = new THREE.Color(0x60a5fa);
    const _cBurst = new THREE.Color(0xf59e0b);
    const _cBurstPeak = new THREE.Color(0xfef08a);
    const _cOutLegal = new THREE.Color(0x10b981);
    const _cOutMasked = new THREE.Color(0x334155);
    const _cWork = new THREE.Color();

    // Camera transition state
    let transitionActive = false;
    const camGoalPos = new THREE.Vector3(...CAMERA_PRESETS.isometric.pos);
    const camGoalTarget = new THREE.Vector3(...CAMERA_PRESETS.isometric.target);
    let lastTime = performance.now();

    const syncVisualsFromNetwork = () => {
      const snap = networkRef.current;
      const burst = snap.solveBurstActive;

      // 1. Update neuron core colors from live activations & Q-values
      let nodeIdx = 0;
      for (let i = 0; i < layers.input.length; i++) {
        const act = snap.inputActivations[i] ?? 0;
        _cWork.copy(_cLow).lerp(_cMid, 0.2 + act * 0.8);
        coreMesh.setColorAt(nodeIdx++, _cWork);
      }
      for (let i = 0; i < layers.h1.length; i++) {
        const act = snap.hidden1Activations[i] ?? 0;
        _cWork.copy(_cLow).lerp(burst ? _cBurst : _cHigh, 0.18 + act * 0.82);
        coreMesh.setColorAt(nodeIdx++, _cWork);
      }
      for (let i = 0; i < layers.h2.length; i++) {
        const act = snap.hidden2Activations[i] ?? 0;
        _cWork.copy(_cLow).lerp(burst ? _cBurstPeak : _cMid, 0.2 + act * 0.8);
        coreMesh.setColorAt(nodeIdx++, _cWork);
      }

      // Output Q-head: highlight highest legal Q in bright emerald/gold
      let bestLegal = -1;
      let bestQ = -Infinity;
      for (let a = 0; a < 4; a++) {
        if (snap.legalMask[a] && (snap.outputQValues[a] ?? -Infinity) > bestQ) {
          bestQ = snap.outputQValues[a];
          bestLegal = a;
        }
      }
      for (let a = 0; a < 4; a++) {
        if (!snap.legalMask[a]) {
          _cWork.copy(_cOutMasked);
        } else if (a === bestLegal) {
          _cWork.copy(burst ? _cBurstPeak : _cOutLegal);
        } else {
          _cWork.copy(_cMid);
        }
        coreMesh.setColorAt(nodeIdx++, _cWork);
      }
      if (coreMesh.instanceColor) coreMesh.instanceColor.needsUpdate = true;

      // 2. Update synapse colors from REAL per-step weight deltas (|Δw|)
      const colorArr = colorAttr.array as Float32Array;
      for (let e = 0; e < edges.length; e++) {
        const edge = edges[e];
        const strength =
          edge.layerIdx === 0
            ? snap.w1DeltaStrengths[edge.deltaIdx] ?? 0
            : edge.layerIdx === 1
              ? snap.w2DeltaStrengths[edge.deltaIdx] ?? 0
              : snap.w3DeltaStrengths[edge.deltaIdx] ?? 0;

        if (burst) {
          _cWork.copy(_cLow).lerp(_cBurst, Math.min(1, 0.28 + strength * 0.9));
          if (strength > 0.65) {
            _cWork.lerp(_cBurstPeak, (strength - 0.65) * 1.6);
          }
        } else {
          _cWork.copy(_cLow).lerp(_cMid, Math.min(1, 0.14 + strength * 0.86));
        }

        const base = e * 6;
        colorArr[base] = _cWork.r;
        colorArr[base + 1] = _cWork.g;
        colorArr[base + 2] = _cWork.b;
        colorArr[base + 3] = _cWork.r;
        colorArr[base + 4] = _cWork.g;
        colorArr[base + 5] = _cWork.b;
      }
      colorAttr.needsUpdate = true;
    };

    // Coalesced on-demand renderer with software-WebGL afterDraw fence sync
    let frameId = 0;
    let disposed = false;
    let renderBusy = false;
    let renderDirty = false;

    const afterDraw = (cb: () => void) => {
      const gl = renderer.getContext() as WebGL2RenderingContext;
      const sync = gl.fenceSync?.(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
      gl.flush();
      const poll = () => {
        if (disposed) return;
        const state = sync ? gl.getSyncParameter(sync, gl.SYNC_STATUS) : gl.SIGNALED;
        if (state === gl.UNSIGNALED) {
          frameId = window.setTimeout(poll, 5);
        } else {
          if (sync) gl.deleteSync(sync);
          frameId = window.setTimeout(cb, 80);
        }
      };
      frameId = window.setTimeout(poll, 5);
    };

    const requestRender = () => {
      if (disposed) return;
      if (renderBusy) {
        renderDirty = true;
        return;
      }
      renderBusy = true;
      frameId = requestAnimationFrame(() => {
        if (disposed) return;
        renderDirty = false;

        const now = performance.now();
        const dt = Math.min(0.1, (now - lastTime) / 1000);
        lastTime = now;

        if (transitionActive) {
          if (isSoftwareGl) {
            camera.position.copy(camGoalPos);
            controls.target.copy(camGoalTarget);
            transitionActive = false;
          } else {
            const alpha = 1 - Math.exp(-7 * dt);
            camera.position.lerp(camGoalPos, alpha);
            controls.target.lerp(camGoalTarget, alpha);
            if (
              camera.position.distanceTo(camGoalPos) < 0.03 &&
              controls.target.distanceTo(camGoalTarget) < 0.03
            ) {
              camera.position.copy(camGoalPos);
              controls.target.copy(camGoalTarget);
              transitionActive = false;
            } else {
              renderDirty = true;
            }
          }
        }

        syncVisualsFromNetwork();
        controls.update();
        renderer.render(scene, camera);

        const done = () => {
          renderBusy = false;
          if (!disposed && renderDirty) {
            requestRender();
          }
        };

        if (isSoftwareGl) {
          afterDraw(done);
        } else {
          done();
        }
      });
    };

    requestRenderRef.current = requestRender;
    applyPresetRef.current = (preset: CameraPreset) => {
      const cfg = CAMERA_PRESETS[preset];
      camGoalPos.set(...cfg.pos);
      camGoalTarget.set(...cfg.target);
      transitionActive = true;
      requestRender();
    };

    controls.addEventListener('change', requestRender);
    controls.addEventListener('start', () => {
      transitionActive = false;
    });

    const resizeObserver = new ResizeObserver(() => {
      if (disposed || !container) return;
      const w = Math.max(240, container.clientWidth);
      const h = Math.max(220, container.clientHeight);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      requestRender();
    });
    resizeObserver.observe(container);

    if (!isSoftwareGl) {
      renderer.compile(scene, camera);
    }
    requestRender();

    return () => {
      disposed = true;
      cancelAnimationFrame(frameId);
      clearTimeout(frameId);
      resizeObserver.disconnect();
      controls.dispose();
      for (const item of createdDisposables) {
        item.dispose();
      }
      renderer.dispose();
      requestRenderRef.current = null;
      applyPresetRef.current = null;
    };
  }, []);

  const handleSelectPreset = (preset: CameraPreset) => {
    setActivePreset(preset);
    applyPresetRef.current?.(preset);
  };

  return (
    <div
      className="relative w-full rounded-lg overflow-hidden border border-slate-800 bg-[#0B101A]"
      data-testid="network-3d-panel"
    >
      {/* Minimal Top-Left Architecture Spec Overlay (solid dark fill, no backdrop-blur) */}
      <div className="absolute top-3 left-3 z-10 px-3 py-2 rounded-md bg-[#0F172A]/95 border border-slate-700/80 text-slate-200 max-w-sm pointer-events-none">
        <div className="text-xs font-semibold text-white">
          Double DQN Policy Network ({network.inputSize} → {network.hidden1Size} → {network.hidden2Size} → 4)
        </div>
        <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] font-mono tabular-nums text-slate-300">
          <span>Grad steps: {network.totalGradientSteps.toLocaleString()}</span>
          <span aria-hidden="true">·</span>
          <span>Mean |Δw|: {network.meanWeightDelta.toExponential(2)}</span>
          <span aria-hidden="true">·</span>
          <span>Loss: {network.lastLoss.toFixed(4)}</span>
        </div>
        <div className="mt-1 text-[11px] text-slate-400">
          {network.solveBurstActive
            ? 'Solve reward burst propagating through replay batch (+10 terminal TD error)'
            : 'Weights update live on every replay-buffer step (drag to orbit, scroll to zoom)'}
        </div>
        <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[10.5px] font-mono text-slate-300">
          <span className="inline-flex items-center gap-1">
            <span className="w-2 h-2 rounded-xs bg-sky-400 inline-block" /> |Δw| step
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="w-2 h-2 rounded-xs bg-amber-400 inline-block" /> +10 solve burst
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="w-2 h-2 rounded-xs bg-emerald-400 inline-block" /> Best Q-head
          </span>
        </div>
      </div>

      {/* Dedicated Isolated WebGL Mount Container */}
      <div
        ref={mountRef}
        className={`w-full ${heightClass}`}
        data-testid="network-3d-canvas-mount"
      />

      {/* Unified Collapsible Bottom Camera Dock (Zero Side-Panel Clutter, No backdrop-blur) */}
      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-10 flex flex-col items-center">
        <button
          type="button"
          onClick={() => setHudExpanded((v) => !v)}
          className="mb-1 px-2.5 py-1 rounded text-[11px] font-medium text-slate-300 bg-[#0F172A]/95 border border-slate-700/80 hover:text-white transition-colors inline-flex items-center gap-1.5 whitespace-nowrap"
          aria-label={hudExpanded ? 'Hide 3D camera controls' : 'Show 3D camera controls'}
        >
          {hudExpanded ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
          {hudExpanded ? 'Hide Camera Dock' : 'Camera Controls'}
        </button>

        {hudExpanded && (
          <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md bg-[#0F172A]/95 border border-slate-700/90 text-xs">
            {(
              [
                ['isometric', 'Isometric'],
                ['side', 'Layer Profile'],
                ['output', 'Q-Head'],
                ['top', 'Top Down'],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                onClick={() => handleSelectPreset(id)}
                className={`px-2.5 py-1 rounded text-xs font-medium whitespace-nowrap transition-colors ${
                  activePreset === id
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800'
                }`}
              >
                {label}
              </button>
            ))}
            <span className="w-px h-4 bg-slate-700 mx-0.5" aria-hidden="true" />
            <button
              type="button"
              onClick={() => handleSelectPreset('isometric')}
              className="p-1.5 rounded text-slate-300 hover:text-white hover:bg-slate-800 transition-colors"
              aria-label="Reset 3D camera view"
              title="Reset view"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
