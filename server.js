/**
 * EchoSight — Node.js Socket Bridge
 * ===================================
 * Orchestrates the full inference pipeline:
 *   1. Accepts trigger via HTTP POST /scan or WebSocket event
 *   2. Spawns Python preprocessor as child process
 *   3. Pipes preprocessor output (spectrogram JSON) → model inference
 *   4. Broadcasts structured result over Socket.io to all connected clients
 *
 * Architecture:
 *   Client ←─── Socket.io ─── Express ─── spawn() ─── Python (preprocess.py)
 *                                    └─── spawn() ─── Python (unet_model.py)
 *
 * PM2 ecosystem (add to ecosystem.config.js):
 *   { name: 'echosight-server', script: 'server.js', interpreter: 'node' }
 *
 * Termux start:
 *   node server.js
 *   OR: pm2 start ecosystem.config.js
 */

"use strict";

const express    = require("express");
const http       = require("http");
const { Server } = require("socket.io");
const { spawn }  = require("child_process");
const path       = require("path");
const fs         = require("fs");

// ── Config ────────────────────────────────────────────────────────────────────

const CONFIG = {
  port:         process.env.PORT || 3000,
  python:        "python3",               // or 'python' on some Termux installs

  // Paths relative to this file's directory
  preprocessScript: path.join(__dirname, "preprocess.py"),
  modelScript:      path.join(__dirname, "unet_model.py"),
  onnxModel:        path.join(__dirname, "echosight_unet.onnx"),
  staticDir:        path.join(__dirname, "public"),

  // Mock data — used when no real sonar file is provided
  useMockData:   true,
  mockSonarFile: path.join(__dirname, "data", "mock_sonar.wav"),

  // Inference mode: 'pytorch' | 'onnx' | 'passthrough' (echo spec without ML)
  inferenceMode: "onnx",

  // Scan interval for continuous "live" mode (ms). 0 = manual only.
  autoScanInterval: 0,

  // Per-scan timeout (ms) — kill zombie Python procs after this
  processTimeout:  15000,
};

// ── App Setup ─────────────────────────────────────────────────────────────────

const app    = express();
const server = http.createServer(app);
const io     = new Server(server, {
  cors: { origin: "*" },          // local-only, CORS doesn't matter
  maxHttpBufferSize: 5e6,          // 5MB max message — fits 256×128 float array
  pingTimeout:  30000,
  pingInterval: 10000,
});

app.use(express.json());
app.use(express.static(CONFIG.staticDir));

// ── State ─────────────────────────────────────────────────────────────────────

const state = {
  scanning:       false,
  scanCount:      0,
  lastScanTime:   null,
  connectedClients: 0,
  errors:         [],
  autoInterval:   null,
};

// ── Utility: Spawn Python with timeout ───────────────────────────────────────

/**
 * Spawns a Python process, accumulates stdout, returns parsed JSON.
 * stderr is forwarded to Node.js console (for debug visibility in Termux).
 *
 * @param {string[]} args   - argv passed to Python script
 * @param {string|null} stdinData - optional string piped to process stdin
 * @returns {Promise<object>}
 */
function spawnPython(args, stdinData = null) {
  return new Promise((resolve, reject) => {
    const proc = spawn(CONFIG.python, args, {
      stdio: ["pipe", "pipe", "pipe"],
      env:   { ...process.env, PYTHONUNBUFFERED: "1" },
    });

    let stdoutBuf = "";
    let stderrBuf = "";
    let settled   = false;

    // Kill process if it hangs
    const killer = setTimeout(() => {
      if (!settled) {
        proc.kill("SIGTERM");
        reject(new Error(`Python process timed out after ${CONFIG.processTimeout}ms`));
        settled = true;
      }
    }, CONFIG.processTimeout);

    proc.stdout.on("data", (chunk) => { stdoutBuf += chunk.toString(); });
    proc.stderr.on("data", (chunk) => {
      stderrBuf += chunk.toString();
      // Forward Python stderr lines to Node console (debug visibility)
      chunk.toString().split("\n").filter(Boolean).forEach(l => {
        console.log(`  [Python] ${l}`);
      });
    });

    // Pipe stdin if provided
    if (stdinData !== null) {
      proc.stdin.write(stdinData);
      proc.stdin.end();
    } else {
      proc.stdin.end();
    }

    proc.on("close", (code) => {
      clearTimeout(killer);
      if (settled) return;
      settled = true;

      if (code !== 0) {
        return reject(new Error(`Python exited ${code}: ${stderrBuf.slice(-500)}`));
      }

      const trimmed = stdoutBuf.trim();
      if (!trimmed) {
        return reject(new Error("Python produced no output on stdout"));
      }

      try {
        resolve(JSON.parse(trimmed));
      } catch (e) {
        reject(new Error(`JSON parse failed. Raw output: ${trimmed.slice(0, 200)}`));
      }
    });

    proc.on("error", (err) => {
      clearTimeout(killer);
      if (!settled) {
        settled = true;
        reject(new Error(`Failed to spawn Python: ${err.message}`));
      }
    });
  });
}

// ── Pipeline: Preprocess ──────────────────────────────────────────────────────

async function runPreprocess(sonarFilePath = null) {
  const args = [CONFIG.preprocessScript];

  if (sonarFilePath && fs.existsSync(sonarFilePath)) {
    args.push("--input", sonarFilePath);
    // Auto-detect mode from extension
    const ext = path.extname(sonarFilePath).toLowerCase();
    args.push("--mode", ext === ".csv" ? "csv" : "stft");
  } else {
    // Always fall back to mock synthetic data
    args.push("--mock");
  }

  return spawnPython(args);
}

// ── Pipeline: Model Inference ─────────────────────────────────────────────────

async function runInference(preprocessResult) {
  if (CONFIG.inferenceMode === "passthrough") {
    // Return spectrogram as-is (no ML) — useful for UI dev/testing
    return {
      structural_map: preprocessResult.spectrogram,
      mode:           "passthrough",
    };
  }

  if (CONFIG.inferenceMode === "onnx") {
    // ONNX inference via dedicated Python wrapper
    const args     = [CONFIG.modelScript, "--infer"];
    const stdinData = JSON.stringify({ spectrogram: preprocessResult.spectrogram });
    return spawnPython(args, stdinData);
  }

  if (CONFIG.inferenceMode === "pytorch") {
    // Full PyTorch inference (slower, higher RAM)
    const args     = [CONFIG.modelScript, "--infer"];
    const stdinData = JSON.stringify({ spectrogram: preprocessResult.spectrogram });
    return spawnPython(args, stdinData);
  }

  throw new Error(`Unknown inference mode: ${CONFIG.inferenceMode}`);
}

// ── Core Scan Function ────────────────────────────────────────────────────────

/**
 * Runs full pipeline and broadcasts result to all connected Socket.io clients.
 * Called by HTTP POST /scan, Socket.io 'scan:start', or auto-interval.
 *
 * @param {string|null} sonarFile - optional path to sonar file override
 * @param {string|null} requesterId - socket ID of requester (for targeted ACK)
 */
async function executeScan(sonarFile = null, requesterId = null) {
  if (state.scanning) {
    const msg = { type: "warn", message: "Scan already in progress" };
    if (requesterId) io.to(requesterId).emit("scan:warn", msg);
    return;
  }

  state.scanning     = true;
  state.scanCount   += 1;
  state.lastScanTime = new Date().toISOString();
  const scanId       = `scan_${state.scanCount}_${Date.now()}`;

  // Broadcast scan start
  io.emit("scan:start", { scanId, timestamp: state.lastScanTime });

  try {
    // Step 1: Preprocess
    const t0 = Date.now();
    io.emit("scan:status", { scanId, stage: "preprocessing", progress: 20 });

    const preprocessResult = await runPreprocess(sonarFile);
    const preprocessMs     = Date.now() - t0;

    // Step 2: Inference
    io.emit("scan:status", { scanId, stage: "inference", progress: 60 });
    const t1 = Date.now();

    const inferenceResult  = await runInference(preprocessResult);
    const inferenceMs      = Date.now() - t1;
    const totalMs          = Date.now() - t0;

    // Step 3: Compose final payload
    const payload = {
      scanId,
      timestamp:      state.lastScanTime,
      spectrogram:    preprocessResult.spectrogram,    // raw acoustic heatmap
      structural_map: inferenceResult.structural_map,  // ML output
      targets:        inferenceResult.targets || [],
      meta: {
        ...preprocessResult.meta,
        preprocessMs,
        inferenceMs,
        totalMs,
        inferenceMode: CONFIG.inferenceMode,
        scanNumber:    state.scanCount,
      },
    };

    // Broadcast to all clients
    io.emit("scan:complete", payload);
    io.emit("scan:status", { scanId, stage: "complete", progress: 100 });

    console.log(
      `[EchoSight] Scan #${state.scanCount} complete | ` +
      `Pre: ${preprocessMs}ms | Infer: ${inferenceMs}ms | ` +
      `Total: ${totalMs}ms | Clients: ${state.connectedClients}`
    );

  } catch (err) {
    const errPayload = {
      scanId,
      message: err.message,
      timestamp: new Date().toISOString(),
    };

    state.errors.push(errPayload);
    io.emit("scan:error", errPayload);
    console.error(`[EchoSight] Scan #${state.scanCount} ERROR:`, err.message);

  } finally {
    state.scanning = false;
  }
}

// ── HTTP Routes ───────────────────────────────────────────────────────────────

// Trigger scan via HTTP
app.post("/api/scan", async (req, res) => {
  const { sonarFile } = req.body || {};
  res.json({ status: "scan_initiated", scanCount: state.scanCount + 1 });
  executeScan(sonarFile || null);  // fire and forget; results come via socket
});

// Server health + stats
app.get("/api/status", (req, res) => {
  res.json({
    status:          "online",
    scanning:        state.scanning,
    scanCount:       state.scanCount,
    lastScanTime:    state.lastScanTime,
    connectedClients: state.connectedClients,
    inferenceMode:   CONFIG.inferenceMode,
    errors:          state.errors.slice(-5),  // last 5 errors
  });
});

// Config endpoint (read-only — no secrets)
app.get("/api/config", (req, res) => {
  res.json({
    inferenceMode:   CONFIG.inferenceMode,
    autoScanInterval: CONFIG.autoScanInterval,
    useMockData:     CONFIG.useMockData,
    onnxModelExists: fs.existsSync(CONFIG.onnxModel),
  });
});

// 404 fallback → serve SPA
app.get("*", (req, res) => {
  const indexPath = path.join(CONFIG.staticDir, "index.html");
  if (fs.existsSync(indexPath)) {
    res.sendFile(indexPath);
  } else {
    res.status(404).json({ error: "Frontend not built. Check public/index.html" });
  }
});

// ── Socket.io Events ──────────────────────────────────────────────────────────

io.on("connection", (socket) => {
  state.connectedClients++;
  console.log(`[EchoSight] Client connected: ${socket.id} (total: ${state.connectedClients})`);

  // Send current server state to newly connected client
  socket.emit("server:state", {
    scanCount:     state.scanCount,
    lastScanTime:  state.lastScanTime,
    scanning:      state.scanning,
    inferenceMode: CONFIG.inferenceMode,
    config:        {
      autoScanInterval: CONFIG.autoScanInterval,
      onnxModelExists:  fs.existsSync(CONFIG.onnxModel),
    },
  });

  // Client requests a scan
  socket.on("scan:request", async (data) => {
    console.log(`[EchoSight] Scan requested by ${socket.id}`);
    await executeScan(data?.sonarFile || null, socket.id);
  });

  // Client toggles auto-scan
  socket.on("scan:auto", ({ enabled, intervalMs }) => {
    if (enabled && intervalMs >= 500) {
      if (state.autoInterval) clearInterval(state.autoInterval);
      state.autoInterval = setInterval(() => {
        if (!state.scanning) executeScan();
      }, intervalMs);
      console.log(`[EchoSight] Auto-scan enabled: ${intervalMs}ms interval`);
      io.emit("scan:auto:ack", { enabled: true, intervalMs });
    } else {
      if (state.autoInterval) {
        clearInterval(state.autoInterval);
        state.autoInterval = null;
      }
      console.log("[EchoSight] Auto-scan disabled");
      io.emit("scan:auto:ack", { enabled: false });
    }
  });

  // Client uploads a sonar file path (Termux local path)
  socket.on("file:set", ({ filePath }) => {
    const exists = fs.existsSync(filePath);
    socket.emit("file:ack", { filePath, exists });
    if (exists) {
      console.log(`[EchoSight] Sonar file set: ${filePath}`);
    } else {
      console.warn(`[EchoSight] File not found: ${filePath}`);
    }
  });

  socket.on("disconnect", () => {
    state.connectedClients = Math.max(0, state.connectedClients - 1);
    console.log(`[EchoSight] Client disconnected: ${socket.id} (remaining: ${state.connectedClients})`);
  });
});

// ── Start Server ──────────────────────────────────────────────────────────────

server.listen(CONFIG.port, "0.0.0.0", () => {
  console.log("╔═══════════════════════════════════════════╗");
  console.log("║         EchoSight Neural Radar            ║");
  console.log("║         Sovereign Local AI OS             ║");
  console.log("╠═══════════════════════════════════════════╣");
  console.log(`║  HTTP/WS : http://localhost:${CONFIG.port}         ║`);
  console.log(`║  Mode    : ${CONFIG.inferenceMode.padEnd(32)}║`);
  console.log(`║  Mock    : ${String(CONFIG.useMockData).padEnd(32)}║`);
  console.log("╚═══════════════════════════════════════════╝");

  // Optional: auto-trigger first scan on startup
  // setTimeout(() => executeScan(), 1000);

  // Optional: start auto-scan on boot
  if (CONFIG.autoScanInterval > 0) {
    state.autoInterval = setInterval(() => {
      if (!state.scanning) executeScan();
    }, CONFIG.autoScanInterval);
    console.log(`[EchoSight] Auto-scan: every ${CONFIG.autoScanInterval}ms`);
  }
});

// ── Graceful Shutdown ─────────────────────────────────────────────────────────

process.on("SIGTERM", () => {
  console.log("[EchoSight] SIGTERM received — shutting down gracefully");
  if (state.autoInterval) clearInterval(state.autoInterval);
  server.close(() => process.exit(0));
});

process.on("SIGINT", () => {
  console.log("\n[EchoSight] SIGINT received — shutting down");
  if (state.autoInterval) clearInterval(state.autoInterval);
  server.close(() => process.exit(0));
});

// Catch uncaught promise rejections (common with child process races)
process.on("unhandledRejection", (reason) => {
  console.error("[EchoSight] Unhandled rejection:", reason);
});
