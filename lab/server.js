/**
 * Clap Lab file writer.
 * Serves ./lab static files and writes session WAV + JSONL under ../data/sessions.
 * PC-only by default. LAN access requires an explicit CLAP_LAB_HOST opt-in.
 */
const http = require("http");
const fsp = require("fs/promises");
const path = require("path");
const os = require("os");
const { exec } = require("child_process");

const ROOT = __dirname;
const PUBLIC_DIR = ROOT;
const DATA_DIR = path.resolve(process.env.CLAP_LAB_DATA_DIR || path.join(ROOT, "..", "data", "sessions"));
const PORT = Number(process.env.PORT) || 8788;
const HOST = resolveLabHost(process.env);
const MAX_BODY = 8 * 1024 * 1024;
const sessionWrites = new Map();

// Reserve at request arrival, before reading its body. A close request must never
// overtake an audio upload whose body is still in flight.
function serializeSession(id, action) {
  const previous = sessionWrites.get(id) || Promise.resolve();
  const next = previous.catch(() => {}).then(action);
  sessionWrites.set(id, next);
  next.finally(() => {
    if (sessionWrites.get(id) === next) sessionWrites.delete(id);
  }).catch(() => {});
  return next;
}

async function readSession(dir) {
  return JSON.parse(await fsp.readFile(path.join(dir, "session.json"), "utf8"));
}

async function writeSession(dir, session) {
  const temporary = path.join(dir, "session.json.tmp");
  await fsp.writeFile(temporary, JSON.stringify(session, null, 2) + "\n");
  await fsp.rename(temporary, path.join(dir, "session.json"));
}

function captureMetadata(value) {
  const c = value && typeof value === "object" ? value : {};
  return {
    pcmVersion: c.pcmVersion === 2 ? 2 : 1,
    raw: c.raw === true && c.pcmVersion === 2,
    format: "s16le",
    sampleRate: 16000,
    mode: c.raw === true && c.pcmVersion === 2 ? "raw" : "legacy",
    deviceSampleStart: Number.isSafeInteger(c.deviceSampleStart) && c.deviceSampleStart >= 0 ? c.deviceSampleStart : null,
    deviceSampleEnd: Number.isSafeInteger(c.deviceSampleEnd) && c.deviceSampleEnd >= 0 ? c.deviceSampleEnd : null,
  };
}

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".woff2": "font/woff2",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".wav": "audio/wav",
};

function sendJson(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
    "Content-Length": Buffer.byteLength(body),
  });
  res.end(body);
}

function sendText(res, status, text) {
  res.writeHead(status, { "Content-Type": "text/plain; charset=utf-8" });
  res.end(text);
}

function resolvePublic(urlPath) {
  const decoded = decodeURIComponent(urlPath.split("?")[0]);
  const relative = decoded === "/" ? "index.html" : decoded.replace(/^\/+/, "");
  const resolved = path.resolve(PUBLIC_DIR, relative);
  if (resolved !== PUBLIC_DIR && !resolved.startsWith(PUBLIC_DIR + path.sep)) return null;
  return resolved;
}

function sessionDir(id) {
  if (!/^[a-zA-Z0-9][a-zA-Z0-9._-]*$/.test(id)) return null;
  return path.join(DATA_DIR, id);
}

async function readBody(req, max) {
  const chunks = [];
  let n = 0;
  for await (const chunk of req) {
    n += chunk.length;
    if (n > max) throw new Error("body too large");
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
}

function wavHeader(dataBytes, sampleRate) {
  const b = Buffer.alloc(44);
  b.write("RIFF", 0);
  b.writeUInt32LE(36 + dataBytes, 4);
  b.write("WAVE", 8);
  b.write("fmt ", 12);
  b.writeUInt32LE(16, 16);
  b.writeUInt16LE(1, 20);
  b.writeUInt16LE(1, 22);
  b.writeUInt32LE(sampleRate, 24);
  b.writeUInt32LE(sampleRate * 2, 28);
  b.writeUInt16LE(2, 32);
  b.writeUInt16LE(16, 34);
  b.write("data", 36);
  b.writeUInt32LE(dataBytes, 40);
  return b;
}

async function closeWav(dir, sampleRate = 16000) {
  const pcmPath = path.join(dir, "audio.pcm");
  const wavPath = path.join(dir, "audio.wav");
  let pcm = Buffer.alloc(0);
  try {
    pcm = await fsp.readFile(pcmPath);
  } catch {
    pcm = Buffer.alloc(0);
  }
  const wav = Buffer.concat([wavHeader(pcm.length, sampleRate), pcm]);
  await fsp.writeFile(wavPath, wav);
  return pcm.length;
}

async function countEvents(dir) {
  try {
    const ev = await fsp.readFile(path.join(dir, "events.jsonl"), "utf8");
    return ev.split("\n").filter((line) => line.trim()).length;
  } catch {
    return 0;
  }
}

async function pcmBytes(dir) {
  try {
    return (await fsp.stat(path.join(dir, "audio.pcm"))).size;
  } catch {
    try {
      const wav = (await fsp.stat(path.join(dir, "audio.wav"))).size;
      return Math.max(0, wav - 44);
    } catch {
      return 0;
    }
  }
}

async function sessionSummary(id) {
  const dir = sessionDir(id);
  if (!dir) return null;
  const jsonPath = path.join(dir, "session.json");
  let session;
  try {
    session = JSON.parse(await fsp.readFile(jsonPath, "utf8"));
  } catch {
    return null;
  }
  const bytes = session.bytes ?? (await pcmBytes(dir));
  session.id = session.id || id;
  session.events = await countEvents(dir);
  session.bytes = bytes;
  session.seconds = Number((bytes / 2 / (session.rate || 16000)).toFixed(2));
  session.hasAudio = bytes > 0;
  return session;
}

async function listSessions() {
  await fsp.mkdir(DATA_DIR, { recursive: true });
  const names = await fsp.readdir(DATA_DIR).catch(() => []);
  const sessions = [];
  for (const name of names) {
    const summary = await sessionSummary(name);
    if (summary) sessions.push(summary);
  }
  sessions.sort((a, b) => String(b.id).localeCompare(String(a.id)));
  return sessions;
}

function resolveLabHost(environment) {
  const host = environment.CLAP_LAB_HOST;
  if (!host || host === "127.0.0.1") return "127.0.0.1";
  if (host === "0.0.0.0") return host;
  throw new Error("CLAP_LAB_HOST must be 127.0.0.1 (PC only) or 0.0.0.0 (explicit trusted-LAN access)");
}

function labUrls(port) {
  const urls = [`http://127.0.0.1:${port}/`];
  if (HOST !== "0.0.0.0") return urls;
  for (const addrs of Object.values(os.networkInterfaces())) {
    for (const a of addrs || []) {
      if (a.family === "IPv4" && !a.internal) urls.push(`http://${a.address}:${port}/`);
    }
  }
  return urls;
}

async function handleApi(req, res, url, serialized = false) {
  const parts = url.pathname.split("/").filter(Boolean);
  if (!serialized && parts[0] === "api" && parts[1] === "sessions" && parts[2] && ["POST", "PATCH", "DELETE"].includes(req.method)) {
    return serializeSession(parts[2], () => handleApi(req, res, url, true));
  }

  if (req.method === "GET" && url.pathname === "/api/health") {
    return sendJson(res, 200, { ok: true, urls: labUrls(PORT) });
  }

  if (req.method === "GET" && url.pathname === "/api/sessions") {
    return sendJson(res, 200, { sessions: await listSessions() });
  }

  if (req.method === "POST" && url.pathname === "/api/sessions") {
    const raw = await readBody(req, 64 * 1024);
    let body = {};
    try {
      body = JSON.parse(raw.toString("utf8") || "{}");
    } catch {
      return sendText(res, 400, "json");
    }
    if (!body || typeof body !== "object" || Array.isArray(body)) return sendText(res, 400, "json object required");
    const label = String(body.label || "other");
    const allowed = ["clap_far", "tongue_click", "door_slam", "finger_snap", "other"];
    if (!allowed.includes(label)) return sendText(res, 400, "label");
    const id = new Date().toISOString().replace(/[:.]/g, "-") + "-" + require("crypto").randomBytes(3).toString("hex") + "-" + label;
    const dir = path.join(DATA_DIR, id);
    await fsp.mkdir(dir, { recursive: true });
    const session = {
      id,
      label,
      started: new Date().toISOString(),
      rate: 16000,
      channels: 1,
      note: "",
      capture: captureMetadata(body.capture),
      trainingReady: false,
      audioUploads: 0,
      checkedUploads: 0,
    };
    await writeSession(dir, session);
    await fsp.writeFile(path.join(dir, "events.jsonl"), "");
    await fsp.writeFile(path.join(dir, "audio.pcm"), Buffer.alloc(0));
    return sendJson(res, 201, session);
  }

  if (parts[0] === "api" && parts[1] === "sessions" && parts[2] && !parts[3] && req.method === "GET") {
    const session = await sessionSummary(parts[2]);
    if (!session) return sendText(res, 404, "missing");
    return sendJson(res, 200, session);
  }

  if (parts[0] === "api" && parts[1] === "sessions" && parts[2] && !parts[3] && req.method === "PATCH") {
    const dir = sessionDir(parts[2]);
    if (!dir) return sendText(res, 400, "id");
    let session;
    try {
      session = await readSession(dir);
    } catch {
      return sendText(res, 404, "missing");
    }
    const raw = await readBody(req, 64 * 1024);
    let body = {};
    try {
      body = JSON.parse(raw.toString("utf8") || "{}");
    } catch {
      return sendText(res, 400, "json");
    }
    if (!body || typeof body !== "object" || Array.isArray(body)) return sendText(res, 400, "json object required");
    if (typeof body.note === "string") session.note = body.note.slice(0, 240);
    await writeSession(dir, session);
    return sendJson(res, 200, session);
  }

  if (parts[0] === "api" && parts[1] === "sessions" && parts[2] && !parts[3] && req.method === "DELETE") {
    const dir = sessionDir(parts[2]);
    if (!dir) return sendText(res, 400, "id");
    try {
      await fsp.rm(dir, { recursive: true, force: true });
    } catch {
      return sendText(res, 404, "missing");
    }
    return sendJson(res, 200, { ok: true, id: parts[2] });
  }

  if (parts[0] === "api" && parts[1] === "sessions" && parts[2] && parts[3] === "audio" && req.method === "GET") {
    const dir = sessionDir(parts[2]);
    if (!dir) return sendText(res, 400, "id");
    const wavPath = path.join(dir, "audio.wav");
    const pcmPath = path.join(dir, "audio.pcm");
    let wav;
    try {
      wav = await fsp.readFile(wavPath);
    } catch {
      let pcm = Buffer.alloc(0);
      try {
        pcm = await fsp.readFile(pcmPath);
      } catch {
        pcm = Buffer.alloc(0);
      }
      wav = Buffer.concat([wavHeader(pcm.length, 16000), pcm]);
    }
    res.writeHead(200, {
      "Content-Type": "audio/wav",
      "Content-Disposition": url.searchParams.get("dl") === "1"
        ? `attachment; filename="${parts[2]}.wav"`
        : "inline",
      "Content-Length": wav.length,
      "Cache-Control": "no-store",
    });
    return res.end(wav);
  }

  if (parts[0] === "api" && parts[1] === "sessions" && parts[2] && parts[3] === "events" && req.method === "GET") {
    const dir = sessionDir(parts[2]);
    if (!dir) return sendText(res, 400, "id");
    let body = "";
    try {
      body = await fsp.readFile(path.join(dir, "events.jsonl"), "utf8");
    } catch {
      body = "";
    }
    const buf = Buffer.from(body, "utf8");
    res.writeHead(200, {
      "Content-Type": "application/x-ndjson; charset=utf-8",
      "Content-Disposition": `attachment; filename="${parts[2]}.jsonl"`,
      "Content-Length": buf.length,
      "Cache-Control": "no-store",
    });
    return res.end(buf);
  }

  if (parts[0] === "api" && parts[1] === "sessions" && parts[2] && parts[3] === "audio" && req.method === "POST") {
    const dir = sessionDir(parts[2]);
    if (!dir) return sendText(res, 400, "id");
    const session = await readSession(dir);
    if (session.ended) return sendText(res, 409, "session is closed");
    const buf = await readBody(req, MAX_BODY);
    if (!buf.length || buf.length % 2) return sendText(res, 400, "audio must contain complete int16 samples");
    const currentBytes = await pcmBytes(dir);
    const offsetHeader = req.headers["x-sample-offset"];
    if (offsetHeader != null) {
      const offset = Number(offsetHeader);
      if (!Number.isSafeInteger(offset) || offset < 0 || offset !== currentBytes / 2) {
        return sendText(res, 409, "audio sample offset mismatch");
      }
      session.checkedUploads = (session.checkedUploads || 0) + 1;
    }
    await fsp.appendFile(path.join(dir, "audio.pcm"), buf);
    session.audioUploads = (session.audioUploads || 0) + 1;
    await writeSession(dir, session);
    return sendJson(res, 200, { bytes: buf.length, sampleOffset: currentBytes / 2, samples: (currentBytes + buf.length) / 2 });
  }

  if (parts[0] === "api" && parts[1] === "sessions" && parts[2] && parts[3] === "events" && req.method === "POST") {
    const dir = sessionDir(parts[2]);
    if (!dir) return sendText(res, 400, "id");
    const session = await readSession(dir);
    if (session.ended) return sendText(res, 409, "session is closed");
    const raw = await readBody(req, 64 * 1024);
    let line = raw.toString("utf8").trim();
    if (!line) return sendText(res, 400, "empty");
    try {
      const event = JSON.parse(line);
      if (!event || typeof event !== "object" || Array.isArray(event)) return sendText(res, 400, "event must be an object");
      line = JSON.stringify(event);
    } catch {
      return sendText(res, 400, "json");
    }
    await fsp.appendFile(path.join(dir, "events.jsonl"), line + "\n");
    return sendJson(res, 200, { ok: true });
  }

  if (parts[0] === "api" && parts[1] === "sessions" && parts[2] && parts[3] === "close" && req.method === "POST") {
    const dir = sessionDir(parts[2]);
    if (!dir) return sendText(res, 400, "id");
    let session = await readSession(dir);
    if (session.ended) return sendJson(res, 200, session);
    const raw = await readBody(req, 64 * 1024);
    let body;
    try {
      body = JSON.parse(raw.toString("utf8") || "{}");
    } catch {
      return sendText(res, 400, "json");
    }
    if (!body || typeof body !== "object" || Array.isArray(body)) return sendText(res, 400, "json object required");
    const bytes = await closeWav(dir, session.rate);
    session.capture = captureMetadata(body.capture || session.capture);
    const input = body.integrity || {};
    const reasons = [];
    const measuredSamples = bytes / 2;
    const integrity = { samples: measuredSamples };
    for (const key of ["uploadErrors", "pcmDropped", "discontinuities", "packetMetadataMissing", "clippedSamples"]) {
      integrity[key] = Number.isSafeInteger(input[key]) && input[key] >= 0 ? input[key] : 0;
    }
    if (!session.capture.raw || session.capture.pcmVersion !== 2) reasons.push("legacy or amplified audio");
    if (!measuredSamples) reasons.push("empty audio");
    if (!Number.isSafeInteger(input.samples) || input.samples !== measuredSamples) reasons.push("uploaded sample count mismatch");
    if (!session.audioUploads || session.checkedUploads !== session.audioUploads) reasons.push("unchecked audio uploads");
    if (session.capture.deviceSampleStart == null || session.capture.deviceSampleEnd == null || session.capture.deviceSampleEnd - session.capture.deviceSampleStart !== measuredSamples) reasons.push("device sample range mismatch");
    for (const key of ["uploadErrors", "pcmDropped", "discontinuities", "packetMetadataMissing"]) {
      if (!Number.isSafeInteger(input[key]) || input[key] < 0 || integrity[key]) reasons.push(key);
    }
    integrity.reasons = reasons;
    session.integrity = integrity;
    session.trainingReady = reasons.length === 0;
    session.ended = new Date().toISOString();
    session.bytes = bytes;
    session.seconds = bytes / 2 / 16000;
    await writeSession(dir, session);
    return sendJson(res, 200, session);
  }

  return sendText(res, 404, "not found");
}

async function handle(req, res) {
  const url = new URL(req.url, "http://127.0.0.1");
  try {
    if (url.pathname.startsWith("/api/")) return await handleApi(req, res, url);
    const file = resolvePublic(url.pathname);
    if (!file) return sendText(res, 403, "no");
    const data = await fsp.readFile(file);
    const ext = path.extname(file);
    res.writeHead(200, { "Content-Type": MIME[ext] || "application/octet-stream" });
    res.end(data);
  } catch (err) {
    if (err && err.code === "ENOENT") return sendText(res, 404, "missing");
    if (err && err.message === "body too large") return sendText(res, 413, err.message);
    sendText(res, 500, String(err.message || err));
  }
}

async function main() {
  await fsp.mkdir(DATA_DIR, { recursive: true });
  const server = http.createServer((req, res) => {
    handle(req, res);
  });
  server.listen(PORT, HOST, () => {
    const urls = labUrls(PORT);
    console.log("Clap Lab");
    for (const url of urls) console.log("  " + url);
    console.log("Recordings stay on this PC.");
    if (HOST === "0.0.0.0") {
      console.log("LAN access enabled: other devices that can reach this PC can read, create, and delete recordings.");
      console.log("Use this option only on a trusted LAN. Phone on Wi-Fi: use a LAN URL above.");
    } else {
      console.log("PC-only mode: the recording server listens only on this computer's loopback interface.");
    }
    if (process.platform === "win32" && process.env.CLAP_LAB_NO_OPEN !== "1") exec('start "" "' + urls[0] + '"');
  });
  return server;
}

if (require.main === module) main();
module.exports = { handle, wavHeader, resolveLabHost, main };
