const { test, before, after } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const fsp = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const http = require("node:http");
const vm = require("node:vm");

const temporaryPrefix = path.join(os.tmpdir(), "clap-lab-test-");
const temporary = fs.mkdtempSync(temporaryPrefix);
process.env.CLAP_LAB_DATA_DIR = temporary;
process.env.CLAP_LAB_HOST = "127.0.0.1";
process.env.PORT = "8788";
const { handle, resolveLabHost } = require("../lab/server.js");
let server;
let base;
before(async () => {
  server = http.createServer((req, res) => handle(req, res));
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  base = `http://127.0.0.1:${server.address().port}`;
});
after(async () => {
  await new Promise((resolve) => server.close(resolve));
  assert.ok(path.resolve(temporary).startsWith(path.resolve(temporaryPrefix)));
  await fsp.rm(temporary, { recursive: true, force: true });
});

async function jsonRequest(url, method, data) {
  const response = await fetch(base + url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: data === undefined ? undefined : JSON.stringify(data),
  });
  assert.ok(response.ok, await response.clone().text());
  return response.json();
}

const capture = { pcmVersion: 2, raw: true, sampleRate: 16000, deviceSampleStart: 100, deviceSampleEnd: 104 };
const integrity = { samples: 4, uploadErrors: 0, pcmDropped: 0, discontinuities: 0, packetMetadataMissing: 0, clippedSamples: 0 };

function startupFixture(host) {
  const listeners = [];
  const messages = [];
  const browserCommands = [];
  let interfaceReads = 0;
  const environment = { CLAP_LAB_DATA_DIR: temporary, PORT: "8788" };
  if (host !== undefined) environment.CLAP_LAB_HOST = host;
  const fakeRequire = (name) => {
    if (name === "http") return {
      createServer() {
        return { listen(port, address, callback) { listeners.push({ port, address }); callback(); } };
      },
    };
    if (name === "os") return {
      networkInterfaces() {
        interfaceReads += 1;
        // Documentation-only address: no real network interface is queried.
        return { test: [{ family: "IPv4", internal: false, address: "192.0.2.10" }] };
      },
    };
    if (name === "child_process") return { exec(command) { browserCommands.push(command); } };
    return require(name);
  };
  const sandbox = {
    require: fakeRequire, module: { exports: {} },
    __dirname: path.join(__dirname, "../lab"),
    process: { env: environment, platform: "win32" },
    console: { log(message) { messages.push(message); } },
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../lab/server.js"), "utf8"), sandbox,
    { filename: "lab/server.js" });
  return { start: sandbox.module.exports.main, listeners, messages, browserCommands,
    interfaceReads: () => interfaceReads };
}

test("recording server defaults to loopback and exposes only the PC URL", async () => {
  const fixture = startupFixture();
  await fixture.start();
  assert.deepEqual(fixture.listeners, [{ port: 8788, address: "127.0.0.1" }]);
  assert.equal(fixture.interfaceReads(), 0, "PC mode must not enumerate or advertise LAN addresses");
  assert.ok(fixture.messages.some((line) => line.includes("PC-only mode")));
  assert.equal(fixture.browserCommands[0], 'start "" "http://127.0.0.1:8788/"');
  const health = await (await fetch(base + "/api/health")).json();
  assert.deepEqual(health.urls, ["http://127.0.0.1:8788/"]);
  assert.equal(server.address().address, "127.0.0.1");
});

test("LAN recording access needs an explicit opt-in and announces access to recordings", async () => {
  const fixture = startupFixture("0.0.0.0");
  await fixture.start();
  assert.deepEqual(fixture.listeners, [{ port: 8788, address: "0.0.0.0" }]);
  assert.equal(fixture.interfaceReads(), 1);
  assert.ok(fixture.messages.some((line) => line.includes("read, create, and delete recordings")));
  assert.ok(fixture.messages.some((line) => line.includes("trusted LAN")));
  assert.ok(fixture.messages.includes("  http://192.0.2.10:8788/"));
  assert.equal(fixture.browserCommands[0], 'start "" "http://127.0.0.1:8788/"',
    "even explicit LAN mode opens only the local browser URL");
  assert.equal(resolveLabHost({}), "127.0.0.1");
  assert.equal(resolveLabHost({ CLAP_LAB_HOST: "127.0.0.1" }), "127.0.0.1");
  assert.equal(resolveLabHost({ CLAP_LAB_HOST: "0.0.0.0" }), "0.0.0.0");
  for (const invalid of ["localhost", "::", "*", "192.0.2.10"]) {
    assert.throws(() => resolveLabHost({ CLAP_LAB_HOST: invalid }), /CLAP_LAB_HOST must be/);
    assert.throws(() => startupFixture(invalid), /CLAP_LAB_HOST must be/);
  }
});

test("raw sessions validate offsets, preserve samples and events, and reject writes after close", async () => {
  const session = await jsonRequest("/api/sessions", "POST", { label: "finger_snap", capture });
  const url = `/api/sessions/${session.id}`;
  const pcm = Buffer.from([1, 0, 255, 127, 0, 128, 254, 255]);
  const uploaded = await fetch(base + url + "/audio", { method: "POST", headers: { "X-Sample-Offset": "0" }, body: pcm });
  assert.equal(uploaded.status, 200);
  assert.equal((await uploaded.json()).samples, 4);
  const outOfOrder = await fetch(base + url + "/audio", { method: "POST", headers: { "X-Sample-Offset": "0" }, body: pcm });
  assert.equal(outOfOrder.status, 409);
  const partialSample = await fetch(base + url + "/audio", { method: "POST", headers: { "X-Sample-Offset": "4" }, body: Buffer.from([0]) });
  assert.equal(partialSample.status, 400);
  await jsonRequest(url + "/events", "POST", { kind: "clap", ms: 8, sample: 104, onsetSample: 101, sessionSample: 4 });
  const ended = await jsonRequest(url + "/close", "POST", { capture, integrity });
  assert.equal(ended.trainingReady, true);
  assert.equal(ended.bytes, 8);
  const wav = Buffer.from(await (await fetch(base + url + "/audio")).arrayBuffer());
  assert.equal(wav.readUInt32LE(24), 16000);
  assert.deepEqual(wav.subarray(44), pcm);
  const events = (await (await fetch(base + url + "/events")).text()).trim();
  assert.equal(JSON.parse(events).onsetSample, 101);
  assert.equal((await fetch(base + url + "/audio", { method: "POST", body: pcm })).status, 409);
  assert.equal((await fetch(base + url + "/events", { method: "POST", body: "{}" })).status, 409);
  assert.equal((await jsonRequest(url + "/close", "POST", {})).bytes, 8);
});

test("close waits for an earlier audio request whose body is still arriving", async () => {
  const smallCapture = { ...capture, deviceSampleEnd: 102 };
  const session = await jsonRequest("/api/sessions", "POST", { label: "clap_far", capture: smallCapture });
  const url = `/api/sessions/${session.id}`;
  let upload;
  const firstRequest = new Promise((resolve) => server.once("request", resolve));
  const completedUpload = new Promise((resolve, reject) => {
    upload = http.request(base + url + "/audio", { method: "POST", headers: { "Content-Length": "4", "X-Sample-Offset": "0" } }, (res) => {
      res.resume();
      res.on("end", () => resolve(res.statusCode));
    });
    upload.on("error", reject);
    upload.write(Buffer.from([1, 0]));
  });
  await firstRequest;
  const closeRequest = new Promise((resolve) => server.once("request", resolve));
  const closed = jsonRequest(url + "/close", "POST", { capture: smallCapture, integrity: { ...integrity, samples: 2 } });
  await closeRequest;
  upload.end(Buffer.from([2, 0]));
  assert.equal(await completedUpload, 200);
  const ended = await closed;
  assert.equal(ended.bytes, 4);
  assert.equal(ended.trainingReady, true);
  const wav = Buffer.from(await (await fetch(base + url + "/audio")).arrayBuffer());
  assert.deepEqual(wav.subarray(44), Buffer.from([1, 0, 2, 0]));
});

test("legacy, missing integrity, and dropped audio remain review-only", async () => {
  for (const variant of ["legacy", "missing", "dropped", "unchecked"]) {
    const session = await jsonRequest("/api/sessions", "POST", { label: "other", capture: variant === "legacy" ? undefined : capture });
    const url = `/api/sessions/${session.id}`;
    const uploaded = await fetch(base + url + "/audio", {
      method: "POST", headers: variant === "unchecked" ? {} : { "X-Sample-Offset": "0" }, body: Buffer.alloc(8),
    });
    assert.equal(uploaded.status, 200);
    const ended = await jsonRequest(url + "/close", "POST", {
      capture: variant === "legacy" ? undefined : capture,
      integrity: variant === "missing" ? undefined : { ...integrity, pcmDropped: variant === "dropped" ? 256 : 0 },
    });
    assert.equal(ended.trainingReady, false, variant);
    assert.ok(ended.integrity.reasons.length, variant);
  }
});

function browserFixture(fetchImplementation) {
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, {
      value: id === "gain" ? "4" : "", textContent: "", disabled: false,
      classList: { toggle() {}, add() {}, remove() {} }, style: {}, dataset: {}, children: [],
      setAttribute() {}, removeAttribute() {}, addEventListener() {}, appendChild(child) { this.children.push(child); },
      querySelector() { return { textContent: "" }; }, pause() {},
      getContext() { return { fillRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {} }; },
    });
    return elements.get(id);
  };
  const sandbox = {
    document: { getElementById: element, querySelectorAll: () => [], addEventListener() {}, createElement: () => element(Symbol()) },
    localStorage: { getItem: () => "", setItem() {} }, window: { addEventListener() {}, setTimeout },
    fetch: fetchImplementation, setTimeout, clearTimeout, AbortController, Date, ArrayBuffer, Int16Array, Float32Array,
  };
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../lab/app.js"), "utf8"), sandbox);
  return { sandbox, element, run: (code) => vm.runInContext(code, sandbox) };
}

function response(data, ok = true) {
  return { ok, headers: { get: () => "application/json" }, json: async () => data, text: async () => String(data) };
}

test("browser serializes PCM uploads, keeps device timing, and drains before closing", async () => {
  const writes = [];
  let releaseFirst;
  const firstUpload = new Promise((resolve) => { releaseFirst = resolve; });
  const fixture = browserFixture(async (url, opts = {}) => {
    if (url === "/api/sessions" && opts.method === "POST") return response({ id: "take" });
    if (url === "/api/sessions") return response({ sessions: [] });
    writes.push({ url, opts });
    if (url.endsWith("/audio") && writes.filter((r) => r.url.endsWith("/audio")).length === 1) await firstUpload;
    return response(url.endsWith("/close") ? { trainingReady: true } : {});
  });
  await fixture.run("state.ws = { readyState: 1 }; state.latestPcmAt = Date.now(); state.pcmInfo = {pcmVersion:2,sampleRate:16000}; startRec('finger_snap')");
  fixture.run("onEspMessage({data:JSON.stringify({t:'pcm',pcmVersion:2,sampleRate:16000,sample:100,samples:2,pcmDropped:5})}); onEspMessage({data:new Int16Array([1,2]).buffer}); flushAudio();");
  await Promise.resolve();
  fixture.run("onEspMessage({data:JSON.stringify({t:'ev',kind:'clap',ms:25,sample:102,onsetSample:101,sampleRate:16000,hp:88})}); onEspMessage({data:JSON.stringify({t:'pcm',pcmVersion:2,sampleRate:16000,sample:102,samples:2,pcmDropped:5})}); onEspMessage({data:new Int16Array([3,4]).buffer});");
  const stopping = fixture.run("stopRec()");
  fixture.run("onEspMessage({data:new Int16Array([99,99]).buffer})");
  assert.equal(writes.length, 1, "second upload/event/close cannot start while first is pending");
  releaseFirst();
  await stopping;
  assert.deepEqual(writes.map((r) => r.url), ["/api/sessions/take/audio", "/api/sessions/take/events", "/api/sessions/take/audio", "/api/sessions/take/close"]);
  assert.equal(writes[0].opts.headers["X-Sample-Offset"], "0");
  assert.equal(writes[2].opts.headers["X-Sample-Offset"], "2");
  assert.deepEqual(Array.from(new Int16Array(writes[2].opts.body)), [3, 4]);
  const event = JSON.parse(writes[1].opts.body);
  assert.equal(event.ms, 25);
  assert.equal(event.sample, 102);
  assert.equal(event.onsetSample, 101);
  assert.equal(event.sessionOnsetSample, 1);
  assert.equal(event.browserSampleOffset, 2);
  const closed = JSON.parse(writes[3].opts.body);
  assert.equal(closed.integrity.samples, 4);
  assert.equal(closed.integrity.pcmDropped, 0, "historical drops before the take do not contaminate it");
  assert.equal(closed.capture.deviceSampleStart, 100);
  assert.equal(closed.capture.deviceSampleEnd, 104);
});

test("browser records gaps and HTTP failure without pretending a save is training-ready", async () => {
  let closeBody;
  let audioRequests = 0;
  const fixture = browserFixture(async (url, opts = {}) => {
    if (url === "/api/sessions" && opts.method === "POST") return response({ id: "failed" });
    if (url === "/api/sessions") return response({ sessions: [] });
    if (url.endsWith("/audio")) { audioRequests += 1; return response("disk failed", false); }
    if (url.endsWith("/close")) { closeBody = JSON.parse(opts.body); return response({ trainingReady: false, integrity: { reasons: ["uploadErrors"] } }); }
    return response({});
  });
  await fixture.run("state.ws = { readyState: 1 }; state.latestPcmAt = Date.now(); startRec('clap_far')");
  fixture.run("acceptPcm(new Int16Array([1,2]),{pcmVersion:2,sampleRate:16000,sample:100,samples:2,pcmDropped:0}); flushAudio(); acceptPcm(new Int16Array([3,4]),{pcmVersion:2,sampleRate:16000,sample:110,samples:2,pcmDropped:8})");
  await fixture.run("stopRec()");
  assert.equal(audioRequests, 1, "a failed upload must not shift later PCM into its place");
  assert.equal(closeBody.integrity.uploadErrors, 1);
  assert.equal(closeBody.integrity.discontinuities, 1);
  assert.equal(closeBody.integrity.pcmDropped, 8);
  assert.match(fixture.element("session-line").textContent, /review only/);
});

test("browser retains a failed close for retry and excludes additional audio while saving", async () => {
  let closes = 0;
  let uploads = 0;
  const fixture = browserFixture(async (url, opts = {}) => {
    if (url === "/api/sessions" && opts.method === "POST") return response({ id: "retry" });
    if (url === "/api/sessions") return response({ sessions: [] });
    if (url.endsWith("/audio")) uploads += 1;
    if (url.endsWith("/close")) {
      closes += 1;
      return closes === 1 ? response("server unavailable", false) : response({ trainingReady: true });
    }
    return response({});
  });
  await fixture.run("state.ws = { readyState: 1 }; state.latestPcmAt = Date.now(); startRec('finger_snap')");
  fixture.run("acceptPcm(new Int16Array([1,2]),{pcmVersion:2,sampleRate:16000,sample:100,samples:2,pcmDropped:0})");
  await fixture.run("stopRec()");
  assert.equal(fixture.run("state.rec.id"), "retry");
  assert.equal(fixture.element("save-take").disabled, false);
  fixture.run("acceptPcm(new Int16Array([99]),null)");
  await fixture.run("stopRec()");
  assert.equal(fixture.run("state.rec"), null);
  assert.equal(uploads, 1);
  assert.equal(closes, 2);
});

test("recording requires a live stream and listening gain cannot alter raw samples", async () => {
  const sent = [];
  let sessions = 0;
  const fixture = browserFixture(async (url, opts = {}) => {
    if (opts.method === "POST") sessions += 1;
    return response({ sessions: [] });
  });
  await fixture.run("startRec('clap_far')");
  assert.equal(sessions, 0);
  assert.match(fixture.element("session-line").textContent, /Connect the ESP32/);
  fixture.sandbox.sent = sent;
  fixture.run("state.ws = {readyState:1, send: message => sent.push(message)}; setGain(8)");
  assert.equal(sent.length, 0);
});

test("connection accepts a nine-character password", () => {
  const urls = [];
  const fixture = browserFixture(async () => response({ sessions: [] }));
  fixture.sandbox.WebSocket = class {
    constructor(url) { urls.push(url); }
    close() {}
  };
  fixture.element("esp-pass").value = "test12345";
  fixture.run("connect()");
  assert.equal(urls[0], "ws://clap.local:81/?k=test12345");
});
