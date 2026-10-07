const $ = (id) => document.getElementById(id);

const state = {
  ip: localStorage.getItem("clap.esp") || "clap.local",
  pass: localStorage.getItem("clap.pass") || "",
  ws: null,
  audio: null,
  gainNode: null,
  playAt: 0,
  listen: false,
  rec: null,
  counts: { clap_far: 0, finger_snap: 0, tongue_click: 0, door_slam: 0, other: 0 },
  hist: new Float32Array(640),
  peak: 0,
  bay: "capture",
  lampOn: false,
  lampBri: 1000,
  lampTemp: 500,
  pendingDelete: null,
  wantLink: false,
  hadLink: false,
  pcmHeader: null,
  pcmInfo: null,
  latestPcmAt: 0,
  recStarting: false,
};

$("esp-ip").value = state.ip;
$("esp-pass").value = state.pass;

const wave = $("wave");
const ctx2d = wave.getContext("2d");

function kelvin(t) {
  return Math.round(2700 + (Number(t) / 1000) * 3800) + " K";
}

function setFlag(id, on, textOn, textOff) {
  const el = $(id);
  const slide = el.querySelector(".flag__slide");
  el.classList.toggle("is-ok", !!on);
  el.classList.toggle("is-bad", on === false);
  slide.textContent = on ? textOn : textOff;
}

function drawWave() {
  const w = wave.width;
  const h = wave.height;
  ctx2d.fillStyle = "#071614";
  ctx2d.fillRect(0, 0, w, h);
  ctx2d.strokeStyle = "rgba(62,240,200,0.12)";
  ctx2d.lineWidth = 1;
  for (let y = 0; y < h; y += 30) {
    ctx2d.beginPath();
    ctx2d.moveTo(0, y);
    ctx2d.lineTo(w, y);
    ctx2d.stroke();
  }
  ctx2d.strokeStyle = "#3ef0c8";
  ctx2d.lineWidth = 1.6;
  ctx2d.beginPath();
  const n = state.hist.length;
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * w;
    const y = h / 2 - state.hist[i] * (h * 0.42);
    if (i === 0) ctx2d.moveTo(x, y);
    else ctx2d.lineTo(x, y);
  }
  ctx2d.stroke();
}

function park(show, text) {
  const el = $("park");
  if (text) el.textContent = text;
  el.classList.toggle("is-hidden", !show);
}

function pushHist(int16) {
  const step = Math.max(1, Math.floor(int16.length / 8));
  state.hist.copyWithin(0, 8);
  let k = 0;
  for (let i = 0; i < 8; i++) {
    const s = int16[Math.min(int16.length - 1, i * step)] / 32768;
    state.hist[state.hist.length - 8 + i] = s;
    const a = Math.abs(s);
    if (a > k) k = a;
  }
  state.peak = Math.max(k, state.peak * 0.88);
  $("vu-needle").style.left = `${Math.min(98, state.peak * 100)}%`;
  drawWave();
}

function ensureAudio() {
  if (state.audio) return;
  const Ctx = window.AudioContext || window.webkitAudioContext;
  try {
    state.audio = new Ctx();
  } catch {
    state.audio = new Ctx();
  }
  state.gainNode = state.audio.createGain();
  state.gainNode.gain.value = Number($("gain").value) || 4;
  state.gainNode.connect(state.audio.destination);
}

function playPcm(int16) {
  if (!state.listen || !state.audio || !int16.length || state.rec) return;
  if (state.audio.state === "suspended") state.audio.resume();
  const rate = state.audio.sampleRate || 44100;
  const n = Math.max(1, Math.round(int16.length * rate / 16000));
  const f = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const src = (i * 16000) / rate;
    const i0 = Math.min(int16.length - 1, src | 0);
    let s = int16[i0] / 32768;
    if (s > 1) s = 1;
    if (s < -1) s = -1;
    f[i] = s;
  }
  const buf = state.audio.createBuffer(1, f.length, rate);
  buf.copyToChannel(f, 0);
  const src = state.audio.createBufferSource();
  src.buffer = buf;
  src.connect(state.gainNode);
  const now = state.audio.currentTime;
  if (state.playAt < now - 0.05) state.playAt = now;
  const t = Math.max(now, state.playAt);
  try {
    src.start(t);
    state.playAt = t + buf.duration;
  } catch {
    /* dropped a chunk */
  }
}

async function api(path, opts) {
  const abort = new AbortController();
  const timeout = setTimeout(() => abort.abort(), 10000);
  try {
    const res = await fetch(path, { ...opts, signal: abort.signal });
    if (!res.ok) throw new Error(await res.text());
    const type = res.headers.get("content-type") || "";
    if (type.includes("json")) return await res.json();
    return res;
  } finally {
    clearTimeout(timeout);
  }
}

async function refreshCounts() {
  try {
    const data = await api("/api/sessions");
    const c = { clap_far: 0, finger_snap: 0, tongue_click: 0, door_slam: 0, other: 0 };
    for (const s of data.sessions || []) {
      if (c[s.label] != null) c[s.label] += 1;
    }
    state.counts = c;
    $("counts").textContent =
      `clap_far ${c.clap_far} · finger_snap ${c.finger_snap} · tongue_click ${c.tongue_click} · door_slam ${c.door_slam} · other ${c.other}`;
    renderTakes(data.sessions || []);
  } catch {
    /* lab server missing */
  }
}

function renderTakes(sessions) {
  const ul = $("takes");
  ul.innerHTML = "";
  $("takes-empty").classList.toggle("is-hidden", sessions.length > 0);
  for (const s of sessions) {
    const li = document.createElement("li");
    li.className = "take";
    li.dataset.id = s.id;
    const dur = s.seconds != null ? Number(s.seconds).toFixed(1) + "s" : "—";
    const kb = s.bytes != null ? Math.round(s.bytes / 1024) + " KB" : "—";
    li.innerHTML = `
      <div>
        <p class="take__id">${s.id}</p>
        <p class="take__meta">${s.label} · ${dur} · ${kb} · ${s.events || 0} events${s.ended ? (s.trainingReady ? " · ready for training" : " · review only") : " · open"}</p>
      </div>
      <div class="take__actions">
        <button type="button" class="mark" data-act="play">Play</button>
        <a class="mark" data-act="wav" href="/api/sessions/${encodeURIComponent(s.id)}/audio?dl=1">WAV</a>
        <a class="mark" data-act="ev" href="/api/sessions/${encodeURIComponent(s.id)}/events">JSONL</a>
        <button type="button" class="mark" data-act="del">Delete</button>
      </div>`;
    ul.appendChild(li);
  }
}

function stampRail(kind, dur, hp, ms) {
  const li = document.createElement("li");
  li.dataset.kind = kind;
  const t = ((ms ?? Date.now()) / 1000).toFixed(1);
  li.innerHTML = `${kind}<br>${t}s<br>${hp || "—"}`;
  $("rail").appendChild(li);
  while ($("rail").children.length > 48) $("rail").removeChild($("rail").firstChild);
  $("rail").scrollLeft = $("rail").scrollWidth;
}

function recordingError(rec, message) {
  rec.integrity.uploadErrors += 1;
  rec.uploadFailed = true;
  if (state.rec === rec) $("session-line").textContent = "Take is incomplete: " + message + ". Save it for review, then record again.";
}

function queueWrite(rec, action) {
  rec.uploadQueue = rec.uploadQueue.then(action).catch((err) => recordingError(rec, err.message));
  return rec.uploadQueue;
}

function postEvent(kind, extra) {
  const rec = state.rec;
  if (!rec || rec.closing) return Promise.resolve();
  const receivedAt = Date.now();
  const body = Object.assign({ kind, t: receivedAt, session: rec.id }, extra || {});
  body.receivedAt = receivedAt;
  body.browserSampleOffset = rec.samples;
  const origin = rec.capture.deviceSampleStart;
  body.sessionSample = origin != null && Number.isSafeInteger(body.sample) ? body.sample - origin : null;
  body.sessionOnsetSample = origin != null && Number.isSafeInteger(body.onsetSample) ? body.onsetSample - origin : null;
  return queueWrite(rec, async () => {
    await api(`/api/sessions/${rec.id}/events`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  });
}

function acceptPcm(int16, header) {
  const rec = state.rec;
  if (!rec || rec.closing) return;
  const valid = header && header.pcmVersion === 2 && header.sampleRate === 16000 &&
    Number.isSafeInteger(header.sample) && header.sample >= 0 && header.samples === int16.length &&
    Number.isSafeInteger(header.pcmDropped) && header.pcmDropped >= 0;
  if (valid) {
    if (rec.capture.deviceSampleStart == null) {
      rec.capture.deviceSampleStart = header.sample;
      rec.capture.raw = true;
      rec.capture.pcmVersion = 2;
      rec.capture.mode = "raw";
      rec.dropStart = header.pcmDropped;
    } else if (header.sample !== rec.capture.deviceSampleEnd) {
      rec.integrity.discontinuities += 1;
    }
    rec.capture.deviceSampleEnd = header.sample + int16.length;
    rec.integrity.pcmDropped = Math.max(rec.integrity.pcmDropped, header.pcmDropped - rec.dropStart);
  } else {
    rec.integrity.packetMetadataMissing += 1;
  }
  for (const sample of int16) if (Math.abs(sample) >= 32760) rec.integrity.clippedSamples += 1;
  rec.chunks.push(new Int16Array(int16));
  rec.samples += int16.length;
  if (rec.chunks.length >= 8) flushAudio(rec);
}

function onEspMessage(ev) {
  if (typeof ev.data === "string") {
    let j;
    try {
      j = JSON.parse(ev.data);
    } catch {
      return;
    }
    if (j.t === "pcm") {
      if (state.pcmHeader && state.rec && !state.rec.closing) state.rec.integrity.discontinuities += 1;
      state.pcmHeader = j;
      state.pcmInfo = j;
      return;
    }
    if (j.t === "stat") {
      if (j.aiMode && $("ai-state")) {
        const status = j.aiMode === "dsp" ? "DSP detector active; AI model is disabled."
          : j.aiMode.includes("unavailable") ? "AI could not start. Check device status."
          : j.aiMode === "shadow" ? "AI comparison mode: DSP controls the light."
          : "AI verifies each gesture before light control.";
        $("ai-state").textContent = status + ` ${j.aiDropped || 0} skipped candidates; ${j.aiErrors || 0} errors.`;
      }
      setFlag("m-tuya", j.tuya === 1, "LIVE", "DOWN");
      setFlag("m-i2s", j.i2s === 1, "LIVE", "DOWN");
      setFlag("m-lamp", j.lamp === 1, "ON", "OFF");
      $("m-heap").textContent = j.heap ? String(j.heap) : "—";
      $("m-arm").textContent = j.arm ? String(j.arm) : "—";
      $("m-link").textContent = "ws live";
      if (j.lamp != null) syncLampUi(j.lamp === 1, j.bri, j.temp);
      if (state.rec && !state.rec.closing && state.rec.dropStart != null && Number.isSafeInteger(j.pcmDropped)) {
        state.rec.integrity.pcmDropped = Math.max(state.rec.integrity.pcmDropped, j.pcmDropped - state.rec.dropStart);
      }
    }
    if (j.t === "ev") {
      stampRail(j.kind, j.dur, j.hp, j.ms);
      if (state.rec) postEvent(j.kind, { ...j, src: "fsm", firmwareType: j.t, t: Date.now() });
    }
    if (j.t === "ai") {
      const names = ["noise", "clap", "snap"];
      if (Array.isArray(j.scores) && j.scores.length === 3) names.forEach((name, index) => {
        const score = Math.max(0, Math.min(1, Number(j.scores[index]) || 0));
        if ($(`ai-${name}`)) $(`ai-${name}`).value = score;
        if ($(`ai-${name}-score`)) $(`ai-${name}-score`).textContent = Math.round(score * 100) + "%";
      });
      if ($("ai-state")) $("ai-state").textContent =
        `${j.mode}: ${j.accepted ? "gesture accepted" : "sound rejected"}; analysis ${j.frontendMs} ms, AI ${j.inferenceMs} ms.`;
      if (state.rec) postEvent("ai", { ...j, src: "ai", deviceMs: j.ms, firmwareType: j.t, t: Date.now() });
    }
    return;
  }
  const header = state.pcmHeader;
  state.pcmHeader = null;
  if (!(ev.data instanceof ArrayBuffer) || ev.data.byteLength % 2) {
    if (state.rec && !state.rec.closing) state.rec.integrity.discontinuities += 1;
    return;
  }
  const int16 = new Int16Array(ev.data);
  state.latestPcmAt = Date.now();
  pushHist(int16);
  playPcm(int16);
  acceptPcm(int16, header);
}

function flushAudio(rec = state.rec) {
  if (!rec || !rec.chunks.length) return rec ? rec.uploadQueue : Promise.resolve();
  const total = rec.chunks.reduce((n, a) => n + a.length, 0);
  const merged = new Int16Array(total);
  let o = 0;
  for (const c of rec.chunks) {
    merged.set(c, o);
    o += c.length;
  }
  rec.chunks = [];
  const sampleOffset = rec.queuedSamples;
  rec.queuedSamples += total;
  if (rec.uploadFailed) return rec.uploadQueue;
  if (rec.pendingBytes + merged.byteLength > 2 * 1024 * 1024) {
    recordingError(rec, "file server is too slow");
    return rec.uploadQueue;
  }
  rec.pendingBytes += merged.byteLength;
  return queueWrite(rec, async () => {
    try {
      if (rec.uploadFailed) return;
      await api(`/api/sessions/${rec.id}/audio`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream", "X-Sample-Offset": String(sampleOffset) },
        body: merged.buffer,
      });
    } finally {
      rec.pendingBytes -= merged.byteLength;
    }
  });
}

function disconnect(opts) {
  if (state.rec && !state.rec.closing) {
    state.rec.integrity.discontinuities += 1;
    stopRec();
  }
  state.pcmHeader = null;
  state.pcmInfo = null;
  state.latestPcmAt = 0;
  if (opts && opts.stop) {
    state.wantLink = false;
    state.hadLink = false;
    clearTimeout(state.linkTimer);
  }
  if (state.ws) {
    state.ws.onclose = null;
    state.ws.close();
    state.ws = null;
  }
  $("m-link").textContent = "idle";
  $("connect-btn").disabled = false;
  $("connect-btn").textContent = "Connect";
}

function connect(ev) {
  if (ev) ev.preventDefault();
  const ip = $("esp-ip").value.trim();
  const pass = $("esp-pass").value.trim();
  if (!ip || pass.length < 6 || pass.length > 32) {
    park(true, "Enter the ESP32 address and its Lab password (6–32 characters).");
    return;
  }
  localStorage.setItem("clap.esp", ip);
  localStorage.setItem("clap.pass", pass);
  state.wantLink = true;
  clearTimeout(state.linkTimer);
  disconnect();
  $("connect-btn").disabled = true;
  $("connect-btn").textContent = "Linking";
  park(true, "Opening socket. Speakers stay off until you press Listen.");

  const ws = new WebSocket(`ws://${ip}:81/?k=${encodeURIComponent(pass)}`);
  ws.binaryType = "arraybuffer";
  state.ws = ws;
  ws.onopen = async () => {
    state.hadLink = true;
    $("connect-btn").disabled = false;
    $("connect-btn").textContent = "Reconnect";
    $("m-link").textContent = "ws live";
    setFlag("m-wifi", true, "OK", "DOWN");
    park(false);
    try {
      const st = await fetch(`http://${ip}/api/status?k=${encodeURIComponent(pass)}`);
      if (st.ok) {
        const j = await st.json();
        setFlag("m-tuya", j.tuya === 1, "LIVE", "DOWN");
        setFlag("m-i2s", j.i2s === 1, "LIVE", "DOWN");
        setFlag("m-lamp", j.lamp === 1, "ON", "OFF");
        $("m-heap").textContent = String(j.heap);
        $("m-arm").textContent = String(j.arm);
        $("arm-mul").value = String(j.armMul || 8);
        $("arm-val").textContent = String(j.armMul || 8);
        syncLampUi(j.lamp === 1, j.bri, j.temp);
      }
    } catch {
      /* status is optional */
    }
  };
  ws.onmessage = onEspMessage;
  ws.onclose = () => {
    if (state.rec && !state.rec.closing) {
      state.rec.integrity.discontinuities += 1;
      stopRec();
    }
    state.pcmHeader = null;
    state.pcmInfo = null;
    state.latestPcmAt = 0;
    $("m-link").textContent = "closed";
    setFlag("m-wifi", false, "OK", "DOWN");
    $("connect-btn").disabled = false;
    $("connect-btn").textContent = "Connect";
    if (state.ws === ws) state.ws = null;
    if (state.wantLink && state.hadLink) {
      park(true, "Socket dropped. Linking again. Clap still runs on the ESP32.");
      clearTimeout(state.linkTimer);
      state.linkTimer = setTimeout(() => {
        if (state.wantLink) connect();
      }, 700);
      return;
    }
    park(true, "Socket closed. Connect again. Clap still runs on the ESP32.");
  };
  ws.onerror = () => {
    park(true, "Socket error. Same Wi-Fi? Password from Serial lab pass=?");
  };
}

function syncLampUi(on, bri, temp) {
  if (on != null) {
    state.lampOn = !!on;
    $("lamp-on").classList.toggle("is-lit", state.lampOn);
    $("lamp-off").classList.toggle("is-in", !state.lampOn);
  }
  if (bri != null && bri !== "") {
    state.lampBri = Number(bri);
    $("lamp-bri").value = String(state.lampBri);
    $("lamp-bri-val").textContent = Math.round(state.lampBri / 10) + "%";
  }
  if (temp != null && temp !== "") {
    state.lampTemp = Number(temp);
    $("lamp-temp").value = String(state.lampTemp);
    $("lamp-temp-val").textContent = kelvin(state.lampTemp);
  }
}

async function setLamp(on, bri, temp) {
  const ip = $("esp-ip").value.trim();
  const pass = $("esp-pass").value.trim();
  const nextOn = on == null ? state.lampOn : !!on;
  const nextBri = bri == null ? state.lampBri : bri;
  const nextTemp = temp == null ? state.lampTemp : temp;
  syncLampUi(nextOn, nextBri, nextTemp);
  if (state.ws && state.ws.readyState === 1) {
    if (on === false) {
      state.ws.send("lamp:0");
      return;
    }
    state.ws.send("bri:" + nextBri);
    state.ws.send("temp:" + nextTemp);
    return;
  }
  if (ip && pass) {
    try {
      await fetch(
        `http://${ip}/api/lamp?k=${encodeURIComponent(pass)}&on=${nextOn ? 1 : 0}&bri=${nextBri}&temp=${nextTemp}`,
        { method: "POST" }
      );
    } catch {
      /* queued on chip if WS got there */
    }
  }
}

async function startRec(label) {
  if (state.recStarting) return;
  state.recStarting = true;
  try {
    if (state.rec) {
      await stopRec();
      if (state.rec) throw new Error("Finish saving the previous take before starting another.");
    }
    if (!state.ws || state.ws.readyState !== 1 || Date.now() - state.latestPcmAt > 2000) {
      throw new Error("Connect the ESP32 and wait for its live waveform first.");
    }
    $("take-player").pause();
    toggleListen(false, { locked: true });
    const raw = state.pcmInfo && state.pcmInfo.pcmVersion === 2 && state.pcmInfo.sampleRate === 16000;
    const capture = { pcmVersion: raw ? 2 : 1, raw: !!raw, format: "s16le", sampleRate: 16000, mode: raw ? "raw" : "legacy", deviceSampleStart: null, deviceSampleEnd: null };
    const data = await api("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label, capture }),
    });
    state.rec = {
      id: data.id, label, capture, chunks: [], samples: 0, queuedSamples: 0, pendingBytes: 0,
      uploadQueue: Promise.resolve(), uploadFailed: false, closing: false, dropStart: null,
      integrity: { uploadErrors: 0, pcmDropped: 0, discontinuities: 0, packetMetadataMissing: 0, clippedSamples: 0 },
    };
    $("session-line").textContent =
      `Recording ${label}. Speakers muted.${raw ? " Raw samples with device timing." : " Legacy audio: review only; update firmware before training."}`;
    $("mark").disabled = false;
    $("save-take").textContent = "Save take";
    $("save-take").disabled = false;
    document.querySelectorAll(".paddle").forEach((p) => {
      p.classList.toggle("is-rec", p.dataset.class === label);
      p.querySelector("[data-state]").textContent =
        p.dataset.class === label ? "recording" : "idle";
    });
  } catch (err) {
    $("session-line").textContent = "Cannot start session. " + err.message;
    toggleListen(false);
  } finally {
    state.recStarting = false;
  }
}

function stopRec() {
  const rec = state.rec;
  if (!rec) return Promise.resolve();
  if (rec.stopPromise) return rec.stopPromise;
  rec.closing = true;
  $("mark").disabled = true;
  $("save-take").disabled = true;
  $("save-take").textContent = "Saving…";
  $("session-line").textContent = "Saving the take and checking every audio upload…";
  rec.stopPromise = finishRec(rec);
  return rec.stopPromise;
}

async function finishRec(rec) {
  await flushAudio(rec);
  await rec.uploadQueue;
  let summary;
  let closeError;
  try {
    summary = await api(`/api/sessions/${rec.id}/close`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ capture: rec.capture, integrity: { ...rec.integrity, samples: rec.samples } }),
    });
  } catch (err) {
    closeError = err;
  }
  // A failed close stays attached locally so Save can retry the same take.
  if (closeError) {
    rec.stopPromise = null;
    $("save-take").disabled = false;
    $("save-take").textContent = "Retry save";
    $("session-line").textContent = "Could not finalize " + rec.id + ". Press Save to retry. " + closeError.message;
    return;
  }
  if (state.rec === rec) state.rec = null;
  $("save-take").textContent = "Take saved";
  toggleListen(false);
  document.querySelectorAll(".paddle").forEach((p) => {
    p.classList.remove("is-rec", "is-down");
    p.querySelector("[data-state]").textContent = "idle";
  });
  $("session-line").textContent = summary.trainingReady
    ? `Saved ${rec.id}. Audio integrity passed; ready for training after labels are verified.`
    : `Saved ${rec.id} for review only. ${summary.integrity?.reasons?.join(", ") || "Capture integrity was not verified"}. Record a new take for training.`;
  await refreshCounts();
}

function toggleListen(force, opts) {
  const locked = !!(opts && opts.locked) || !!state.rec;
  if (locked && force !== false) {
    state.listen = false;
  } else {
    const on = force == null ? !state.listen : !!force;
    state.listen = on && !state.rec;
  }
  $("listen-arm").setAttribute("aria-pressed", state.listen ? "true" : "false");
  $("listen-arm").disabled = locked;
  $("listen-hint").textContent = locked
    ? "muted for take"
    : state.listen
      ? "speakers live"
      : "speakers off";
  if (state.listen) {
    ensureAudio();
    if (state.audio && state.audio.state === "suspended") state.audio.resume();
  }
}

function chipTalk(msg) {
  if (state.ws && state.ws.readyState === 1) state.ws.send(msg);
}

function setGain(v) {
  const g = Number(v);
  $("gain-val").textContent = g + "×";
  if (state.gainNode) state.gainNode.gain.value = g;
  // Monitoring gain is applied only by this browser; raw recordings stay fixed.
}

function setArmMul(v) {
  const m = Number(v);
  $("arm-val").textContent = String(m);
  $("m-arm").textContent = "…";
  clearTimeout(state.armTimer);
  state.armTimer = setTimeout(() => chipTalk("arm:" + m), 80);
}

function openBay(name) {
  if (state.bay === name) {
    document.querySelectorAll(".key").forEach((key) => {
      const on = key.dataset.bay === name;
      key.classList.toggle("is-in", on);
      key.setAttribute("aria-selected", on ? "true" : "false");
    });
    return;
  }
  const prev = state.bay;
  state.bay = name;
  document.querySelectorAll(".key").forEach((key) => {
    const on = key.dataset.bay === name;
    key.classList.toggle("is-in", on);
    key.setAttribute("aria-selected", on ? "true" : "false");
  });
  const next = document.querySelector(`.bay[data-bay="${name}"]`);
  const old = document.querySelector(`.bay[data-bay="${prev}"]`);
  if (old) {
    old.classList.remove("is-open");
    old.classList.add("is-closing");
    const finish = () => {
      old.classList.remove("is-closing");
      if (state.bay !== old.dataset.bay) old.setAttribute("hidden", "");
    };
    old.addEventListener("animationend", finish, { once: true });
    window.setTimeout(finish, 320);
  }
  if (next) {
    next.removeAttribute("hidden");
    next.classList.remove("is-closing");
    next.classList.add("is-open");
  }
  if (name === "takes") refreshCounts();
}

$("connect-form").addEventListener("submit", connect);
$("listen-arm").addEventListener("click", () => {
  if (state.rec) return;
  toggleListen();
});
$("gain").addEventListener("input", (e) => setGain(e.target.value));
$("arm-mul").addEventListener("input", (e) => setArmMul(e.target.value));

document.querySelectorAll(".key").forEach((key) => {
  key.addEventListener("click", () => openBay(key.dataset.bay));
});

document.querySelectorAll(".paddle").forEach((p) => {
  p.addEventListener("click", async () => {
    const label = p.dataset.class;
    if (state.rec && state.rec.label === label) {
      p.classList.add("is-down");
      await stopRec();
      p.classList.remove("is-down");
      return;
    }
    p.classList.add("is-down");
    await startRec(label);
    p.classList.remove("is-down");
  });
});

$("save-take").addEventListener("click", () => stopRec());
$("mark").addEventListener("click", () => {
  stampRail("mark", 0, 0, Date.now());
  postEvent("mark", { src: "human", note: "fsm missed" });
  $("session-line").textContent = "Marked a miss on the current take.";
});

$("lamp-on").addEventListener("click", () => setLamp(true));
$("lamp-off").addEventListener("click", () => setLamp(false));

let lampTimer;
function lampSlide() {
  $("lamp-bri-val").textContent = Math.round(Number($("lamp-bri").value) / 10) + "%";
  $("lamp-temp-val").textContent = kelvin($("lamp-temp").value);
  clearTimeout(lampTimer);
  lampTimer = setTimeout(() => {
    setLamp(true, Number($("lamp-bri").value), Number($("lamp-temp").value));
  }, 180);
}
$("lamp-bri").addEventListener("input", lampSlide);
$("lamp-temp").addEventListener("input", lampSlide);

$("refresh-takes").addEventListener("click", refreshCounts);

$("takes").addEventListener("click", async (ev) => {
  const btn = ev.target.closest("[data-act]");
  if (!btn) return;
  const li = ev.target.closest(".take");
  if (!li) return;
  const id = li.dataset.id;
  const act = btn.dataset.act;
  if (act === "play") {
    ev.preventDefault();
    $("take-player").src = `/api/sessions/${encodeURIComponent(id)}/audio`;
    $("take-player").play().catch(() => {});
    return;
  }
  if (act === "del") {
    ev.preventDefault();
    if (state.pendingDelete !== id) {
      state.pendingDelete = id;
      btn.textContent = "Confirm";
      li.classList.add("is-armed");
      return;
    }
    try {
      await api(`/api/sessions/${encodeURIComponent(id)}`, { method: "DELETE" });
      if ($("take-player").src.includes(encodeURIComponent(id))) {
        $("take-player").pause();
        $("take-player").removeAttribute("src");
      }
      state.pendingDelete = null;
      refreshCounts();
    } catch (err) {
      $("session-line").textContent = "Delete failed. " + err.message;
    }
  }
});

document.addEventListener("keydown", (e) => {
  if (e.target.matches("input")) return;
  if (e.key === "1") $("pad-clap").click();
  if (e.key === "2") $("pad-snap").click();
  if (e.key === "3") $("pad-tongue").click();
  if (e.key === "4") $("pad-door").click();
  if (e.key === "5") $("pad-other").click();
  if (e.key === " ") {
    e.preventDefault();
    if (!state.rec) toggleListen();
  }
  if (e.key === "m" || e.key === "M") $("mark").click();
  if (e.key === "Escape" && state.rec) stopRec();
  if (e.key === "s" || e.key === "S") {
    if (state.rec) stopRec();
  }
});

drawWave();
window.addEventListener("beforeunload", (e) => {
  if (!state.rec) return;
  e.preventDefault();
  e.returnValue = "";
});
refreshCounts();
openBay("capture");
