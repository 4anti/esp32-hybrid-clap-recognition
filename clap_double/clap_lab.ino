#include <WebServer.h>
#include <ESPmDNS.h>
#include <string.h>
#include "mbedtls/sha1.h"
#include "mbedtls/base64.h"

#include "clap_pcm.h"
#include "clap_ws.h"
#define WS_MAX 2
#define EV_Q_DEPTH 24

struct LabEv {
  uint32_t ms;
  uint32_t dur;
  int32_t hp;
  uint8_t kind;
  uint64_t sample;
  uint64_t onsetSample;
  bool isAi;
  float scores[3];
  uint32_t frontendMs;
  uint32_t inferenceMs;
  bool accepted;
  const char *mode;
};

volatile uint8_t labListeners = 0;
volatile int listenGain = 4;
char labPass[33] = {0};

static ClapPcmBuffer pcmBuffer;
static portMUX_TYPE pcmMux = portMUX_INITIALIZER_UNLOCKED;

static QueueHandle_t evQ = NULL;
static WebServer labHttp(80);
static WiFiServer wsServer(81);
static WiFiServer wavServer(82);
static WiFiClient wsCli[WS_MAX];
static bool wsOn[WS_MAX];
static uint8_t wsRx[WS_MAX][128];
static size_t wsRxN[WS_MAX];
static WiFiClient wavCli;
static bool wavOn = false;
static bool mdnsUp = false;

void pcmTeeBlock(const int32_t *samples, unsigned count, uint64_t firstSample) {
  if (labListeners == 0) return;
  portENTER_CRITICAL(&pcmMux);
  pcmBuffer.push(samples, count, firstSample);
  portEXIT_CRITICAL(&pcmMux);
}

void labEventAt(uint8_t kind, uint32_t dur, int32_t hp, uint64_t sample, uint64_t onsetSample) {
  if (!evQ) return;
  LabEv e = {};
  e.ms = millis(); e.dur = dur; e.hp = hp; e.kind = kind;
  e.sample = sample; e.onsetSample = onsetSample;
  xQueueSend(evQ, &e, 0);
}

void labEvent(uint8_t kind, uint32_t dur, int32_t hp) {
  labEventAt(kind, dur, hp, audioSamples, audioSamples);
}

void labAiEvent(const ClapCandidate &candidate, const float *scores, uint32_t frontendMs,
                uint32_t inferenceMs, bool accepted, const char *mode) {
  if (!evQ) return;
  LabEv e = {};
  e.isAi = true;
  e.ms = millis();
  e.hp = candidate.peakHp;
  e.sample = absoluteSample(candidate.decisionSample);
  e.onsetSample = absoluteSample(candidate.onsetSample);
  for (int i = 0; i < 3; ++i) e.scores[i] = scores[i];
  e.frontendMs = frontendMs;
  e.inferenceMs = inferenceMs;
  e.accepted = accepted;
  e.mode = mode;
  xQueueSend(evQ, &e, 0);
}

static uint32_t pcmDropped() {
  portENTER_CRITICAL(&pcmMux);
  const uint32_t dropped = pcmBuffer.dropped() + i2sDroppedSamples;
  portEXIT_CRITICAL(&pcmMux);
  return dropped;
}

static int pcmPop(int16_t *out, int maxN, uint64_t &sample, uint32_t &dropped) {
  portENTER_CRITICAL(&pcmMux);
  const int count = pcmBuffer.pop(out, maxN, sample, dropped);
  dropped += i2sDroppedSamples;
  portEXIT_CRITICAL(&pcmMux);
  return count;
}

static unsigned pcmAvailable() {
  portENTER_CRITICAL(&pcmMux);
  const unsigned count = pcmBuffer.available();
  portEXIT_CRITICAL(&pcmMux);
  return count;
}

static void labEnsurePass() {
  String p = prefs.getString("labpw", "");
  const String preferred = CLAP_LAB_PASSWORD;
  if (preferred.length() >= 6 && preferred.length() <= 32 && p != preferred) {
    p = preferred;
    prefs.putString("labpw", p);
  }
  if (p.length() >= 6 && p.length() <= 32) {
    strncpy(labPass, p.c_str(), sizeof(labPass) - 1);
    labPass[sizeof(labPass) - 1] = 0;
    return;
  }
  const char *abc = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ";
  for (int i = 0; i < 6; i++) labPass[i] = abc[esp_random() % 32];
  labPass[6] = 0;
  prefs.putString("labpw", labPass);
}

static bool labKeyOk(const String &k) {
  return k.length() == strlen(labPass) && strcmp(k.c_str(), labPass) == 0;
}

static bool labAuthed() {
  if (labHttp.hasArg("k") && labKeyOk(labHttp.arg("k"))) return true;
  if (labHttp.hasArg("pass") && labKeyOk(labHttp.arg("pass"))) return true;
  String ck = labHttp.header("Cookie");
  int at = ck.indexOf("clap=");
  if (at >= 0) {
    const int end = ck.indexOf(';', at + 5);
    String v = end < 0 ? ck.substring(at + 5) : ck.substring(at + 5, end);
    v.trim();
    if (labKeyOk(v)) return true;
  }
  return false;
}

static void labCors() {
  labHttp.sendHeader("Access-Control-Allow-Origin", "*");
  labHttp.sendHeader("Access-Control-Allow-Headers", "Content-Type");
  labHttp.sendHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
}

static const char LOGIN_PAGE[] PROGMEM = R"HTML(<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Clap Lab</title>
<style>
html,body{margin:0;background:#141311;color:#1c1b18;font:15px/1.4 "Public Sans",ui-sans-serif,system-ui,sans-serif}
.p{max-width:22rem;margin:14vh auto;background:#c6c0b4;padding:1.6rem 1.5rem 1.4rem;box-shadow:0 18px 40px #0008,inset 0 1px 0 #efeae0}
.p h1{margin:0 0 .2rem;font:700 1.15rem/1.2 ui-sans-serif;letter-spacing:.04em}
.p p{margin:0 0 1rem;color:#3a372e}
input,button{font:inherit;width:100%;box-sizing:border-box;padding:.7rem .8rem;border:1px solid #2a2924}
input{background:#ebe4d6;margin-bottom:.7rem}
button{background:#9e1d14;color:#f4ece0;font-weight:700;border:0}
</style>
<form class="p" method="post" action="/login">
<h1>CLAP LAB</h1>
<p>Room mic. Password from Serial.</p>
<input name="pass" type="password" autocomplete="current-password" autofocus>
<button type="submit">Open listen</button>
</form>)HTML";

static const char LISTEN_PAGE[] PROGMEM = R"HTML(<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Clap Lab listen</title>
<style>
html,body{margin:0;height:100%;background:#0c0e10;color:#d7dde2;font:15px/1.35 Chivo,ui-sans-serif,system-ui,sans-serif}
.p{min-height:100%;padding:1rem 1rem 1.4rem;box-sizing:border-box;background:linear-gradient(180deg,#1c232a,#12161a 40%,#0c0e10)}
h1{margin:0;font:700 1.05rem/1.1 Chivo,sans-serif;letter-spacing:.14em;text-transform:uppercase}
.s{margin:.4rem 0 0;font:12px/1.3 ui-monospace,monospace;color:#8b98a3}
canvas{display:block;width:100%;height:26vh;margin:.7rem 0 .45rem;background:#071614;box-shadow:inset 0 0 0 1px #1a5c56,0 0 24px #0d3d3a88}
.vu{height:8px;background:#1a242c;margin:0 0 .9rem}
.vu>i{display:block;height:100%;width:0;background:#3ef0c8}
.row,.lamp{display:flex;gap:.5rem;margin:0 0 .55rem}
button{flex:1;padding:.85rem .6rem;border:0;background:#2a333c;color:#e8eef2;font:700 .78rem/1 Chivo,sans-serif;letter-spacing:.08em;text-transform:uppercase;box-shadow:inset 0 1px 0 #5a6773,0 3px 0 #0a0c0e}
button.on{background:#9e1d14;box-shadow:inset 0 2px 6px #0008,0 1px 0 #0a0c0e}
.lamp button.lit{background:#c9892e;color:#1a1208}
label{display:flex;flex-direction:column;gap:.25rem;font:11px/1 Chivo,sans-serif;letter-spacing:.1em;text-transform:uppercase;color:#8b98a3}
input[type=range]{width:100%;accent-color:#3ef0c8}
#gate{position:fixed;inset:0;background:#0c0e10f2;display:flex;align-items:center;justify-content:center;z-index:9}
#gate b{display:block;padding:1.4rem 1.6rem;background:#c9892e;color:#1a1208;letter-spacing:.12em;text-transform:uppercase}
#gate.hid{display:none}
audio{width:100%;margin:.4rem 0 .6rem}
</style>
<div id="gate"><b>Tap to hear the room</b></div>
<div class="p">
<h1>Clap Lab</h1>
<div class="s" id="st">waiting for tap</div>
<canvas id="c" width="640" height="180"></canvas>
<div class="vu"><i id="vu"></i></div>
<audio id="a" playsinline controls></audio>
<div class="row">
<button id="arm" type="button">Listen on</button>
<button id="g" type="button">Gain 4x</button>
</div>
<div class="lamp">
<button id="on" type="button">Lamp on</button>
<button id="off" type="button">Lamp off</button>
</div>
<label>Brightness <input id="bri" type="range" min="10" max="1000" value="1000"><span id="bv">100%</span></label>
<label>White temp <input id="tmp" type="range" min="0" max="1000" value="500"><span id="tv">mid</span></label>
<div class="s">Phone silent switch mutes Web Audio. Use the player above if the tap did not unlock speakers. Unmute the tab.</div>
</div>
<script>
const k=(document.cookie.match(/clap=([^;]+)/)||[])[1]||'';
const st=document.getElementById('st');
const c=document.getElementById('c');
const x=c.getContext('2d');
const vu=document.getElementById('vu');
const a=document.getElementById('a');
let on=true,gain=4,ws,peak=0,actx,gn,t0=0,useWav=false,armed=false;
const hist=new Float32Array(640);
function kelvin(t){return Math.round(2700+(t/1000)*3800)+' K'}
function unlock(){
  const C=window.AudioContext||window.webkitAudioContext;
  if(!actx){
    try{actx=new C();}catch(e){actx=new C();}
    gn=actx.createGain();gn.gain.value=gain;gn.connect(actx.destination);
    const b=actx.createBuffer(1,1,actx.sampleRate||44100);
    const s=actx.createBufferSource();s.buffer=b;s.connect(gn);try{s.start(0);}catch(e){}
  }
  if(actx.state==='suspended')actx.resume();
}
function play(s){
  if(!on||useWav||!actx||!s.length)return;
  const rate=actx.sampleRate||44100;
  const n=Math.max(1,Math.round(s.length*rate/16000));
  const f=new Float32Array(n);
  for(let i=0;i<n;i++){
    const src=i*16000/rate;
    const i0=Math.min(s.length-1,src|0);
    let v=(s[i0]/32768)*4;
    if(v>1)v=1;if(v<-1)v=-1;
    f[i]=v;
  }
  const b=actx.createBuffer(1,f.length,rate);b.copyToChannel(f,0);
  const src=actx.createBufferSource();src.buffer=b;src.connect(gn);
  const now=actx.currentTime;if(t0<now-0.05)t0=now;
  const t=Math.max(now,t0);
  try{src.start(t);t0=t+b.duration;}catch(e){}
}
function draw(){
  x.fillStyle='#071614';x.fillRect(0,0,c.width,c.height);
  x.strokeStyle='#3ef0c8';x.lineWidth=1.4;x.beginPath();
  for(let i=0;i<hist.length;i++){
    const y=90-hist[i]*80;
    if(i)x.lineTo(i,y);else x.moveTo(i,y);
  }
  x.stroke();
}
function go(){
  ws=new WebSocket('ws://'+location.hostname+':81/?k='+encodeURIComponent(k));
  ws.binaryType='arraybuffer';
  ws.onopen=()=>{if(!useWav)st.textContent='live  '+location.host};
  ws.onclose=()=>{st.textContent='socket closed';setTimeout(go,1200)};
  ws.onmessage=ev=>{
    if(typeof ev.data==='string'){
      try{
        const j=JSON.parse(ev.data);
        if(j.t==='stat'){
          st.textContent='heap '+j.heap+'  tuya '+(j.tuya?'ok':'down')+'  lamp '+(j.lamp?'ON':'OFF');
          document.getElementById('on').classList.toggle('lit',!!j.lamp);
          if(j.bri){document.getElementById('bri').value=j.bri;document.getElementById('bv').textContent=Math.round(j.bri/10)+'%';}
          if(j.temp!=null){document.getElementById('tmp').value=j.temp;document.getElementById('tv').textContent=kelvin(j.temp);}
        }
      }catch(e){}
      return;
    }
    const s=new Int16Array(ev.data);
    let p=0;
    for(let i=0;i<s.length;i++){const a=Math.abs(s[i]);if(a>p)p=a;}
    peak=Math.max(p/32768,peak*0.86);
    vu.style.width=(peak*100)+'%';
    hist.copyWithin(0,8);
    for(let i=0;i<8;i++)hist[hist.length-8+i]=s[Math.floor(i*s.length/8)]/32768;
    draw();
    play(s);
  };
}
function lamp(on,bri,temp){
  const q='k='+encodeURIComponent(k)+'&on='+(on?1:0)+'&bri='+bri+'&temp='+temp;
  fetch('/api/lamp?'+q,{method:'POST'}).catch(()=>{});
}
function armHear(){
  if(armed)return;
  armed=true;
  document.getElementById('gate').classList.add('hid');
  unlock();
  a.src='http://'+location.hostname+':82/?k='+encodeURIComponent(k);
  a.play().then(()=>{useWav=true;st.textContent='speakers live (phone stream)';}).catch(()=>{useWav=false;st.textContent='speakers via tap (web audio)';});
  go();
}
document.getElementById('gate').onclick=armHear;
document.getElementById('arm').onclick=e=>{
  armHear();
  on=!on;e.target.classList.toggle('on',on);e.target.textContent=on?'Listen on':'Listen off';
  if(!on){try{a.pause();}catch(e){}}
  else if(useWav){a.play().catch(()=>{});}
};
document.getElementById('g').onclick=e=>{
  armHear();
  gain=gain>=8?1:gain*2;e.target.textContent='Gain '+gain+'x';
  if(gn)gn.gain.value=gain;
  if(ws&&ws.readyState===1)ws.send('gain:'+gain);
};
document.getElementById('arm').classList.add('on');
document.getElementById('on').onclick=()=>{armHear();lamp(true,+document.getElementById('bri').value,+document.getElementById('tmp').value);};
document.getElementById('off').onclick=()=>{armHear();lamp(false,+document.getElementById('bri').value,+document.getElementById('tmp').value);};
let tmr;
function slide(){
  clearTimeout(tmr);
  const bri=+document.getElementById('bri').value,temp=+document.getElementById('tmp').value;
  document.getElementById('bv').textContent=Math.round(bri/10)+'%';
  document.getElementById('tv').textContent=kelvin(temp);
  tmr=setTimeout(()=>lamp(true,bri,temp),180);
}
document.getElementById('bri').oninput=slide;
document.getElementById('tmp').oninput=slide;
</script>)HTML";

static void handleRoot() {
  labCors();
  if (!labAuthed()) {
    labHttp.send_P(200, "text/html", LOGIN_PAGE);
    return;
  }
  labHttp.send_P(200, "text/html", LISTEN_PAGE);
}

static void handleLogin() {
  labCors();
  if (!labKeyOk(labHttp.arg("pass")) && !labKeyOk(labHttp.arg("k"))) {
    labHttp.send(401, "text/plain", "bad pass");
    return;
  }
  labHttp.sendHeader("Set-Cookie", String("clap=") + labPass + "; Path=/; SameSite=Lax");
  labHttp.sendHeader("Location", "/");
  labHttp.send(302, "text/plain", "ok");
}

static void handleStatus() {
  labCors();
  if (!labAuthed()) {
    labHttp.send(401, "application/json", "{\"err\":\"auth\"}");
    return;
  }
  char js[620];
  snprintf(js, sizeof(js),
           "{\"wifi\":%d,\"tuya\":%d,\"i2s\":%d,\"heap\":%u,\"minHeap\":%u,"
           "\"lamp\":%d,\"known\":%d,\"bri\":%d,\"temp\":%d,\"colour\":%d,"
           "\"arm\":%ld,\"armMul\":%d,\"gain\":%d,\"listeners\":%u,\"ip\":\"%s\",\"pcmVersion\":2,\"sampleRate\":16000,\"pcmDropped\":%lu,\"commandDrops\":%lu,\"aiMode\":\"%s\",\"aiDropped\":%lu,\"aiErrors\":%lu,\"inferenceMs\":%lu}",
           (int)WiFi.status(), tuyaLive ? 1 : 0, i2sOk ? 1 : 0,
           (unsigned)ESP.getFreeHeap(), (unsigned)ESP.getMinFreeHeap(),
           lampOn ? 1 : 0, lampKnown ? 1 : 0, lampBri, lampTemp, lampHasColour ? 1 : 0,
           (long)armLevel, (int)armMul,
           (int)listenGain, (unsigned)labListeners,
           WiFi.localIP().toString().c_str(), (unsigned long)pcmDropped(), (unsigned long)commandDrops, aiMode(), (unsigned long)aiDropped(), (unsigned long)aiErrors(), (unsigned long)aiLastInferenceMs());
  labHttp.send(200, "application/json", js);
}

static void handleArm() {
  labCors();
  if (!labAuthed()) {
    labHttp.send(401, "text/plain", "auth");
    return;
  }
  if (labHttp.hasArg("mul")) {
    armMul = labHttp.arg("mul").toInt();
    applyArm();
  }
  char js[80];
  snprintf(js, sizeof(js), "{\"arm\":%ld,\"armMul\":%d}", (long)armLevel, (int)armMul);
  labHttp.send(200, "application/json", js);
}

static void handleGain() {
  labCors();
  if (!labAuthed()) {
    labHttp.send(401, "text/plain", "auth");
    return;
  }
  if (labHttp.hasArg("g")) {
    int g = labHttp.arg("g").toInt();
    if (g < 1) g = 1;
    if (g > 8) g = 8;
    listenGain = g;
  }
  char js[40];
  snprintf(js, sizeof(js), "{\"gain\":%d}", (int)listenGain);
  labHttp.send(200, "application/json", js);
}

static void handleLamp() {
  labCors();
  if (!labAuthed()) {
    labHttp.send(401, "text/plain", "auth");
    return;
  }
  bool on = lampOn;
  if (labHttp.hasArg("on")) on = labHttp.arg("on").toInt() != 0;
  int bri = labHttp.hasArg("bri") ? labHttp.arg("bri").toInt() : lampBri;
  int tmp = labHttp.hasArg("temp") ? labHttp.arg("temp").toInt() : lampTemp;
  labRequestLamp(on, bri, tmp);
  char js[120];
  snprintf(js, sizeof(js), "{\"ok\":1,\"lamp\":%d,\"bri\":%d,\"temp\":%d}", on ? 1 : 0, bri, tmp);
  labHttp.send(200, "application/json", js);
}

static void handleOpt() {
  labCors();
  labHttp.send(204);
}

static bool wsAcceptKey(const char *key, char *out, size_t outMax) {
  char concat[80];
  snprintf(concat, sizeof(concat), "%s258EAFA5-E914-47DA-95CA-C5AB0DC85B11", key);
  uint8_t sha[20];
  if (mbedtls_sha1((const unsigned char *)concat, strlen(concat), sha) != 0) return false;
  size_t olen = 0;
  if (mbedtls_base64_encode((unsigned char *)out, outMax, &olen, sha, 20) != 0) return false;
  out[olen] = 0;
  return true;
}

static int wsSlot() {
  for (int i = 0; i < WS_MAX; i++) {
    if (!wsOn[i]) return i;
  }
  return -1;
}

static void wsDrop(int i) {
  if (!wsOn[i]) return;
  wsCli[i].stop();
  wsOn[i] = false;
  wsRxN[i] = 0;
  if (labListeners) labListeners--;
}

static bool labWrite(WiFiClient &client, const uint8_t *p, size_t n) {
  size_t off = 0;
  unsigned long t0 = millis();
  while (off < n) {
    if (millis() - t0 >= 8) return false;
    if (!client.connected() || client.fd() < 0) return false;
    // NetworkClient::write can block inside its own retry loop, outlasting an
    // outer timeout. A slow viewer must never stall every other audio viewer.
    int w = send(client.fd(), p + off, n - off, MSG_DONTWAIT);
    if (w > 0) {
      off += (size_t)w;
      continue;
    }
    if (w == 0 || (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)) return false;
    if (millis() - t0 >= 8) return false;
    vTaskDelay(1);
  }
  return true;
}

static bool wsSend(int i, uint8_t opcode, const uint8_t *data, size_t n) {
  if (!wsOn[i] || !wsCli[i].connected()) {
    wsDrop(i);
    return false;
  }
  uint8_t frame[512]; // Status/event/pong payloads are bounded below 508 bytes.
  const size_t size = clapWsFrame(frame, sizeof(frame), opcode, data, n);
  if (!size || !labWrite(wsCli[i], frame, size)) {
    wsDrop(i);
    return false;
  }
  return true;
}

static void labAcceptWs() {
  WiFiClient c = wsServer.available();
  if (!c) return;
  const uint32_t started = millis();
  char httpLine[513]; // At most 512 bytes per line, including an optional CR.
  auto readLine = [&]() {
    return clapWsReadHttpLine(c, httpLine, sizeof(httpLine), started, 400,
        []() { return (uint32_t)millis(); }, []() { vTaskDelay(1); });
  };
  if (!readLine()) { c.stop(); return; }
  String req(httpLine);
  String key;
  String qk;
  int q = req.indexOf("k=");
  if (q > 0) {
    qk = req.substring(q + 2);
    int end = qk.indexOf(' ');
    if (end < 0) end = qk.indexOf('&');
    if (end >= 0) qk = qk.substring(0, end);
    qk.trim();
  }
  for (;;) {
    if (!readLine()) { c.stop(); return; }
    String line(httpLine);
    if (line.length() == 0) break;
    int col = line.indexOf(':');
    if (line.startsWith("Sec-WebSocket-Key") && col > 0) {
      key = line.substring(col + 1);
      key.trim();
    }
  }
  int slot = wsSlot();
  if (!labKeyOk(qk) || key.length() < 10) {
    c.print("HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n");
    c.stop();
    return;
  }
  if (slot < 0) {
    c.print("HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\nRetry-After: 2\r\n\r\n");
    c.stop();
    return;
  }
  char accept[40];
  if (!wsAcceptKey(key.c_str(), accept, sizeof(accept))) {
    c.stop();
    return;
  }
  c.print("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ");
  c.print(accept);
  c.print("\r\n\r\n");
  c.setNoDelay(true);
  c.setTimeout(20);
  wsCli[slot] = c;
  wsOn[slot] = true;
  wsRxN[slot] = 0;
  labListeners++;
}

static void labApplyWsText(const char *payload, size_t n) {
  if (n > 5 && strncmp(payload, "gain:", 5) == 0) {
    int g = atoi(payload + 5);
    if (g < 1) g = 1;
    if (g > 8) g = 8;
    listenGain = g;
  }
  if (n > 4 && strncmp(payload, "arm:", 4) == 0) {
    armMul = atoi(payload + 4);
    applyArm();
  }
  if (n > 5 && strncmp(payload, "lamp:", 5) == 0) {
    labRequestLamp(atoi(payload + 5) != 0, lampBri, lampTemp);
  }
  if (n > 4 && strncmp(payload, "bri:", 4) == 0) {
    labRequestLamp(true, atoi(payload + 4), lampTemp);
  }
  if (n > 5 && strncmp(payload, "temp:", 5) == 0) {
    labRequestLamp(true, lampBri, atoi(payload + 5));
  }
}

static void labReadWs(int i) {
  if (!wsOn[i] || !wsCli[i].connected()) {
    wsDrop(i);
    return;
  }
  while (wsCli[i].available() && wsRxN[i] < sizeof(wsRx[0])) {
    wsRx[i][wsRxN[i]++] = (uint8_t)wsCli[i].read();
  }
  size_t off = 0;
  while (wsRxN[i] - off >= 2) {
    uint8_t h0 = wsRx[i][off];
    uint8_t h1 = wsRx[i][off + 1];
    uint8_t op = h0 & 0x0F;
    bool masked = h1 & 0x80;
    size_t n = h1 & 0x7F;
    size_t hdr = 2;
    if (n == 126) {
      if (wsRxN[i] - off < 4) break;
      n = ((size_t)wsRx[i][off + 2] << 8) | (size_t)wsRx[i][off + 3];
      hdr = 4;
    } else if (n == 127) {
      wsDrop(i);
      return;
    }
    if (masked) hdr += 4;
    if (n > 80) {
      wsDrop(i);
      return;
    }
    if (wsRxN[i] - off < hdr + n) break;
    uint8_t mask[4] = {0, 0, 0, 0};
    if (masked) memcpy(mask, wsRx[i] + off + hdr - 4, 4);
    char payload[81];
    const uint8_t *body = wsRx[i] + off + hdr;
    for (size_t k = 0; k < n; k++) {
      payload[k] = (char)(masked ? (body[k] ^ mask[k & 3]) : body[k]);
    }
    payload[n] = 0;
    off += hdr + n;
    if (op == 0x8) {
      memmove(wsRx[i], wsRx[i] + off, wsRxN[i] - off);
      wsRxN[i] -= off;
      wsDrop(i);
      return;
    }
    if (op == 0x9) {
      // A failed send clears the receive count; do not subtract the old offset
      // afterward or size_t underflow can turn cleanup into a huge memmove.
      if (!wsSend(i, 0xA, (const uint8_t *)payload, n)) return;
      continue;
    }
    if (op == 0x1) labApplyWsText(payload, n);
  }
  if (off) {
    memmove(wsRx[i], wsRx[i] + off, wsRxN[i] - off);
    wsRxN[i] -= off;
  }
  if (wsRxN[i] == sizeof(wsRx[0])) wsDrop(i);
}

static const char *evName(uint8_t k) {
  switch (k) {
    case EV_HIT: return "hit";
    case EV_IGNORE: return "ignore";
    case EV_BUSY: return "busy";
    case EV_SINGLE: return "single";
    case EV_DOUBLE: return "double";
    case EV_SLAM: return "slam";
    default: return "dull";
  }
}

static void labPumpEvents() {
  if (!evQ) return;
  LabEv e;
  while (xQueueReceive(evQ, &e, 0) == pdTRUE) {
    char js[440];
    if (e.isAi) snprintf(js, sizeof(js),
             "{\"t\":\"ai\",\"ms\":%lu,\"sample\":%llu,\"onsetSample\":%llu,\"sampleRate\":16000,"
             "\"scores\":[%.5f,%.5f,%.5f],\"frontendMs\":%lu,\"inferenceMs\":%lu,\"accepted\":%s,\"mode\":\"%s\"}",
             (unsigned long)e.ms, (unsigned long long)e.sample, (unsigned long long)e.onsetSample,
             e.scores[0], e.scores[1], e.scores[2], (unsigned long)e.frontendMs,
             (unsigned long)e.inferenceMs, e.accepted ? "true" : "false", e.mode);
    else snprintf(js, sizeof(js),
             "{\"t\":\"ev\",\"kind\":\"%s\",\"ms\":%lu,\"dur\":%lu,\"hp\":%ld,"
             "\"sample\":%llu,\"onsetSample\":%llu,\"sampleRate\":16000}",
             evName(e.kind), (unsigned long)e.ms, (unsigned long)e.dur, (long)e.hp,
             (unsigned long long)e.sample, (unsigned long long)e.onsetSample);
    size_t n = strlen(js);
    for (int i = 0; i < WS_MAX; i++) {
      if (wsOn[i]) wsSend(i, 0x1, (const uint8_t *)js, n);
    }
  }
}

static void wavDrop() {
  if (!wavOn) return;
  wavCli.stop();
  wavOn = false;
  if (labListeners) labListeners--;
}

static void wavPutU32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
  p[3] = (uint8_t)(v >> 24);
}

static void wavAccept() {
  WiFiClient c = wavServer.available();
  if (!c) return;
  const uint32_t started = millis();
  char httpLine[513];
  auto readLine = [&]() {
    return clapWsReadHttpLine(c, httpLine, sizeof(httpLine), started, 400,
        []() { return (uint32_t)millis(); }, []() { vTaskDelay(1); });
  };
  if (!readLine()) { c.stop(); return; }
  String req(httpLine);
  String qk;
  int q = req.indexOf("k=");
  if (q > 0) {
    qk = req.substring(q + 2);
    int end = qk.indexOf(' ');
    if (end < 0) end = qk.indexOf('&');
    if (end >= 0) qk = qk.substring(0, end);
    qk.trim();
  }
  for (;;) {
    if (!readLine()) { c.stop(); return; }
    if (httpLine[0] == 0) break;
  }
  if (!labKeyOk(qk)) {
    c.print("HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n");
    c.stop();
    return;
  }
  if (wavOn) wavDrop();
  const uint32_t dataBytes = 16000UL * 2UL * 3600UL;
  uint32_t riff = 36 + dataBytes;
  uint8_t h[44];
  memcpy(h, "RIFF", 4);
  wavPutU32(h + 4, riff);
  memcpy(h + 8, "WAVEfmt ", 8);
  h[16] = 16; h[17] = 0; h[18] = 0; h[19] = 0;
  h[20] = 1; h[21] = 0; h[22] = 1; h[23] = 0;
  wavPutU32(h + 24, 16000);
  wavPutU32(h + 28, 32000);
  h[32] = 2; h[33] = 0; h[34] = 16; h[35] = 0;
  memcpy(h + 36, "data", 4);
  wavPutU32(h + 40, dataBytes);
  c.print("HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\nContent-Length: ");
  c.print(44UL + dataBytes);
  c.print("\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n");
  if (c.write(h, 44) != 44) {
    c.stop();
    return;
  }
  c.setNoDelay(true);
  wavCli = c;
  wavOn = true;
  labListeners++;
}

static void labPumpPcm() {
  if (labListeners == 0) return;
  // Batch 32 ms of raw audio. Sending every tiny available block generated too
  // many Wi-Fi writes with two viewers, overflowing the recording FIFO.
  if (pcmAvailable() < 512) return;
  int16_t pkt[512];
  uint64_t sample;
  uint32_t dropped;
  int n = pcmPop(pkt, 512, sample, dropped);
  if (n <= 0) return;
  char header[180];
  snprintf(header, sizeof(header),
           "{\"t\":\"pcm\",\"sample\":%llu,\"samples\":%d,\"sampleRate\":16000,"
           "\"pcmVersion\":2,\"pcmDropped\":%lu}",
           (unsigned long long)sample, n, (unsigned long)dropped);
  uint8_t frames[sizeof(pkt) + sizeof(header) + 8];
  const size_t textBytes = clapWsFrame(frames, sizeof(frames), 0x1,
      (const uint8_t *)header, strlen(header));
  const size_t audioBytes = textBytes ? clapWsFrame(frames + textBytes,
      sizeof(frames) - textBytes, 0x2, (const uint8_t *)pkt, (size_t)n * 2) : 0;
  for (int i = 0; i < WS_MAX; i++)
    if (wsOn[i] && (!textBytes || !audioBytes ||
        !labWrite(wsCli[i], frames, textBytes + audioBytes))) wsDrop(i);
  if (wavOn) {
    // Playback gain is applied only to the phone WAV path. Training PCM stays raw.
    const int gain = listenGain;
    for (int i = 0; i < n; ++i) {
      int32_t value = (int32_t)pkt[i] * gain;
      if (value > 32767) value = 32767;
      if (value < -32768) value = -32768;
      pkt[i] = (int16_t)value;
    }
    if (!labWrite(wavCli, (const uint8_t *)pkt, (size_t)n * 2))
      wavDrop();
  }
}

static void labPumpStat() {
  static unsigned long last = 0;
  if (millis() - last < 1000) return;
  last = millis();
  if (labListeners == 0) return;
  char js[480];
  snprintf(js, sizeof(js),
           "{\"t\":\"stat\",\"heap\":%u,\"tuya\":%d,\"i2s\":%d,\"lamp\":%d,\"bri\":%d,\"temp\":%d,\"arm\":%ld,\"listeners\":%u,\"pcmVersion\":2,\"sampleRate\":16000,\"pcmDropped\":%lu,\"commandDrops\":%lu,\"aiMode\":\"%s\",\"aiDropped\":%lu,\"aiErrors\":%lu,\"inferenceMs\":%lu}",
           (unsigned)ESP.getFreeHeap(), tuyaLive ? 1 : 0, i2sOk ? 1 : 0,
           lampOn ? 1 : 0, lampBri, lampTemp, (long)armLevel, (unsigned)labListeners, (unsigned long)pcmDropped(), (unsigned long)commandDrops, aiMode(), (unsigned long)aiDropped(), (unsigned long)aiErrors(), (unsigned long)aiLastInferenceMs());
  size_t n = strlen(js);
  for (int i = 0; i < WS_MAX; i++) {
    if (wsOn[i]) wsSend(i, 0x1, (const uint8_t *)js, n);
  }
}

static void labServiceWs() {
  for (int i = 0; i < WS_MAX; i++) {
    if (wsOn[i]) labReadWs(i);
  }
}

void labOnWifi() {
  if (!mdnsUp) mdnsUp = MDNS.begin("clap");
  elog("lab http://%s/  pass=%s", WiFi.localIP().toString().c_str(), labPass);
}

static void streamTask(void *arg) {
  (void)arg;
  esp_task_wdt_add(NULL);
  elog("lab wdt ok");
  const char *hdrs[] = {"Cookie"};
  labHttp.collectHeaders(hdrs, 1);
  labHttp.on("/", handleRoot);
  labHttp.on("/login", HTTP_POST, handleLogin);
  labHttp.on("/api/status", handleStatus);
  labHttp.on("/api/arm", HTTP_POST, handleArm);
  labHttp.on("/api/gain", HTTP_POST, handleGain);
  labHttp.on("/api/lamp", HTTP_POST, handleLamp);
  labHttp.on("/api/status", HTTP_OPTIONS, handleOpt);
  labHttp.on("/api/arm", HTTP_OPTIONS, handleOpt);
  labHttp.on("/api/gain", HTTP_OPTIONS, handleOpt);
  labHttp.on("/api/lamp", HTTP_OPTIONS, handleOpt);
  while (WiFi.getMode() == WIFI_OFF) {
    wdtFeed();
    vTaskDelay(pdMS_TO_TICKS(20));
  }
  labHttp.begin();
  wsServer.begin();
  wsServer.setNoDelay(true);
  wavServer.begin();
  wavServer.setNoDelay(true);
  for (;;) {
    wdtFeed();
    labHttp.handleClient();
    labAcceptWs();
    wavAccept();
    if (wavOn && !wavCli.connected()) wavDrop();
    labServiceWs();
    labPumpPcm();
    labPumpEvents();
    labPumpStat();
    vTaskDelay(pdMS_TO_TICKS(2));
  }
}

bool labSetup() {
  labEnsurePass();
  evQ = xQueueCreate(EV_Q_DEPTH, sizeof(LabEv));
  if (!evQ) return false;
  elog("lab pass=%s  (phone: http://<esp-ip>/ )", labPass);
  // Higher than netTask so a Tuya handshake cannot starve the lab socket.
  return xTaskCreatePinnedToCore(streamTask, "lab", 12288, NULL, 2, NULL, 0) == pdPASS;
}
