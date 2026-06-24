import os
import ssl
import time
import json
import subprocess
import signal
import sys
import threading
import requests as req
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

HOME            = os.path.expanduser("~")
APP_PORT        = 8501
LLAMA_PORT      = 8080
LLAMA_BIN       = f"{HOME}/llama.cpp/build/bin/llama-server"
LLAMA_URL       = f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions"
CERT_FILE       = f"{HOME}/cert.pem"
KEY_FILE        = f"{HOME}/key.pem"

IMAGE_MAX_TOKENS = 16

MODELS = {
    "smolvlm-256m-q4": {
        "label": "SmolVLM 256M Q4 (ULTRA FAST)",
        "model":   f"{HOME}/models/smolvlm/SmolVLM-256M-Instruct-Q4_K_M.gguf",
        "mmproj":  f"{HOME}/models/smolvlm/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf",
        "ctx": 1024,
        "threads": 4,
    },
    "smolvlm-256m": {
        "label": "SmolVLM 256M Q8",
        "model":   f"{HOME}/models/smolvlm/SmolVLM-256M-Instruct-Q8_0.gguf",
        "mmproj":  f"{HOME}/models/smolvlm/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf",
        "ctx": 1024,
        "threads": 4,
    },
    "smolvlm-500m": {
        "label": "SmolVLM 500M Q8 (BETTER)",
        "model":   f"{HOME}/models/smolvlm/SmolVLM-500M-Instruct-Q8_0.gguf",
        "mmproj":  f"{HOME}/models/smolvlm/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf",
        "ctx": 1024,
        "threads": 4,
    },
}

llama_process   = None
last_encode_ms  = 0
last_eval_ms    = 0


def set_cpu_performance():
    """Lock CPU to max frequency — biggest single speedup available."""
    try:
        result = subprocess.run(
            ["sudo", "sh", "-c",
             "echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print("[PERF] CPU governor set to PERFORMANCE (max freq locked)", flush=True)
            return True
        else:
            print(f"[PERF] Could not set governor (need sudo): {result.stderr.strip()}", flush=True)
            return False
    except Exception as e:
        print(f"[PERF] Governor set failed: {e}", flush=True)
        return False


def get_cpu_freq():
    """Get current CPU frequency in MHz."""
    try:
        r = subprocess.check_output(
            ["cat", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"],
            text=True, timeout=2
        ).strip()
        return f"{int(r)//1000}MHz"
    except:
        try:
            r = subprocess.check_output(
                ["vcgencmd", "measure_clock", "arm"],
                text=True, timeout=2
            ).strip()
            hz = int(r.split("=")[1])
            return f"{hz//1_000_000}MHz"
        except:
            return "?"


def stop_llama():
    global llama_process
    if llama_process and llama_process.poll() is None:
        try:
            os.killpg(os.getpgid(llama_process.pid), signal.SIGTERM)
            llama_process.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(llama_process.pid), signal.SIGKILL)
            except Exception:
                pass
    llama_process = None
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(1)


def check_throttle():
    try:
        r = subprocess.check_output(["vcgencmd", "get_throttled"],
                                     text=True, timeout=2).strip()
        val = int(r.split("=")[1], 16)
        return val != 0, hex(val)
    except:
        return False, "unknown"


def get_temp():
    try:
        r = subprocess.check_output(["vcgencmd", "measure_temp"],
                                     text=True, timeout=2).strip()
        return r.split("=")[1]
    except:
        return "?"


def start_llama(model_key):
    global llama_process, last_encode_ms, last_eval_ms
    stop_llama()
    last_encode_ms = 0
    last_eval_ms   = 0

    m = MODELS[model_key]
    if not os.path.isfile(m["model"]):
        return False, f"Model file not found: {m['model']}"
    if not os.path.isfile(m["mmproj"]):
        return False, f"mmproj file not found: {m['mmproj']}"

    throttled, tval = check_throttle()
    temp = get_temp()
    freq = get_cpu_freq()
    if throttled:
        print(f"[WARN] THROTTLED ({tval}) temp={temp} freq={freq} — add cooling!", flush=True)
    else:
        print(f"[INFO] Thermal OK ({tval}) temp={temp} freq={freq}", flush=True)

    cmd = [
        LLAMA_BIN,
        "-m",        m["model"],
        "--mmproj",  m["mmproj"],
        "-c",        str(m["ctx"]),
        "--threads", str(m["threads"]),
        "--host",    "0.0.0.0",
        "--port",    str(LLAMA_PORT),
        "--no-jinja",
        "--no-webui",
        "--no-warmup",
        "--mlock",
        "--no-mmap",
        "--flash-attn", "on",
        "--image-max-tokens", str(IMAGE_MAX_TOKENS),
        "-b",  "512",
        "-ub", "512",
        "-ngl", "0",
    ]

    print(f"\n[SERVER] {' '.join(cmd)}\n", flush=True)

    env = os.environ.copy()
    env["GGML_VK_VISIBLE_DEVICES"] = ""      # hide GPU entirely
    env["OMP_NUM_THREADS"] = str(m["threads"])
    env["OMP_PROC_BIND"] = "close"            # bind threads to nearby cores
    env["GOMP_SPINCOUNT"] = "0"               # don't spin-wait (saves power/heat)

    llama_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        text=True,
        bufsize=1,
        env=env,
    )

    def log_filter():
        global last_encode_ms, last_eval_ms
        for line in llama_process.stdout:
            line = line.rstrip()
            lo = line.lower()
            if any(k in lo for k in ("system_info", "encoded in", "image slice",
                                      "eval time", "prompt eval", "error",
                                      "warning", "ready", "crash")):
                print(f"  [llama] {line}", flush=True)
            if "encoded in" in lo:
                try:
                    last_encode_ms = float(line.split("encoded in")[1].strip().split()[0])
                except:
                    pass
            if "eval time" in lo and " = " in lo and "prompt" not in lo:
                try:
                    last_eval_ms = float(line.split("=")[1].strip().split()[0])
                except:
                    pass
    threading.Thread(target=log_filter, daemon=True).start()

    for _ in range(45):
        try:
            r = req.get(f"http://127.0.0.1:{LLAMA_PORT}/health", timeout=1)
            if r.status_code < 500:
                freq = get_cpu_freq()
                print(f"[SERVER] Ready: {m['label']} | CPU={freq}", flush=True)
                return True, "OK"
        except:
            pass
        time.sleep(1)
        if llama_process.poll() is not None:
            return False, "llama-server crashed on startup"

    return True, "Started (health pending)"


def analyze_image(b64):
    payload = {
        "temperature": 0.0,
        "max_tokens": 12,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": "Describe briefly:"},
            ],
        }],
    }
    try:
        r = req.post(LLAMA_URL, json=payload, timeout=30)
        if r.status_code != 200:
            return f"LLM error {r.status_code}"
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {e}"


def get_html():
    model_options = ""
    for key, m in MODELS.items():
        exists = "OK" if os.path.isfile(m["model"]) and os.path.isfile(m["mmproj"]) else "MISSING"
        model_options += f'<option value="{key}">[{exists}] {m["label"]}</option>\n'

    token_options = ""
    labels = {12:"12 (~1.5s)", 16:"16 (~2s)", 24:"24 (~2.5s)", 32:"32 (~3s)", 48:"48 (~3.5s)"}
    for t, lbl in labels.items():
        sel = "selected" if t == IMAGE_MAX_TOKENS else ""
        token_options += f'<option value="{t}" {sel}>{lbl}</option>\n'

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>Pi 5 Live Vision</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,sans-serif;background:#0e1117;color:#e0e0e0;
      display:flex;flex-direction:column;min-height:100vh;overflow-x:hidden}}
.header{{background:#1a1d23;padding:12px 16px;text-align:center;border-bottom:1px solid #333}}
.header h1{{font-size:1.3em;margin-bottom:2px}}
.header small{{color:#888;font-size:0.78em}}
.controls{{background:#161922;padding:10px 16px;display:flex;flex-wrap:wrap;gap:8px;
           align-items:center;justify-content:center;border-bottom:1px solid #333}}
select,button{{font-size:0.85em;padding:7px 12px;border-radius:8px;border:1px solid #444;
               background:#222;color:#e0e0e0;cursor:pointer}}
button{{font-weight:bold}}
button:active{{transform:scale(0.97)}}
.btn-start{{background:#1b8a2e;border-color:#1b8a2e;color:#fff}}
.btn-stop{{background:#c0392b;border-color:#c0392b;color:#fff}}
.btn-load{{background:#2471a3;border-color:#2471a3;color:#fff}}
.btn-perf{{background:#7d3c98;border-color:#7d3c98;color:#fff}}
.row{{background:#161922;padding:6px 16px 8px;display:flex;align-items:center;
      justify-content:center;gap:12px;border-bottom:1px solid #333;flex-wrap:wrap}}
.row label{{font-size:0.82em;color:#aaa}}
.row input[type=range]{{width:150px;accent-color:#2471a3}}
.row .val{{font-size:0.9em;font-weight:bold;color:#2ecc71;min-width:28px;text-align:center}}
.video-wrap{{position:relative;width:100%;max-width:600px;margin:8px auto;
             background:#000;border-radius:10px;overflow:hidden}}
video{{width:100%;display:block;border-radius:10px}}
canvas{{display:none}}
.caption-box{{background:#1a2332;margin:6px 14px;padding:12px 16px;border-radius:10px;
              font-size:1.1em;text-align:center;min-height:48px;border:1px solid #2471a3}}
.timing{{background:#0d1520;margin:0 14px 4px;padding:5px 12px;border-radius:7px;
         font-size:0.76em;font-family:monospace;color:#555;display:flex;
         gap:14px;justify-content:center;flex-wrap:wrap}}
.timing b{{color:#2471a3}}
.status{{text-align:center;padding:5px;font-size:0.8em;color:#888}}
.status .dot{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}}
.dot-green{{background:#2ecc71}}.dot-red{{background:#e74c3c}}
.dot-yellow{{background:#f39c12}}.dot-orange{{background:#e67e22}}
.log-wrap{{margin:6px 14px 14px;flex:1}}
.log-title{{font-size:0.82em;color:#888;margin-bottom:3px}}
.log{{background:#0a0c10;border:1px solid #2a2a2a;border-radius:8px;padding:7px 10px;
      max-height:170px;overflow-y:auto;font-family:monospace;font-size:0.73em;line-height:1.55}}
.log div{{border-bottom:1px solid #141420;padding:2px 0}}
.log div:first-child{{color:#2ecc71}}
</style>
</head>
<body>

<div class="header">
  <h1>Pi 5 Live Vision</h1>
  <small>SmolVLM · CPU · DOTPROD optimized</small>
</div>

<div class="controls">
  <select id="modelSelect">{model_options}</select>
  <button class="btn-load" onclick="loadModel()">Load Model</button>
  <select id="camSelect">
    <option value="environment">Back Camera</option>
    <option value="user">Front Camera</option>
  </select>
  <button class="btn-start" onclick="startLoop()">Start</button>
  <button class="btn-stop"  onclick="stopLoop()">Stop</button>
  <button class="btn-perf"  onclick="setPerfMode()">⚡ Perf Mode</button>
</div>

<div class="row">
  <label>Every:</label>
  <input type="range" id="intervalSlider" min="1" max="10" step="1" value="3">
  <span class="val" id="intervalLabel">3s</span>
  <label style="margin-left:8px">Img tokens:</label>
  <select id="tokenSelect" onchange="reloadNeeded()">{token_options}</select>
</div>

<div class="status" id="statusBar">
  <span class="dot dot-yellow"></span>Ready — Load model then Start.
</div>

<div class="video-wrap">
  <video id="video" autoplay playsinline muted></video>
</div>
<canvas id="canvas"></canvas>

<div class="caption-box" id="captionBox">Press Start to begin...</div>

<div class="timing">
  total: <b id="tTotal">—</b> &nbsp;|&nbsp;
  server: <b id="tServer">—</b> &nbsp;|&nbsp;
  avg: <b id="tAvg">—</b> &nbsp;|&nbsp;
  frames: <b id="tCount">0</b>
</div>

<div class="log-wrap">
  <div class="log-title">Log (latest on top)</div>
  <div class="log" id="logBox"></div>
</div>

<script>
const video    = document.getElementById('video');
const canvas   = document.getElementById('canvas');
const ctx      = canvas.getContext('2d');
const capBox   = document.getElementById('captionBox');
const logBox   = document.getElementById('logBox');
const statusEl = document.getElementById('statusBar');
const slider   = document.getElementById('intervalSlider');
const sliderLbl= document.getElementById('intervalLabel');

let running = false, busy = false, timerId = null, stream = null;
let count = 0, totalTime = 0;

function getIntervalMs() {{ return parseInt(slider.value) * 1000; }}

slider.addEventListener('input', () => {{
  sliderLbl.textContent = slider.value + 's';
  if (running) {{ clearInterval(timerId); timerId = setInterval(analyzeOnce, getIntervalMs()); }}
}});

function reloadNeeded() {{
  setStatus('orange', 'Token count changed — reload model to apply!');
}}

function setStatus(color, text) {{
  statusEl.innerHTML = '<span class="dot dot-' + color + '"></span>' + text;
}}

async function setPerfMode() {{
  setStatus('yellow', 'Setting CPU to performance mode...');
  const r = await fetch('/set_perf', {{method:'POST'}});
  const d = await r.json();
  setStatus(d.ok ? 'green' : 'red', d.msg);
}}

async function loadModel() {{
  const key    = document.getElementById('modelSelect').value;
  const tokens = document.getElementById('tokenSelect').value;
  setStatus('yellow', 'Loading model... (20-60s)');
  try {{
    const r = await fetch('/load_model', {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{model: key, image_max_tokens: parseInt(tokens)}})
    }});
    const d = await r.json();
    setStatus(d.ok ? 'green' : 'red', d.ok ? 'Loaded: ' + d.label : 'Failed: ' + d.error);
  }} catch(e) {{
    setStatus('red', 'Error: ' + e);
  }}
}}

async function startCamera() {{
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({{
    video: {{ facingMode: document.getElementById('camSelect').value,
              width: {{ideal:640}}, height: {{ideal:480}} }}
  }});
  video.srcObject = stream;
  await video.play();
}}

function captureFrame() {{
  canvas.width = 192; canvas.height = 192;
  ctx.drawImage(video, 0, 0, 192, 192);
  return canvas.toDataURL('image/jpeg', 0.35).split(',')[1];
}}

async function analyzeOnce() {{
  if (!running || busy) return;
  busy = true;
  try {{
    const b64 = captureFrame();
    const t0  = performance.now();
    const r   = await fetch('/analyze', {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{image: b64}})
    }});
    const d  = await r.json();
    const dt = (performance.now() - t0) / 1000;
    const now = new Date().toLocaleTimeString();
    const txt = d.caption || 'No response';

    count++; totalTime += dt;
    document.getElementById('tTotal').textContent  = dt.toFixed(2) + 's';
    document.getElementById('tServer').textContent = (d.time_s || dt.toFixed(2)) + 's';
    document.getElementById('tAvg').textContent    = (totalTime/count).toFixed(2) + 's';
    document.getElementById('tCount').textContent  = count;

    capBox.textContent = txt;
    const line = document.createElement('div');
    line.textContent = '[' + now + '] (' + dt.toFixed(2) + 's) ' + txt;
    logBox.prepend(line);
    while (logBox.children.length > 60) logBox.removeChild(logBox.lastChild);
    setStatus('green', 'Running — ' + dt.toFixed(2) + 's | every ' + slider.value + 's');
  }} catch(e) {{
    setStatus('red', 'Error: ' + e);
  }}
  busy = false;
}}

async function startLoop() {{
  try {{ await startCamera(); }} catch(e) {{
    setStatus('red', 'Camera error: ' + e); return;
  }}
  running = true; count = 0; totalTime = 0;
  setStatus('green', 'Running — every ' + slider.value + 's');
  analyzeOnce();
  timerId = setInterval(analyzeOnce, getIntervalMs());
}}

function stopLoop() {{
  running = false;
  if (timerId) {{ clearInterval(timerId); timerId = null; }}
  setStatus('yellow', 'Stopped — ' + count + ' frames, avg ' +
    (count ? (totalTime/count).toFixed(2) : '—') + 's');
  capBox.textContent = 'Stopped. Press Start to resume.';
}}
</script>
</body>
</html>'''


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(get_html().encode("utf-8"))

    def do_POST(self):
        global IMAGE_MAX_TOKENS

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == "/analyze":
            b64 = body.get("image", "")
            t0 = time.time()
            caption = analyze_image(b64)
            dt = time.time() - t0
            enc = f"{last_encode_ms:.0f}ms" if last_encode_ms else "?"
            evl = f"{last_eval_ms:.0f}ms"   if last_eval_ms   else "?"
            print(f"[{time.strftime('%H:%M:%S')}] {dt:.2f}s "
                  f"(encode={enc} llm={evl}) | {caption}", flush=True)
            self._json_response({"caption": caption, "time_s": f"{dt:.2f}"})

        elif self.path == "/load_model":
            model_key        = body.get("model", "")
            image_max_tokens = body.get("image_max_tokens", IMAGE_MAX_TOKENS)
            if model_key not in MODELS:
                self._json_response({"ok": False, "error": "Unknown model"})
                return
            IMAGE_MAX_TOKENS = int(image_max_tokens)
            ok, msg = start_llama(model_key)
            label = MODELS[model_key]["label"]
            self._json_response({"ok": ok, "label": label,
                                  "error": msg if not ok else ""})

        elif self.path == "/set_perf":
            ok = set_cpu_performance()
            freq = get_cpu_freq()
            msg = f"CPU locked to performance mode | freq={freq}" if ok \
                  else "Failed — run: echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
            self._json_response({"ok": ok, "msg": msg})

        else:
            self.send_response(404)
            self.end_headers()

    def _json_response(self, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


def generate_ssl_cert():
    if os.path.isfile(CERT_FILE) and os.path.isfile(KEY_FILE):
        return
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048",
        "-keyout", KEY_FILE, "-out", CERT_FILE,
        "-days", "365", "-nodes", "-subj", "/CN=pivision",
    ], check=True)


def cleanup(sig, frame):
    print("\n[SERVER] Shutting down...", flush=True)
    stop_llama()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT,  cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    generate_ssl_cert()

    # Try to set performance mode at startup
    set_cpu_performance()

    throttled, tval = check_throttle()
    temp = get_temp()
    freq = get_cpu_freq()
    print(f"[INFO] CPU freq={freq} | temp={temp} | "
          f"throttled={'YES — add cooling!' if throttled else 'NO'} ({tval})", flush=True)

    for key, m in MODELS.items():
        if os.path.isfile(m["mmproj"]):
            mb = os.path.getsize(m["mmproj"]) / (1024*1024)
            print(f"[INFO] {key}: mmproj {mb:.1f} MB", flush=True)

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(CERT_FILE, KEY_FILE)

    server = ThreadingHTTPServer(("0.0.0.0", APP_PORT), Handler)
    server.socket = context.wrap_socket(server.socket, server_side=True)

    try:
        ip = subprocess.check_output(["hostname", "-I"]).decode().strip().split()[0]
    except:
        ip = "YOUR_PI_IP"

    print(f"\n=============================================")
    print(f"  Pi 5 Live Vision (CPU Optimized)")
    print(f"  https://{ip}:{APP_PORT}")
    print(f"=============================================\n")

    server.serve_forever()
