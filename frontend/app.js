const video = document.getElementById("video");
const frameHint = document.getElementById("frameHint");
const statusBadge = document.getElementById("statusBadge");
const statusText = document.getElementById("statusText");
const caption = document.getElementById("caption");
const captionBox = document.getElementById("captionBox");
const errorBanner = document.getElementById("errorBanner");
const healthDot = document.getElementById("healthDot");
const healthText = document.getElementById("healthText");

const startBtn = document.getElementById("startBtn");
const pauseBtn = document.getElementById("pauseBtn");
const resumeBtn = document.getElementById("resumeBtn");
const stopBtn = document.getElementById("stopBtn");
const clearBtn = document.getElementById("clearBtn");

const intervalInput = document.getElementById("interval");
const intervalOut = document.getElementById("intervalOut");
const frequencySelect = document.getElementById("frequency");
const captionSide = document.getElementById("captionSide");
const contextInput = document.getElementById("context");
const stage = document.getElementById("stage");
const voiceSelect = document.getElementById("voice");
const voiceEngine = document.getElementById("voiceEngine");
const ttsToggle = document.getElementById("ttsToggle");
const historyToggle = document.getElementById("historyToggle");

const notes = document.getElementById("notes");
const notesBody = document.getElementById("notesBody");
const historyPanel = document.getElementById("history");
const historyList = document.getElementById("historyList");

const captureCanvas = document.createElement("canvas");

const FREQUENCY_GAP = { sparse: 20, balanced: 6, chatty: 2 };

const state = {
  stream: null,
  running: false,
  paused: false,
  inFlight: false,
  timer: null,
  hideTimer: null,
  lastCaption: null,
  currentAudio: null,
  ttsAvailable: false,
  config: null,
};

function intervalMs() {
  return Math.max(3, Number(intervalInput.value) || 5) * 1000;
}

function minGap() {
  return FREQUENCY_GAP[frequencySelect.value] ?? 6;
}

function setStatus(text, kind) {
  statusText.textContent = text;
  statusBadge.dataset.state = kind;
}

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.hidden = false;
  clearTimeout(showError.timer);
  showError.timer = setTimeout(() => {
    errorBanner.hidden = true;
  }, 8000);
}

function updateButtons() {
  startBtn.disabled = state.running;
  pauseBtn.disabled = !state.running || state.paused;
  resumeBtn.disabled = !state.running || !state.paused;
  stopBtn.disabled = !state.running;
}

function setNotes(text) {
  notesBody.textContent = text || "No observations yet.";
}

function showCaption(text) {
  caption.textContent = text;
  captionBox.classList.add("show");
  clearTimeout(state.hideTimer);
  state.hideTimer = setTimeout(() => captionBox.classList.remove("show"), 20000);
}

function addHistory(text) {
  const li = document.createElement("li");
  const time = document.createElement("span");
  time.className = "history-time";
  time.textContent = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  const body = document.createElement("span");
  body.textContent = text;
  li.appendChild(time);
  li.appendChild(body);
  historyList.prepend(li);
  while (historyList.children.length > 50) {
    historyList.removeChild(historyList.lastChild);
  }
}

function clearCaption() {
  caption.textContent = "";
  captionBox.classList.remove("show");
  clearTimeout(state.hideTimer);
  state.lastCaption = null;
}

function grabFrame() {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return null;
  const targetAspect = 9 / 16;
  let sw = vw;
  let sh = vw / targetAspect;
  if (sh > vh) {
    sh = vh;
    sw = vh * targetAspect;
  }
  const sx = (vw - sw) / 2;
  const sy = (vh - sh) / 2;
  captureCanvas.width = 405;
  captureCanvas.height = 720;
  const ctx = captureCanvas.getContext("2d");
  ctx.drawImage(video, sx, sy, sw, sh, 0, 0, captureCanvas.width, captureCanvas.height);
  return captureCanvas.toDataURL("image/jpeg", 0.72);
}

function handleResult(data) {
  if (!data) return;
  if (data.status === "ok" && data.caption) {
    if (data.observation) setNotes(data.observation);
    if (data.caption !== state.lastCaption) {
      state.lastCaption = data.caption;
      showCaption(data.caption);
      addHistory(data.caption);
      if (ttsToggle.checked) speak(data.caption);
    }
    return;
  }
  if (data.status === "error") {
    setStatus("ERROR", "error");
    showError(data.error || "The model returned an error.");
    return;
  }
  if (data.observation) {
    setNotes(`${data.observation}${data.reason ? `  [${data.reason}]` : ""}`);
  } else if (data.reason) {
    setNotes(`[${data.reason}]`);
  }
}

function schedule() {
  if (!state.running || state.paused) return;
  clearTimeout(state.timer);
  state.timer = setTimeout(runOnce, intervalMs());
}

async function runOnce() {
  if (!state.running || state.paused) return;
  clearTimeout(state.timer);
  if (state.inFlight) {
    schedule();
    return;
  }
  const image = grabFrame();
  if (!image) {
    schedule();
    return;
  }
  state.inFlight = true;
  setStatus("OBSERVING", "observing");
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image,
        min_gap: minGap(),
        context: contextInput.value.trim(),
      }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      showError(data.error || `Analysis request failed (${response.status}).`);
    } else {
      handleResult(data);
    }
  } catch (err) {
    showError("Cannot reach the backend. Is the server still running?");
  } finally {
    state.inFlight = false;
    if (state.running && !state.paused) setStatus("LIVE", "live");
    schedule();
  }
}

function cameraErrorMessage(err) {
  if (!err) return "Could not access the webcam.";
  switch (err.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Webcam permission denied. Allow camera access for this page, then press Start again.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No webcam was found. Connect a camera and press Start again.";
    case "NotReadableError":
      return "The webcam is already in use by another application.";
    default:
      return `Webcam error: ${err.message || err.name}`;
  }
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    state.stream = stream;
    video.srcObject = stream;
    await video.play().catch(() => {});
    video.classList.add("live");
    frameHint.classList.add("hidden");
    return true;
  } catch (err) {
    showError(cameraErrorMessage(err));
    setStatus("ERROR", "error");
    return false;
  }
}

function stopStream() {
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
  video.srcObject = null;
  video.classList.remove("live");
  frameHint.classList.remove("hidden");
}

async function start() {
  if (state.running) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError("This browser does not support webcam capture. Use Chrome, Safari, or Firefox.");
    return;
  }
  const ok = await startCamera();
  if (!ok) return;
  state.running = true;
  state.paused = false;
  updateButtons();
  setStatus("LIVE", "live");
  runOnce();
}

function pause() {
  if (!state.running || state.paused) return;
  state.paused = true;
  clearTimeout(state.timer);
  setStatus("PAUSED", "idle");
  updateButtons();
}

function resume() {
  if (!state.running || !state.paused) return;
  state.paused = false;
  updateButtons();
  setStatus("LIVE", "live");
  runOnce();
}

function stop() {
  state.running = false;
  state.paused = false;
  state.inFlight = false;
  clearTimeout(state.timer);
  stopStream();
  clearCaption();
  setStatus("IDLE", "idle");
  updateButtons();
}

function browserSpeak(text) {
  if (!window.speechSynthesis) return;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 0.92;
  utterance.pitch = 0.85;
  window.speechSynthesis.speak(utterance);
}

function stopAudio() {
  if (state.currentAudio) {
    try {
      state.currentAudio.pause();
    } catch (err) {
      /* ignore */
    }
    state.currentAudio = null;
  }
  if (window.speechSynthesis) window.speechSynthesis.cancel();
}

async function speak(text) {
  stopAudio();
  if (!state.ttsAvailable) {
    browserSpeak(text);
    return;
  }
  try {
    const response = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, voice: voiceSelect.value }),
    });
    if (!response.ok) {
      browserSpeak(text);
      return;
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    state.currentAudio = audio;
    audio.onended = () => URL.revokeObjectURL(url);
    await audio.play().catch(() => browserSpeak(text));
  } catch (err) {
    browserSpeak(text);
  }
}

async function clearHistory() {
  historyList.innerHTML = "";
  clearCaption();
  setNotes("");
  try {
    await fetch("/api/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
  } catch (err) {
    /* reset is best-effort */
  }
}

function updateHealth(data) {
  if (!data || !data.ollama_ok) {
    healthDot.className = "health-dot bad";
    healthText.textContent = "Ollama offline";
    return;
  }
  if (data.models_missing && data.models_missing.length) {
    healthDot.className = "health-dot bad";
    healthText.textContent = `missing model: ${data.models_missing.join(", ")}`;
    return;
  }
  healthDot.className = "health-dot ok";
  healthText.textContent = `${data.vision_model} + ${data.narrator_model}`;
}

async function pollHealth() {
  try {
    const response = await fetch("/api/health");
    updateHealth(await response.json());
  } catch (err) {
    updateHealth(null);
  }
}

async function loadConfig() {
  try {
    const response = await fetch("/api/config");
    const data = await response.json();
    state.config = data;
    state.ttsAvailable = Boolean(data.tts_available);
    if (contextInput && data.narrator_context) {
      contextInput.value = data.narrator_context;
    }
    let voices = Array.isArray(data.voices) ? data.voices.slice() : [];
    if (data.tts_voice && !voices.includes(data.tts_voice)) {
      voices.unshift(data.tts_voice);
    }
    if (!voices.length) voices = [data.tts_voice];
    voiceSelect.innerHTML = "";
    voices.forEach((voice) => {
      const option = document.createElement("option");
      option.value = voice;
      option.textContent = voice;
      if (voice === data.tts_voice) option.selected = true;
      voiceSelect.appendChild(option);
    });
    ttsToggle.disabled = !state.ttsAvailable;
    if (voiceEngine) {
      voiceEngine.textContent = data.tts_engine && data.tts_engine !== "none"
        ? `[${data.tts_engine}]`
        : "";
    }
    if (!state.ttsAvailable) {
      ttsToggle.checked = false;
      ttsToggle.title = "No local TTS engine found; browser speech will be used.";
    }
  } catch (err) {
    healthDot.className = "health-dot bad";
    healthText.textContent = "backend offline";
  }
}

startBtn.addEventListener("click", start);
pauseBtn.addEventListener("click", pause);
resumeBtn.addEventListener("click", resume);
stopBtn.addEventListener("click", stop);
clearBtn.addEventListener("click", clearHistory);

intervalInput.addEventListener("input", () => {
  intervalOut.textContent = `${intervalInput.value}s`;
});

historyToggle.addEventListener("change", () => {
  historyPanel.hidden = !historyToggle.checked;
});

captionSide.addEventListener("change", () => {
  const left = captionSide.value === "left";
  stage.classList.toggle("side-left", left);
  stage.classList.toggle("side-right", !left);
});

notesToggleInit();

function notesToggleInit() {
  const toggle = document.createElement("button");
  toggle.textContent = "Observer notes";
  toggle.className = "btn subtle";
  toggle.style.alignSelf = "flex-start";
  toggle.addEventListener("click", () => {
    notes.hidden = !notes.hidden;
  });
  document.querySelector(".grid").after(toggle);
}

window.addEventListener("beforeunload", () => {
  stop();
});

loadConfig();
pollHealth();
setInterval(pollHealth, 10000);
updateButtons();
setStatus("IDLE", "idle");
