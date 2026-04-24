// ── Chord colour map (one RGB per root, matches new Spotify-palette theme) ──
const CHORD_COLORS = {
    'C':  '229,57,53',   'C#': '237,85,59',
    'D':  '245,124,0',   'D#': '249,168,37',
    'E':  '220,200,50',  'F':  '67,160,71',
    'F#': '0,150,136',   'G':  '30,136,229',
    'G#': '3,169,244',   'A':  '142,36,170',
    'A#': '94,53,177',   'B':  '216,27,96',
};

// ── DOM references ────────────────────────────────────────────────────────────
const connectionStatus = document.getElementById('connection-status');
const statusIndicator  = document.querySelector('.status-indicator');
const micToggleBtn     = document.getElementById('mic-toggle');
const eqIcon           = document.getElementById('eq-icon');
const eqBars           = eqIcon ? Array.from(eqIcon.querySelectorAll('.eq-bar')) : [];

const searchSection   = document.getElementById('search-section');
const overviewSection = document.getElementById('overview-section');
const timelineSection = document.getElementById('timeline-section');

const songInput        = document.getElementById('song-input');
const autocompleteList = document.getElementById('autocomplete-list');
const searchBtn        = document.getElementById('search-btn');
const loadingContainer = document.getElementById('loading-container');
const loadingText      = document.getElementById('loading-text');
const chordGrid        = document.getElementById('chord-grid');

const practiceBtn      = document.getElementById('practice-btn');
const newSearchBtn     = document.getElementById('new-search-btn');
const resultsSection   = document.getElementById('results-section');
const resultsList      = document.getElementById('results-list');

const timelineTrack    = document.getElementById('timeline-track');
const playhead         = document.getElementById('playhead');
const playPauseBtn     = document.getElementById('play-pause-btn');
const backToOverviewBtn= document.getElementById('back-to-overview-btn');
const nowChordEl       = document.getElementById('now-chord');
const timeDisplay      = document.getElementById('time-display');
const audioPlayer      = document.getElementById('audio-player');

const currentChordEl   = document.getElementById('current-chord');
const targetChordEl    = document.getElementById('target-chord');
const confidenceVal    = document.getElementById('confidence-val');
const volumeVal        = document.getElementById('volume-val');
const playerBox        = document.getElementById('player-box');
const targetBox        = document.getElementById('target-box');

let chordTimeline = [];
let songDuration  = 0;
let isListening   = false;
let ws            = null;
let lastChord     = "";
let currentChord  = "--";
let targetChord   = "--";
let animFrame     = null;

// ── Browser microphone state ──────────────────────────────────────────────────
let micStream    = null;   // MediaStream from getUserMedia
let audioCtx     = null;   // AudioContext for mic processing
let micProcessor = null;   // ScriptProcessorNode that sends PCM to server

// ── Equalizer helpers ─────────────────────────────────────────────────────────
function setEqActive(active) {
    if (!eqIcon) return;
    eqIcon.classList.toggle('active', active);
}

function driveEq(normVolume) {
    if (!eqBars.length) return;
    const seeds = [0.9, 1.0, 0.75, 0.85];
    eqBars.forEach((bar, i) => {
        const h = Math.max(3, Math.round(normVolume * 18 * seeds[i % seeds.length]));
        bar.style.height = h + 'px';
    });
}

// ── Browser mic capture ───────────────────────────────────────────────────────
async function startMicCapture() {
    try {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
        alert('Microphone access denied. Please allow mic permission and try again.');
        return false;
    }

    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 22050 });
    const source = audioCtx.createMediaStreamSource(micStream);

    // 4096-sample buffer ≈ 186 ms at 22050 Hz — good balance of latency vs overhead
    micProcessor = audioCtx.createScriptProcessor(4096, 1, 1);
    micProcessor.onaudioprocess = (e) => {
        if (!isListening || !ws || ws.readyState !== WebSocket.OPEN) return;
        const pcm = e.inputBuffer.getChannelData(0);  // Float32Array
        ws.send(pcm.buffer);  // send raw float32 PCM as binary
    };

    source.connect(micProcessor);
    micProcessor.connect(audioCtx.destination);  // required for ScriptProcessor to fire
    return true;
}

function stopMicCapture() {
    if (micProcessor) { micProcessor.disconnect(); micProcessor = null; }
    if (audioCtx)     { audioCtx.close(); audioCtx = null; }
    if (micStream)    { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connect() {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${window.location.host}/ws`);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        connectionStatus.textContent = 'Ready';
        statusIndicator.style.background = '';
        micToggleBtn.disabled = false;
        micToggleBtn.classList.remove('disabled');
    };

    ws.onclose = () => {
        connectionStatus.textContent = 'Reconnecting…';
        statusIndicator.style.background = '#E91429';
        statusIndicator.classList.remove('connected');
        micToggleBtn.disabled = true;
        micToggleBtn.classList.add('disabled');
        isListening = false;
        stopMicCapture();
        setEqActive(false);
        setTimeout(connect, 2000);
    };

    ws.onerror = () => ws.close();

    ws.onmessage = (event) => {
        if (!isListening) return;
        const data = JSON.parse(event.data);

        volumeVal.textContent     = data.volume.toFixed(4);
        confidenceVal.textContent = `${Math.round(data.confidence * 100)}%`;
        currentChord              = data.chord;
        currentChordEl.textContent = currentChord;

        const normVol = Math.min(data.volume * 150, 1);
        driveEq(normVol);

        if (data.chord !== lastChord && lastChord !== "") {
            playerBox.style.transform = 'scale(0.95)';
            setTimeout(() => { playerBox.style.transform = ''; }, 150);
        }
        lastChord = data.chord;
        checkMatch();
    };
}

function checkMatch() {
    const match = currentChord === targetChord && currentChord !== "--";
    playerBox.classList.toggle('match', match);
    targetBox.classList.toggle('match', match);
}

micToggleBtn.addEventListener('click', async () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    if (!isListening) {
        // Start: request mic permission and begin capture
        const ok = await startMicCapture();
        if (!ok) return;
        isListening = true;
        ws.send(JSON.stringify({ command: "START" }));
    } else {
        // Stop: tear down mic capture
        isListening = false;
        ws.send(JSON.stringify({ command: "STOP" }));
        stopMicCapture();
    }

    micToggleBtn.textContent = isListening ? "Stop Listening" : "Start Listening";
    micToggleBtn.className   = isListening ? "btn listening" : "btn primary";

    connectionStatus.textContent = isListening ? 'Listening…' : 'Ready';
    statusIndicator.classList.toggle('connected', isListening);

    setEqActive(isListening);
    if (!isListening) driveEq(0);

    if (!isListening) {
        currentChord = "--";
        currentChordEl.textContent = "--";
        confidenceVal.textContent  = "0%";
        volumeVal.textContent      = "0.000";
        lastChord = "";
        checkMatch();
    }
});

// ── Autocomplete ──────────────────────────────────────────────────────────────
let debounceTimer;
songInput.addEventListener('input', (e) => {
    clearTimeout(debounceTimer);
    const q = e.target.value.trim();
    if (!q) { autocompleteList.classList.add('hidden'); return; }
    debounceTimer = setTimeout(async () => {
        try {
            const res         = await fetch('/api/autocomplete?q=' + encodeURIComponent(q));
            const suggestions = await res.json();
            autocompleteList.innerHTML = '';
            if (suggestions.length > 0) {
                suggestions.forEach(s => {
                    const li = document.createElement('li');
                    li.textContent = s;
                    li.addEventListener('click', () => {
                        songInput.value = s;
                        autocompleteList.classList.add('hidden');
                    });
                    autocompleteList.appendChild(li);
                });
                autocompleteList.classList.remove('hidden');
            } else {
                autocompleteList.classList.add('hidden');
            }
        } catch (err) { console.error(err); }
    }, 300);
});
document.addEventListener('click', (e) => {
    if (!e.target.closest('.autocomplete-wrapper')) autocompleteList.classList.add('hidden');
});

// ── Search & Extract ──────────────────────────────────────────────────────────
const LOADING_MESSAGES = [
    'Downloading audio…',
    'Separating harmonics…',
    'Extracting chroma features…',
    'Tracking beats…',
    'Running transformer inference…',
    'Detecting chord patterns…',
    'Analysis takes up to 2 minutes on free tier…',
    'Almost there…',
];
let loadingInterval = null;

function startLoadingCycle() {
    let i = 0;
    if (loadingText) loadingText.textContent = LOADING_MESSAGES[0];
    loadingInterval = setInterval(() => {
        i = (i + 1) % LOADING_MESSAGES.length;
        if (loadingText) loadingText.textContent = LOADING_MESSAGES[i];
    }, 2500);
}
function stopLoadingCycle() {
    clearInterval(loadingInterval);
    loadingInterval = null;
}

// ── Step 1: search → show song list ───────────────────────────────────────────
searchBtn.addEventListener('click', async () => {
    const query = songInput.value.trim();
    if (!query) return;
    searchBtn.disabled = true;
    autocompleteList.classList.add('hidden');
    resultsSection.classList.add('hidden');
    overviewSection.classList.add('hidden');
    timelineSection.classList.add('hidden');
    loadingContainer.classList.remove('hidden');
    if (loadingText) loadingText.textContent = 'Searching YouTube Music…';

    try {
        const res   = await fetch('/api/suggest?q=' + encodeURIComponent(query));
        const songs = await res.json();

        resultsList.innerHTML = '';
        songs.forEach(song => {
            const li = document.createElement('li');
            li.className = 'result-card';

            if (song.thumbnail) {
                const img = document.createElement('img');
                img.className = 'result-thumb';
                img.src = song.thumbnail;
                img.alt = '';
                li.appendChild(img);
            } else {
                const ph = document.createElement('div');
                ph.className = 'result-thumb result-thumb--empty';
                li.appendChild(ph);
            }

            const info = document.createElement('div');
            info.className = 'result-info';
            const titleEl = document.createElement('span');
            titleEl.className = 'result-title';
            titleEl.textContent = song.title;
            const metaEl = document.createElement('span');
            metaEl.className = 'result-meta';
            metaEl.textContent = [song.artist, song.duration].filter(Boolean).join(' · ');
            info.appendChild(titleEl);
            info.appendChild(metaEl);
            li.appendChild(info);

            const pick = document.createElement('span');
            pick.className = 'result-pick';
            pick.textContent = '▶';
            li.appendChild(pick);

            li.addEventListener('click', () => extractChords(song.videoId, query));
            resultsList.appendChild(li);
        });

        if (songs.length > 0) {
            resultsSection.classList.remove('hidden');
        } else {
            alert('No songs found. Try a different search.');
        }
    } catch (err) {
        alert('Search error: ' + err.message);
    } finally {
        searchBtn.disabled = false;
        loadingContainer.classList.add('hidden');
    }
});

// ── Step 2: user picks a song → download + extract ────────────────────────────
async function extractChords(videoId, query) {
    resultsSection.classList.add('hidden');
    loadingContainer.classList.remove('hidden');
    startLoadingCycle();
    audioPlayer.pause();
    audioPlayer.removeAttribute('src');
    audioPlayer.load();
    chordTimeline = [];

    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, video_id: videoId })
        });
        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Server error ${response.status}`);
        }
        const data = await response.json();
        chordTimeline   = data.timeline;
        audioPlayer.src = data.audio_url + "?t=" + Date.now();

        const mainChords = data.main_chords || [];
        chordGrid.innerHTML = '';
        mainChords.forEach(chord => {
            const pill = document.createElement('div');
            pill.className = 'chord-pill selectable';
            pill.textContent = chord;
            pill.title = 'Click to set as practice target';
            pill.addEventListener('click', () => {
                targetChord = chord;
                if (targetChordEl) targetChordEl.textContent = chord;
                chordGrid.querySelectorAll('.chord-pill').forEach(p => p.classList.remove('selected'));
                pill.classList.add('selected');
                checkMatch();
            });
            chordGrid.appendChild(pill);
        });

        searchSection.classList.add('hidden');
        overviewSection.classList.remove('hidden');
    } catch (err) {
        alert('Error: ' + err.message);
        resultsSection.classList.remove('hidden');
    } finally {
        stopLoadingCycle();
        loadingContainer.classList.add('hidden');
    }
}

// ── Timeline Rendering ────────────────────────────────────────────────────────
// Extract root from any chord name (C, Cm, C5, Csus2, Csus4, F#m, etc.)
function chordRoot(chordName) {
    if (!chordName) return '';
    // Handle sharp roots (2-char) first, then naturals
    if (chordName.length >= 2 && chordName[1] === '#') return chordName.slice(0, 2);
    return chordName.slice(0, 1);
}
function chordColor(chordName, opacity) {
    const rgb = CHORD_COLORS[chordRoot(chordName)] || '100,100,200';
    return `rgba(${rgb}, ${opacity})`;
}

function renderTimeline() {
    timelineTrack.querySelectorAll('.chord-block').forEach(el => el.remove());
    if (chordTimeline.length === 0 || songDuration === 0) return;

    chordTimeline.forEach((seg, i) => {
        const block    = document.createElement('div');
        block.className    = 'chord-block';
        block.dataset.chord = seg.chord;
        block.dataset.index = i;

        const leftPct  = (seg.start / songDuration) * 100;
        const widthPct = ((seg.end - seg.start) / songDuration) * 100;
        block.style.left  = leftPct  + '%';
        block.style.width = widthPct + '%';

        const isMinor = seg.chord.endsWith('m');
        block.style.background = chordColor(seg.chord, isMinor ? 0.28 : 0.48);

        if (widthPct > 2.5) block.textContent = seg.chord;

        timelineTrack.appendChild(block);
    });
}

// ── Playhead & Sync Loop ──────────────────────────────────────────────────────
function updatePlayhead() {
    if (songDuration > 0) {
        const pct = (audioPlayer.currentTime / songDuration) * 100;
        playhead.style.left = Math.min(pct, 100) + '%';
        timeDisplay.textContent = `${formatTime(audioPlayer.currentTime)} / ${formatTime(songDuration)}`;

        let found = "--";
        for (const seg of chordTimeline) {
            if (audioPlayer.currentTime >= seg.start && audioPlayer.currentTime < seg.end) {
                found = seg.chord;
                break;
            }
        }
        if (targetChord !== found) {
            targetChord = found;
            targetChordEl.textContent = targetChord;
            nowChordEl.textContent    = targetChord;

            timelineTrack.querySelectorAll('.chord-block').forEach(b => b.classList.remove('active'));
            const activeIdx = chordTimeline.findIndex(
                s => audioPlayer.currentTime >= s.start && audioPlayer.currentTime < s.end
            );
            if (activeIdx >= 0) {
                const activeBlock = timelineTrack.querySelector(`[data-index="${activeIdx}"]`);
                if (activeBlock) activeBlock.classList.add('active');
            }
            checkMatch();
        }
    }

    playPauseBtn.textContent = audioPlayer.paused ? '▶ Play' : '⏸ Pause';
    animFrame = requestAnimationFrame(updatePlayhead);
}

function formatTime(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

timelineTrack.addEventListener('click', (e) => {
    const rect = timelineTrack.getBoundingClientRect();
    const pct  = (e.clientX - rect.left) / rect.width;
    if (songDuration > 0) audioPlayer.currentTime = pct * songDuration;
});

// ── Practice Flow ─────────────────────────────────────────────────────────────
practiceBtn.addEventListener('click', () => {
    overviewSection.classList.add('hidden');
    timelineSection.classList.remove('hidden');
    micToggleBtn.classList.remove('hidden');

    const go = () => {
        songDuration = audioPlayer.duration;
        renderTimeline();
        if (!animFrame) animFrame = requestAnimationFrame(updatePlayhead);
    };

    if (audioPlayer.readyState >= 1) {
        go();
    } else {
        audioPlayer.addEventListener('loadedmetadata', go, { once: true });
    }
});

playPauseBtn.addEventListener('click', () => {
    audioPlayer.paused ? audioPlayer.play() : audioPlayer.pause();
});

newSearchBtn.addEventListener('click', () => {
    overviewSection.classList.add('hidden');
    timelineSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    searchSection.classList.remove('hidden');
    micToggleBtn.classList.add('hidden');

    // Stop mic if active
    if (isListening && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ command: "STOP" }));
    }
    isListening = false;
    stopMicCapture();
    setEqActive(false);
    driveEq(0);
    micToggleBtn.textContent = "Start Listening";
    micToggleBtn.className   = "btn primary";

    // Release audio file so server can delete it on next search
    audioPlayer.pause();
    audioPlayer.removeAttribute('src');
    audioPlayer.load();
    songInput.value = '';
});

backToOverviewBtn.addEventListener('click', () => {
    timelineSection.classList.add('hidden');
    overviewSection.classList.remove('hidden');
    micToggleBtn.classList.add('hidden');
    setEqActive(false);
    audioPlayer.pause();
});

// ── Init ──────────────────────────────────────────────────────────────────────
micToggleBtn.disabled = true;
connect();
