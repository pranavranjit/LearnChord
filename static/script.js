'use strict';

// ------------------------------------------------------------------ constants
const PITCHES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const MIC_SAMPLE_RATE = 22050;
const LIVE_STEP_SEC = 0.25;   // the server sends a chord estimate this often
const SLEEVE = 'static/sleeve.svg';
const SLOW_NOTE_MS = 8000;

// Standard guitar voicings, low E to high e ('x' = muted, digits = fret).
// `barre` marks the fret one finger holds across several strings.
const GUITAR_SHAPES = {
    'C':   { frets: 'x32010' },            'Cm':  { frets: 'x35543', barre: 3 },
    'C#':  { frets: 'x46664', barre: 4 },  'C#m': { frets: 'x46654', barre: 4 },
    'D':   { frets: 'xx0232' },            'Dm':  { frets: 'xx0231' },
    'D#':  { frets: 'x68886', barre: 6 },  'D#m': { frets: 'x68876', barre: 6 },
    'E':   { frets: '022100' },            'Em':  { frets: '022000' },
    'F':   { frets: '133211', barre: 1 },  'Fm':  { frets: '133111', barre: 1 },
    'F#':  { frets: '244322', barre: 2 },  'F#m': { frets: '244222', barre: 2 },
    'G':   { frets: '320003' },            'Gm':  { frets: '355333', barre: 3 },
    'G#':  { frets: '466544', barre: 4 },  'G#m': { frets: '466444', barre: 4 },
    'A':   { frets: 'x02220' },            'Am':  { frets: 'x02210' },
    'A#':  { frets: 'x13331', barre: 1 },  'A#m': { frets: 'x13321', barre: 1 },
    'B':   { frets: 'x24442', barre: 2 },  'Bm':  { frets: 'x24432', barre: 2 },
};
const FALLBACK_EXAMPLES = [
    { title: 'Let It Be', artist: 'The Beatles', query: 'The Beatles Let It Be' },
    { title: 'Wonderwall', artist: 'Oasis', query: 'Oasis Wonderwall' },
    { title: 'Riptide', artist: 'Vance Joy', query: 'Vance Joy Riptide' },
    { title: 'Perfect', artist: 'Ed Sheeran', query: 'Ed Sheeran Perfect' },
    { title: 'Someone Like You', artist: 'Adele', query: 'Adele Someone Like You' },
];

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const finePointer = window.matchMedia('(hover: hover) and (pointer: fine)');

// ------------------------------------------------------------------ elements
const $ = (id) => document.getElementById(id);
const connectionStatus = $('connection-status');
const statusIndicator  = document.querySelector('.status-indicator');
const panelEl          = $('panel');
const stateItems       = document.querySelectorAll('#states li');
const stateButtons     = document.querySelectorAll('.state-btn');

const searchSection    = $('search-section');
const resultsSection   = $('results-section');
const overviewSection  = $('overview-section');
const timelineSection  = $('timeline-section');

const searchForm       = $('search-form');
const songInput        = $('song-input');
const autocompleteList = $('autocomplete-list');
const searchBtn        = $('search-btn');
const exampleChips     = $('example-chips');
const errorBanner      = $('error-banner');
const errorText        = $('error-text');
const loadingContainer = $('loading-container');
const loadingSteps     = $('loading-steps');
const loadingElapsed   = $('loading-elapsed');
const loadingNote      = $('loading-note');

const resultsList      = $('results-list');
const resultsMeta      = $('results-meta');

const songArt          = $('song-art');
const songTitle        = $('song-title');
const songArtist       = $('song-artist');
const chordGrid        = $('chord-grid');
const progressionEl    = $('progression');
const practiceBtn      = $('practice-btn');
const newSearchBtn     = $('new-search-btn');

const playerArt        = $('player-art');
const playerTitle      = $('player-title');
const playerArtist     = $('player-artist');
const playPauseBtn     = $('play-pause-btn');
const nowChordEl       = $('now-chord');
const nextChordEl      = $('next-chord');
const nextInEl         = $('next-in');
const timeDisplay      = $('time-display');
const backBtn          = $('back-to-overview-btn');
const timelineTrack    = $('timeline-track');
const trackScale       = $('track-scale');
const playhead         = $('playhead');
const audioPlayer      = $('audio-player');

const modeButtons      = document.querySelectorAll('.mode-toggle button');
const drillPicker      = $('drill-picker');
const micToggleBtn     = $('mic-toggle');
const micState         = $('mic-state');
const micAction        = $('mic-action');
const playerBox        = $('player-box');
const targetBox        = $('target-box');
const currentChordEl   = $('current-chord');
const targetChordEl    = $('target-chord');
const targetLabel      = $('target-label');
const targetDiagram    = $('target-diagram');
const matchBadge       = $('match-badge');
const matchText        = $('match-text');
const volumeBar        = $('volume-bar');
const confidenceVal    = $('confidence-val');
const confidenceScale  = $('confidence-scale');
const accuracyVal      = $('accuracy-val');
const accuracyScale    = $('accuracy-scale');
const streakVal        = $('streak-val');

// ------------------------------------------------------------------ state
let stage         = 'search';   // 'search' | 'chords' | 'play'
let chordTimeline = [];
let songDuration  = 0;
let mainChords    = [];
let lastResults   = [];
let currentSong   = null;
let activePreset  = null;       // the preset key whose song is being analysed

let mode          = 'song';     // 'song': target follows playback, 'drill': fixed chord
let drillChord    = null;
let targetChord   = '--';
let currentChord  = '--';
let stats         = { total: 0, hits: 0, streak: 0, best: 0 };

let ws            = null;
let isListening   = false;
let micStream     = null;
let audioCtx      = null;
let micProcessor  = null;
let micSink       = null;
let animFrame     = null;
let busy          = false;

// ------------------------------------------------------------------ helpers
function show(el) { el.classList.remove('hidden'); }
function hide(el) { el.classList.add('hidden'); }

let errorTimer = null;
function showError(message) {
    errorText.textContent = message;
    show(errorBanner);
    clearTimeout(errorTimer);
    errorTimer = setTimeout(clearError, 9000);
}
function clearError() { hide(errorBanner); }
$('error-close').addEventListener('click', clearError);

let loadingTimers = [];
let loadingClock = null;
let slowTimer = null;
function startLoading(labels) {
    loadingSteps.replaceChildren();
    labels.forEach((text, i) => {
        const li = document.createElement('li');
        const lamp = document.createElement('span');
        lamp.className = 'lamp';
        lamp.setAttribute('aria-hidden', 'true');
        li.append(lamp, text);
        if (i === 0) li.classList.add('active');
        loadingSteps.appendChild(li);
    });
    loadingTimers.forEach(clearTimeout);
    // The server does not stream progress; advance on typical timings and hold
    // on the last step until the response arrives.
    loadingTimers = [700, 1800].slice(0, labels.length - 1).map((ms, i) => setTimeout(() => {
        const items = loadingSteps.children;
        items[i].classList.replace('active', 'done');
        items[i + 1].classList.add('active');
    }, ms));
    hide(loadingNote);
    clearTimeout(slowTimer);
    slowTimer = setTimeout(() => show(loadingNote), SLOW_NOTE_MS);
    const t0 = performance.now();
    clearInterval(loadingClock);
    loadingElapsed.textContent = '0.0s';
    loadingClock = setInterval(() => {
        loadingElapsed.textContent = `${((performance.now() - t0) / 1000).toFixed(1)}s`;
    }, 100);
    show(loadingContainer);
}
function stopLoading() {
    loadingTimers.forEach(clearTimeout);
    clearInterval(loadingClock);
    clearTimeout(slowTimer);
    hide(loadingContainer);
}

function setBusy(on) {
    busy = on;
    searchBtn.disabled = on;
    exampleChips.querySelectorAll('button').forEach(b => { b.disabled = on; });
    resultsList.querySelectorAll('button').forEach(b => { b.disabled = on; });
}

async function postJSON(url, body) {
    const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `Server error ${res.status}`);
    }
    return res.json();
}

function formatTime(sec) {
    if (!isFinite(sec)) return '0:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function setArt(img, url) {
    img.onerror = () => { img.onerror = null; img.src = SLEEVE; };
    img.src = url || SLEEVE;
}

// Swap the page between states. Where the browser supports it, the picked
// song is one object that morphs from result to song panel to player strip.
function morph(update, source) {
    if (!document.startViewTransition || reducedMotion.matches) { update(); return; }
    if (source) source.style.viewTransitionName = 'song';
    try {
        document.startViewTransition(() => {
            if (source) source.style.viewTransitionName = '';
            update();
        });
    } catch (err) {
        if (source) source.style.viewTransitionName = '';
        update();
    }
}

function scrollToPanel() {
    const top = panelEl.getBoundingClientRect().top + window.scrollY - 8;
    if (window.scrollY > top) window.scrollTo({ top: Math.max(0, top), behavior: 'instant' });
}

// ------------------------------------------------------------------ chords
function chordRoot(name) {
    if (!name) return '';
    return name.length >= 2 && name[1] === '#' ? name.slice(0, 2) : name.slice(0, 1);
}
function isMinor(name) { return /m$/.test(name); }
function chordNotes(name) {
    const root = PITCHES.indexOf(chordRoot(name));
    if (root < 0) return [];
    return (isMinor(name) ? [0, 3, 7] : [0, 4, 7]).map(i => PITCHES[(root + i) % 12]);
}
function spokenChord(name) {
    const root = chordRoot(name);
    return `${root[0]}${root.length > 1 ? ' sharp' : ''} ${isMinor(name) ? 'minor' : 'major'}`;
}

// Set a chord name in the display face, with the sharp raised like a
// chord chart. An empty or '--' name shows a dash. The name sits in one
// inline span so flex containers (keys, timeline blocks) keep it whole.
function setChord(el, name) {
    const text = name && name !== '--' ? name : '';
    if (el.dataset.chord === text && el.firstChild) return;
    el.dataset.chord = text;
    el.classList.toggle('empty', !text);
    const word = document.createElement('span');
    word.className = 'chord-word';
    if (!text) {
        word.textContent = '–';
    } else {
        const root = chordRoot(text);
        word.append(root[0]);
        if (root.length > 1) {
            const acc = document.createElement('span');
            acc.className = 'acc';
            acc.textContent = '#';
            word.append(acc);
        }
        if (isMinor(text)) word.append('m');
    }
    el.replaceChildren(word);
}

function chordDiagramSVG(name) {
    const shape = GUITAR_SHAPES[name];
    if (!shape) return '';
    const frets = shape.frets.split('').map(c => (c === 'x' ? null : parseInt(c, 10)));
    const fretted = frets.filter(f => f !== null && f > 0);
    const maxFret = fretted.length ? Math.max(...fretted) : 0;
    const base = maxFret > 4 ? (shape.barre || Math.min(...fretted)) : 1;

    const W = 92, H = 108, left = 18, right = 10, top = 22, rows = 5;
    const gridW = W - left - right;
    const gap = gridW / 5;
    const fretGap = (H - top - 8) / rows;
    const yFor = (f) => top + (f - base + 0.5) * fretGap;

    let s = `<svg class="chord-diagram" viewBox="0 0 ${W} ${H}" role="img" aria-label="${spokenChord(name)} guitar shape">`;
    for (let r = 0; r <= rows; r++) {
        const y = top + r * fretGap;
        s += `<line class="${r === 0 && base === 1 ? 'nut' : 'fret'}" x1="${left}" y1="${y}" x2="${left + gridW}" y2="${y}"/>`;
    }
    for (let i = 0; i < 6; i++) {
        const x = left + i * gap;
        s += `<line class="string" x1="${x}" y1="${top}" x2="${x}" y2="${top + rows * fretGap}"/>`;
    }
    if (base > 1) {
        s += `<text class="fret-label" x="${left - 5}" y="${top + fretGap * 0.68}" text-anchor="end">${base}fr</text>`;
    }
    if (shape.barre) {
        const idx = frets.map((f, i) => (f === shape.barre ? i : -1)).filter(i => i >= 0);
        const a = Math.min(...idx), b = Math.max(...idx);
        const y = yFor(shape.barre);
        s += `<rect class="barre" x="${left + a * gap - 5}" y="${y - 5}" width="${(b - a) * gap + 10}" height="10" rx="5"/>`;
    }
    frets.forEach((f, i) => {
        const x = left + i * gap;
        if (f === null) {
            s += `<path class="mute" d="M${x - 3.5} ${top - 13}l7 7M${x + 3.5} ${top - 13}l-7 7"/>`;
        } else if (f === 0) {
            s += `<circle class="open" cx="${x}" cy="${top - 9.5}" r="3.5"/>`;
        } else if (!(shape.barre && f === shape.barre)) {
            s += `<circle class="dot" cx="${x}" cy="${yFor(f)}" r="4.6"/>`;
        }
    });
    return s + '</svg>';
}

function chordAt(t) {
    for (let i = 0; i < chordTimeline.length; i++) {
        const seg = chordTimeline[i];
        if (t >= seg.start && t < seg.end) return i;
    }
    return -1;
}

// ------------------------------------------------------------------ views
function showOnly(...sections) {
    [resultsSection, overviewSection, timelineSection].forEach(s => {
        if (!sections.includes(s)) hide(s);
    });
    sections.forEach(show);
}

// The three named states, each with a lamp; 2 and 3 open once a song is loaded.
function setStage(name) {
    stage = name;
    panelEl.dataset.stage = name;
    stateItems.forEach(li => {
        const on = li.dataset.stage === name;
        li.classList.toggle('on', on);
        const btn = li.querySelector('.state-btn');
        if (on) btn.setAttribute('aria-current', 'step');
        else btn.removeAttribute('aria-current');
    });
    stateButtons.forEach(b => { b.disabled = b.dataset.go !== 'search' && !currentSong; });
}

function markPresets(loadedKey) {
    exampleChips.querySelectorAll('.preset').forEach(k => {
        k.classList.remove('loading');
        k.classList.toggle('loaded', k === loadedKey);
    });
}

function renderResults(songs, query) {
    lastResults = songs;
    resultsList.replaceChildren();
    resultsMeta.textContent = `${songs.length} ${songs.length === 1 ? 'match' : 'matches'} for “${query}”`;
    songs.forEach(song => {
        const li = document.createElement('li');
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'result-card';

        const img = document.createElement('img');
        img.className = 'result-thumb';
        img.alt = '';
        img.loading = 'lazy';
        img.width = 52;
        img.height = 52;
        setArt(img, song.thumbnail);

        const info = document.createElement('span');
        info.className = 'result-info';
        const titleEl = document.createElement('span');
        titleEl.className = 'result-title';
        titleEl.textContent = song.title;
        const metaEl = document.createElement('span');
        metaEl.className = 'result-meta';
        metaEl.textContent = [song.artist, song.duration].filter(Boolean).join(' · ');
        info.append(titleEl, metaEl);

        const pick = document.createElement('span');
        pick.className = 'result-pick';
        pick.innerHTML = '<span class="lamp" aria-hidden="true"></span><span class="result-pick-label">Load</span>';

        btn.append(img, info, pick);
        btn.addEventListener('click', () => analyzeSong(song, { source: btn }));
        li.appendChild(btn);
        resultsList.appendChild(li);
    });
    showOnly(resultsSection);
}

function chordCard(chord, share) {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = `chord-card ${isMinor(chord) ? 'minor' : 'major'}`;
    card.setAttribute('aria-label', `${spokenChord(chord)}, ${share}% of the preview. Practise it with your mic.`);

    const name = document.createElement('span');
    name.className = 'chord-name chord';
    setChord(name, chord);
    const quality = document.createElement('span');
    quality.className = 'chord-quality';
    quality.textContent = isMinor(chord) ? 'Minor' : 'Major';
    card.append(name, quality);
    card.insertAdjacentHTML('beforeend', chordDiagramSVG(chord));

    const notes = document.createElement('span');
    notes.className = 'chord-notes';
    notes.textContent = chordNotes(chord).join(' · ');
    const shareEl = document.createElement('span');
    shareEl.className = 'chord-share';
    shareEl.innerHTML = `<span class="share-bar"><span style="width:${share}%"></span></span><span class="share-val">${share}%</span>`;
    card.append(notes, shareEl);

    card.addEventListener('click', () => morph(() => openPractice('drill', chord)));
    return card;
}

function renderDrillPicker() {
    drillPicker.replaceChildren();
    const legend = document.createElement('span');
    legend.className = 'legend';
    legend.textContent = 'Drill';
    drillPicker.appendChild(legend);
    mainChords.forEach(chord => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'drill-chip chord';
        setChord(b, chord);
        b.setAttribute('aria-label', spokenChord(chord));
        b.setAttribute('aria-pressed', 'false');
        b.addEventListener('click', () => setMode('drill', chord));
        drillPicker.appendChild(b);
    });
}

function renderOverview(data, song) {
    currentSong = song;
    chordTimeline = data.timeline || [];
    mainChords = data.main_chords || [];
    songDuration = 0;
    drillChord = null;
    audioPlayer.src = data.audio_url;
    audioPlayer.load();

    songTitle.textContent = song.title || 'Your song';
    songArtist.textContent = song.artist || '';
    playerTitle.textContent = song.title || 'Your song';
    playerArtist.textContent = song.artist || '';
    setArt(songArt, song.thumbnail);
    setArt(playerArt, song.thumbnail);

    const totals = {};
    chordTimeline.forEach(seg => { totals[seg.chord] = (totals[seg.chord] || 0) + (seg.end - seg.start); });
    const sum = Object.values(totals).reduce((a, b) => a + b, 0) || 1;

    chordGrid.replaceChildren();
    chordGrid.style.setProperty('--cards', String(Math.max(1, mainChords.length)));
    if (!mainChords.length) {
        const p = document.createElement('p');
        p.className = 'empty-note';
        p.textContent = 'No clear chords turned up in this preview. Try another song.';
        chordGrid.appendChild(p);
    }
    mainChords.forEach(chord => {
        chordGrid.appendChild(chordCard(chord, Math.round((totals[chord] || 0) / sum * 100)));
    });

    progressionEl.replaceChildren();
    chordTimeline.slice(0, 18).forEach(seg => {
        const li = document.createElement('li');
        li.className = `step chord ${isMinor(seg.chord) ? 'minor' : 'major'}`;
        setChord(li, seg.chord);
        li.title = `${formatTime(seg.start)}–${formatTime(seg.end)}`;
        progressionEl.appendChild(li);
    });

    markPresets(activePreset);
    activePreset = null;
    renderDrillPicker();

    hide(searchSection);
    showOnly(overviewSection);
    setStage('chords');
    markRowStarts();
    scrollToPanel();
}

let lastActiveIdx = -2;
let lastNextIdx = -2;
let lastSecond = -1;
let lastPaused = null;

// A block whose label overflows drops to the smaller size, and loses the
// label only when even that does not fit.
function labelBlocks() {
    timelineTrack.querySelectorAll('.chord-block').forEach(b => {
        b.classList.remove('tight', 'bare');
        if (b.scrollWidth <= b.clientWidth) return;
        b.classList.add('tight');
        if (b.scrollWidth <= b.clientWidth) return;
        b.classList.replace('tight', 'bare');
    });
}

// Mark the first chip of each wrapped row so no row starts with a chevron.
function markRowStarts() {
    if (!progressionEl.offsetParent) return;
    let lastTop = null;
    progressionEl.querySelectorAll('.step').forEach(step => {
        const top = step.offsetTop;
        step.classList.toggle('row-start', lastTop !== null && top > lastTop + 4);
        lastTop = top;
    });
}

function renderTimeline() {
    timelineTrack.querySelectorAll('.chord-block').forEach(el => el.remove());
    trackScale.replaceChildren();
    lastActiveIdx = -2;
    lastNextIdx = -2;
    lastSecond = -1;
    lastPaused = null;
    if (!chordTimeline.length || !songDuration) return;
    timelineTrack.setAttribute('aria-valuemax', String(Math.round(songDuration)));
    chordTimeline.forEach((seg, i) => {
        const block = document.createElement('div');
        block.className = `chord-block chord ${isMinor(seg.chord) ? 'minor' : 'major'}`;
        block.dataset.index = i;
        block.style.left  = `${(seg.start / songDuration) * 100}%`;
        block.style.width = `${((seg.end - seg.start) / songDuration) * 100}%`;
        block.title = `${seg.chord} · ${formatTime(seg.start)}`;
        setChord(block, seg.chord);
        if (i === chordTimeline.length - 1) block.classList.add('end');
        timelineTrack.insertBefore(block, playhead);
    });
    labelBlocks();
    // A printed scale under the track: a tick a second, a label every five.
    for (let s = 0; s <= Math.floor(songDuration); s++) {
        const tick = document.createElement('span');
        tick.className = 'tick' + (s % 5 === 0 ? ' major' : '') + (s % 10 === 0 ? ' ten' : '');
        tick.style.left = `${(s / songDuration) * 100}%`;
        if (s % 5 === 0) tick.dataset.label = formatTime(s);
        trackScale.appendChild(tick);
    }
}

let resizeTimer = null;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { labelBlocks(); markRowStarts(); }, 150);
});

function updatePlayhead() {
    if (stage !== 'play') { animFrame = null; return; }
    const t = audioPlayer.currentTime || 0;
    if (songDuration > 0) {
        playhead.style.left = `${Math.min((t / songDuration) * 100, 100)}%`;
        const sec = Math.floor(t);
        if (sec !== lastSecond) {
            lastSecond = sec;
            timeDisplay.textContent = `${formatTime(t)} / ${formatTime(songDuration)}`;
            timelineTrack.setAttribute('aria-valuenow', String(sec));
            timelineTrack.setAttribute('aria-valuetext', `${formatTime(t)} of ${formatTime(songDuration)}`);
        }
    }
    const idx = chordAt(t);
    if (idx !== lastActiveIdx) {
        timelineTrack.querySelector('.chord-block.active')?.classList.remove('active');
        if (idx >= 0) timelineTrack.querySelector(`[data-index="${idx}"]`)?.classList.add('active');
        setChord(nowChordEl, idx >= 0 ? chordTimeline[idx].chord : '');
        lastActiveIdx = idx;
    }
    const nextIdx = idx >= 0 ? idx + 1 : chordTimeline.findIndex(s => s.start > t);
    const next = nextIdx >= 0 && nextIdx < chordTimeline.length ? chordTimeline[nextIdx] : null;
    if (nextIdx !== lastNextIdx) {
        setChord(nextChordEl, next ? next.chord : '');
        lastNextIdx = nextIdx;
    }
    nextInEl.textContent = next ? `in ${Math.max(0, next.start - t).toFixed(1)}s` : '';

    updateTarget();
    if (audioPlayer.paused !== lastPaused) {
        lastPaused = audioPlayer.paused;
        playPauseBtn.classList.toggle('playing', !lastPaused);
        playPauseBtn.setAttribute('aria-label', lastPaused ? 'Play' : 'Pause');
    }
    animFrame = requestAnimationFrame(updatePlayhead);
}

// ------------------------------------------------------------------ practice
function setMode(newMode, chord) {
    mode = newMode;
    if (chord) drillChord = chord;
    if (mode === 'drill' && !drillChord) drillChord = mainChords[0] || null;
    modeButtons.forEach(b => {
        const on = b.dataset.mode === mode;
        b.classList.toggle('active', on);
        b.setAttribute('aria-pressed', String(on));
    });
    targetLabel.textContent = mode === 'drill' ? 'Drill target' : 'Song chord now';
    drillPicker.classList.toggle('hidden', mode !== 'drill' || !mainChords.length);
    drillPicker.querySelectorAll('.drill-chip').forEach(b => {
        const on = mode === 'drill' && b.dataset.chord === drillChord;
        b.classList.toggle('active', on);
        b.setAttribute('aria-pressed', String(on));
    });
    resetStats();
    updateTarget(true);
}

function updateTarget(force = false) {
    let t = '--';
    if (mode === 'drill') {
        t = drillChord || '--';
    } else if (!audioPlayer.paused) {
        const idx = chordAt(audioPlayer.currentTime || 0);
        t = idx >= 0 ? chordTimeline[idx].chord : '--';
    } else {
        const idx = chordAt(audioPlayer.currentTime || 0);
        t = idx >= 0 ? chordTimeline[idx].chord : (chordTimeline[0]?.chord || '--');
    }
    if (t !== targetChord || force) {
        targetChord = t;
        setChord(targetChordEl, t);
        targetDiagram.innerHTML = t !== '--' ? chordDiagramSVG(t) : '';
        checkMatch();
    }
}

function checkMatch() {
    const match = isListening && currentChord === targetChord && currentChord !== '--';
    playerBox.classList.toggle('match', match);
    targetBox.classList.toggle('match', match);
    matchBadge.classList.toggle('match', match);
    matchText.textContent = !isListening ? 'Mic off' : (match ? 'On the chord' : 'Listening');
}

function setGauge(scaleEl, pct) {
    const idle = pct == null || !isFinite(pct);
    scaleEl.classList.toggle('idle', idle);
    if (!idle) scaleEl.style.setProperty('--v', Math.max(0, Math.min(100, pct)).toFixed(1));
}

function resetStats() {
    stats = { total: 0, hits: 0, streak: 0, best: 0 };
    accuracyVal.textContent = '–';
    streakVal.textContent = '–';
    setGauge(accuracyScale, null);
}

function scoreSample(chord) {
    // Only score while the player is making sound against a real target.
    if (chord === '--' || targetChord === '--') return;
    stats.total += 1;
    if (chord === targetChord) {
        stats.hits += 1;
        stats.streak += LIVE_STEP_SEC;
        stats.best = Math.max(stats.best, stats.streak);
    } else {
        stats.streak = 0;
    }
    const pct = (stats.hits / stats.total) * 100;
    accuracyVal.textContent = `${Math.round(pct)}%`;
    setGauge(accuracyScale, pct);
    streakVal.textContent = `${stats.best.toFixed(1)}s`;
}

// Show the player. Callers start playback themselves, inside the click that
// asked for it, because browsers only allow play() from a user gesture.
function openPractice(newMode, chord) {
    showOnly(timelineSection);
    hide(searchSection);
    setStage('play');
    setMode(newMode, chord);

    const ready = () => {
        songDuration = audioPlayer.duration || 0;
        renderTimeline();
        if (!animFrame) animFrame = requestAnimationFrame(updatePlayhead);
    };
    if (audioPlayer.readyState >= 1) ready();
    else audioPlayer.addEventListener('loadedmetadata', ready, { once: true });
    if (!animFrame) animFrame = requestAnimationFrame(updatePlayhead);
    scrollToPanel();
}

function startPlayAlong() {
    audioPlayer.play().catch(err => console.warn('Playback did not start:', err));
    morph(() => openPractice('song'));
}

async function goChords() {
    audioPlayer.pause();
    await stopListening();
    morph(() => {
        hide(searchSection);
        showOnly(overviewSection);
        setStage('chords');
        markRowStarts();
        scrollToPanel();
    });
}

async function goSearch() {
    audioPlayer.pause();
    await stopListening();
    morph(() => {
        showOnly();
        show(searchSection);
        if (lastResults.length) show(resultsSection);
        setStage('search');
        scrollToPanel();
        if (finePointer.matches) songInput.focus({ preventScroll: true });
    });
}

// ------------------------------------------------------------------ microphone
function downsample(buf, ratio) {
    const outLen = Math.floor(buf.length / ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
        const start = Math.floor(i * ratio);
        const end = Math.min(buf.length, Math.floor((i + 1) * ratio));
        let sum = 0;
        for (let j = start; j < end; j++) sum += buf[j];
        out[i] = sum / Math.max(1, end - start);
    }
    return out;
}

async function startMicCapture() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showError("This browser can't reach a microphone here. It needs a secure (https) page.");
        return false;
    }
    try {
        // Voice processing (echo cancellation, noise suppression, auto gain)
        // treats a sustained chord as noise and eats it, so turn it off.
        micStream = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
        });
    } catch (err) {
        showError(err && err.name === 'NotAllowedError'
            ? 'Microphone access was blocked. Allow it in your browser’s site settings, then flip the switch again.'
            : `Couldn't open the microphone (${err && err.name ? err.name : 'unknown error'}).`);
        return false;
    }

    const Ctx = window.AudioContext || window.webkitAudioContext;
    let source;
    try {
        audioCtx = new Ctx({ sampleRate: MIC_SAMPLE_RATE });
        source = audioCtx.createMediaStreamSource(micStream);
    } catch (e) {
        // Firefox cannot connect a mic to a context at a different rate:
        // run at the device rate and downsample before sending.
        if (audioCtx) audioCtx.close();
        audioCtx = new Ctx();
        source = audioCtx.createMediaStreamSource(micStream);
    }
    await audioCtx.resume();
    const ratio = audioCtx.sampleRate / MIC_SAMPLE_RATE;

    micProcessor = audioCtx.createScriptProcessor(4096, 1, 1);
    micProcessor.onaudioprocess = (e) => {
        if (!isListening || !ws || ws.readyState !== WebSocket.OPEN) return;
        const input = e.inputBuffer.getChannelData(0);
        const pcm = Math.abs(ratio - 1) < 1e-3 ? new Float32Array(input) : downsample(input, ratio);
        ws.send(pcm.buffer);
    };
    micSink = audioCtx.createGain();
    micSink.gain.value = 0;  // keep the processor running without echoing the mic
    source.connect(micProcessor);
    micProcessor.connect(micSink);
    micSink.connect(audioCtx.destination);
    return true;
}

function stopMicCapture() {
    if (micProcessor) { micProcessor.disconnect(); micProcessor.onaudioprocess = null; micProcessor = null; }
    if (micSink)      { micSink.disconnect(); micSink = null; }
    if (audioCtx)     { audioCtx.close(); audioCtx = null; }
    if (micStream)    { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
}

function setListeningUI() {
    const ready = ws && ws.readyState === WebSocket.OPEN;
    micToggleBtn.setAttribute('aria-checked', String(isListening));
    micState.textContent = isListening ? 'Listening' : 'Standby';
    micAction.textContent = isListening ? 'Flip to stop' : (ready ? 'Flip to listen' : 'Waiting for the server');
    statusIndicator.classList.toggle('live', isListening);
    connectionStatus.textContent = isListening ? 'Listening…' : (ready ? 'Ready' : 'Connecting…');
    if (!isListening) {
        currentChord = '--';
        setChord(currentChordEl, '');
        confidenceVal.textContent = '–';
        setGauge(confidenceScale, null);
        volumeBar.style.width = '0%';
        checkMatch();
    }
}

async function stopListening() {
    if (isListening && ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ command: 'STOP' }));
    isListening = false;
    stopMicCapture();
    setListeningUI();
}

micToggleBtn.addEventListener('click', async () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (isListening) { await stopListening(); return; }
    micToggleBtn.disabled = true;
    micAction.textContent = 'Asking for the mic…';
    const ok = await startMicCapture();
    micToggleBtn.disabled = false;
    if (!ok) { setListeningUI(); return; }
    isListening = true;
    resetStats();
    ws.send(JSON.stringify({ command: 'START' }));
    setListeningUI();
});

// ------------------------------------------------------------------ websocket
function connect() {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${window.location.host}/ws`);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        statusIndicator.classList.remove('down');
        statusIndicator.classList.add('ready');
        micToggleBtn.disabled = false;
        setListeningUI();
    };

    ws.onclose = () => {
        statusIndicator.classList.remove('ready', 'live');
        statusIndicator.classList.add('down');
        micToggleBtn.disabled = true;
        if (isListening) { isListening = false; stopMicCapture(); }
        setListeningUI();
        connectionStatus.textContent = 'Reconnecting…';
        setTimeout(connect, 2000);
    };

    ws.onerror = () => ws.close();

    ws.onmessage = (event) => {
        if (!isListening) return;
        let data;
        try { data = JSON.parse(event.data); } catch { return; }
        volumeBar.style.width = `${Math.round(Math.min(1, Math.sqrt(data.volume || 0) * 4) * 100)}%`;
        if (!data.confident) return;
        const heard = data.chord !== '--';
        confidenceVal.textContent = heard ? `${Math.round(data.confidence * 100)}%` : '–';
        setGauge(confidenceScale, heard ? data.confidence * 100 : null);
        if (data.chord !== currentChord) {
            currentChord = data.chord;
            setChord(currentChordEl, currentChord);
            playerBox.classList.remove('bump');
            void playerBox.offsetWidth;
            playerBox.classList.add('bump');
        }
        scoreSample(data.chord);
        checkMatch();
    };
}

// ------------------------------------------------------------------ search
async function runSearch(rawQuery, { autoPick = false, source = null } = {}) {
    const query = (rawQuery || '').trim();
    if (!query || busy) { source?.classList.remove('loading'); return; }
    clearError();
    hideAutocomplete();
    setBusy(true);
    hide(resultsSection);
    startLoading(['Searching the iTunes catalogue']);
    try {
        const res = await fetch('/api/suggest?q=' + encodeURIComponent(query));
        if (!res.ok) throw new Error(`Search failed (${res.status}). Try again in a moment.`);
        const songs = await res.json();
        if (!songs.length) {
            showError(`No songs found for “${query}”. Try the artist and title together.`);
            return;
        }
        if (autoPick) {
            await analyzeSong(songs[0], { keepBusy: true, source });
        } else {
            renderResults(songs, query);
        }
    } catch (err) {
        showError(err.message || 'Search failed. Check your connection and try again.');
    } finally {
        stopLoading();
        setBusy(false);
        source?.classList.remove('loading');
    }
}

async function analyzeSong(song, { keepBusy = false, source = null } = {}) {
    if (busy && !keepBusy) return;
    clearError();
    setBusy(true);
    show(searchSection);
    source?.classList.add('loading');
    startLoading([
        'Fetching the 30-second preview',
        'Working out the notes',
        'Naming the chords',
    ]);
    audioPlayer.pause();
    try {
        const data = song.audio_url
            ? await postJSON('/api/from-url', { url: song.audio_url })
            : await postJSON('/api/search', { query: song.query || song.title, video_id: song.videoId });
        stopLoading();
        source?.classList.remove('loading');
        morph(() => renderOverview(data, song), source);
    } catch (err) {
        showError(err.message || 'Could not analyse that song. Try another.');
        source?.classList.remove('loading');
        activePreset = null;
    } finally {
        stopLoading();
        if (!keepBusy) setBusy(false);
    }
}

searchForm.addEventListener('submit', (e) => {
    e.preventDefault();
    runSearch(songInput.value);
});

// ------------------------------------------------------------------ autocomplete
let acTimer = null;
let acIndex = -1;
function hideAutocomplete() {
    autocompleteList.classList.add('hidden');
    songInput.setAttribute('aria-expanded', 'false');
    songInput.removeAttribute('aria-activedescendant');
    acIndex = -1;
}
function renderAutocomplete(items) {
    autocompleteList.replaceChildren();
    acIndex = -1;
    if (!items.length) { hideAutocomplete(); return; }
    items.forEach((text, i) => {
        const li = document.createElement('li');
        li.id = `ac-option-${i}`;
        li.textContent = text;
        li.setAttribute('role', 'option');
        li.addEventListener('mousedown', (e) => {
            e.preventDefault();
            songInput.value = text;
            runSearch(text);
        });
        autocompleteList.appendChild(li);
    });
    autocompleteList.classList.remove('hidden');
    songInput.setAttribute('aria-expanded', 'true');
}
songInput.addEventListener('input', () => {
    clearTimeout(acTimer);
    const q = songInput.value.trim();
    if (q.length < 3) { hideAutocomplete(); return; }
    acTimer = setTimeout(async () => {
        try {
            const res = await fetch('/api/autocomplete?q=' + encodeURIComponent(q));
            const items = await res.json();
            if (songInput.value.trim() === q && document.activeElement === songInput) renderAutocomplete(items);
        } catch (err) { console.error(err); }
    }, 400);
});
songInput.addEventListener('keydown', (e) => {
    const items = autocompleteList.querySelectorAll('li');
    if (autocompleteList.classList.contains('hidden') || !items.length) return;
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        acIndex = (acIndex + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
        items.forEach((li, i) => li.classList.toggle('active', i === acIndex));
        songInput.setAttribute('aria-activedescendant', items[acIndex].id);
    } else if (e.key === 'Enter' && acIndex >= 0) {
        e.preventDefault();
        songInput.value = items[acIndex].textContent;
        runSearch(songInput.value);
    } else if (e.key === 'Escape') {
        hideAutocomplete();
    }
});
songInput.addEventListener('blur', () => setTimeout(hideAutocomplete, 120));

// ------------------------------------------------------------------ presets
async function loadExamples() {
    let examples = FALLBACK_EXAMPLES;
    try {
        const res = await fetch('/api/examples');
        if (res.ok) examples = await res.json();
    } catch { /* keep the fallback list */ }
    exampleChips.replaceChildren();
    examples.slice(0, 9).forEach((ex, i) => {
        const n = String(i + 1);
        const key = document.createElement('button');
        key.type = 'button';
        key.className = 'key preset chip';
        key.dataset.n = n;
        key.setAttribute('aria-keyshortcuts', n);
        key.innerHTML = '<span class="preset-top"><span class="lamp" aria-hidden="true"></span><span class="preset-n" aria-hidden="true"></span></span><span class="preset-title"></span><span class="preset-artist"></span>';
        key.querySelector('.preset-n').textContent = n;
        key.querySelector('.preset-title').textContent = ex.title;
        key.querySelector('.preset-artist').textContent = ex.artist;
        key.disabled = busy;
        key.addEventListener('click', () => {
            if (busy) return;
            activePreset = key;
            key.classList.add('loading');
            songInput.value = `${ex.artist} ${ex.title}`;
            runSearch(ex.query, { autoPick: true, source: key });
        });
        exampleChips.appendChild(key);
    });
}

// ------------------------------------------------------------------ player controls
timelineTrack.addEventListener('click', (e) => {
    const rect = timelineTrack.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    if (songDuration > 0) audioPlayer.currentTime = Math.max(0, Math.min(1, pct)) * songDuration;
});

timelineTrack.addEventListener('keydown', (e) => {
    if (!songDuration) return;
    const step = e.shiftKey ? 10 : 5;
    let t = audioPlayer.currentTime || 0;
    if (e.key === 'ArrowRight' || e.key === 'ArrowUp') t += step;
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') t -= step;
    else if (e.key === 'Home') t = 0;
    else if (e.key === 'End') t = songDuration - 0.1;
    else return;
    e.preventDefault();
    audioPlayer.currentTime = Math.max(0, Math.min(songDuration, t));
});

playPauseBtn.addEventListener('click', () => {
    if (audioPlayer.paused) {
        if (mode === 'drill') setMode('song');
        audioPlayer.play().catch(err => console.warn(err));
    } else {
        audioPlayer.pause();
    }
});

modeButtons.forEach(btn => btn.addEventListener('click', () => {
    setMode(btn.dataset.mode);
    if (btn.dataset.mode === 'drill') audioPlayer.pause();
}));

document.addEventListener('keydown', (e) => {
    if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
    const tag = (e.target.tagName || '').toLowerCase();
    const typing = tag === 'input' || tag === 'textarea' || e.target.isContentEditable;
    if (!timelineSection.classList.contains('hidden')) {
        if (e.code === 'Space' && !typing && tag !== 'button') {
            e.preventDefault();
            playPauseBtn.click();
        }
        return;
    }
    // Number keys press the matching preset.
    if (!searchSection.classList.contains('hidden') && !typing && /^[1-9]$/.test(e.key)) {
        const key = exampleChips.querySelector(`.preset[data-n="${e.key}"]`);
        if (key && !key.disabled) {
            e.preventDefault();
            key.click();
        }
    }
});

audioPlayer.addEventListener('error', () => {
    if (!audioPlayer.getAttribute('src')) return;
    const codes = {
        1: 'loading was stopped',
        2: 'a network error interrupted it',
        3: 'it could not be decoded',
        4: 'this browser does not support its format',
    };
    const err = audioPlayer.error;
    showError(`The audio couldn't play: ${(err && codes[err.code]) || 'unknown error'}.`);
});

practiceBtn.addEventListener('click', startPlayAlong);
backBtn.addEventListener('click', goChords);
newSearchBtn.addEventListener('click', goSearch);

stateButtons.forEach(btn => btn.addEventListener('click', () => {
    const go = btn.dataset.go;
    if (go === stage) return;
    if (go === 'search') goSearch();
    else if (go === 'chords') goChords();
    else if (go === 'play') startPlayAlong();
}));

// ------------------------------------------------------------------ boot
micToggleBtn.disabled = true;
setStage('search');
loadExamples();
connect();
