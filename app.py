import asyncio
import os
import glob
import numpy as np
import librosa
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import yt_dlp

import imageio_ffmpeg
import ytmusicapi

import tensorflow as tf
from transformer_model import (
    build_transformer, CHORD_CLASSES, SEQUENCE_LENGTH, positional_encoding
)

ytmusic = ytmusicapi.YTMusic()

# ==========================================
# Load Transformer Model (if trained)
# ==========================================
MODEL_PATH = "transformer_chord_model.keras"
TRANSFORMER_MODEL = None
_MODEL_LOADED = False

def load_transformer_model():
    global TRANSFORMER_MODEL, _MODEL_LOADED
    if _MODEL_LOADED:
        return
    _MODEL_LOADED = True
    if not os.path.exists(MODEL_PATH):
        print("[INFO] No trained model found. Using Chroma template fallback.")
        print("       Run: python build_dataset.py --songs 300  then  python train.py")
        return
    try:
        print(f"[INFO] Loading Transformer model from {MODEL_PATH}...")
        TRANSFORMER_MODEL = build_transformer()
        TRANSFORMER_MODEL.load_weights(MODEL_PATH)
        print("[INFO] Transformer model loaded successfully.")
    except Exception as e:
        print(f"[WARN] Could not load model: {e}. Falling back to Chroma templates.")
        TRANSFORMER_MODEL = None

load_transformer_model()

class SearchQuery(BaseModel):
    query: str
    video_id: Optional[str] = None

# ==========================================
# Audio Config
# ==========================================
SR = 22050
DURATION = 1.0
CHUNK_SAMPLES = int(SR * DURATION)
HOP_LENGTH = 512
VOLUME_THRESHOLD = 0.001

# ==========================================
# Guitar-aware Chord Templates
# ==========================================
# Weights approximate typical guitar voicings:
#   - Root heavier (often doubled: bass string + octave on higher strings)
#   - 3rd slightly lighter (sometimes omitted or sounded weaker)
#   - 5th standard (doubled in barre chords)
# Power / sus chords included because guitarists play them constantly
# and they currently get mis-labelled as maj/min.
PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def create_chord_templates():
    templates = {}
    for i, root in enumerate(PITCH_CLASSES):
        # Major triad (guitar-weighted)
        maj = np.zeros(12)
        maj[i]            = 1.3
        maj[(i + 4) % 12] = 0.9
        maj[(i + 7) % 12] = 1.0
        templates[f"{root}"] = maj / np.linalg.norm(maj)

        # Minor triad
        minor = np.zeros(12)
        minor[i]            = 1.3
        minor[(i + 3) % 12] = 0.9
        minor[(i + 7) % 12] = 1.0
        templates[f"{root}m"] = minor / np.linalg.norm(minor)

        # Power chord (root + 5th, no 3rd) — rock/punk/metal staple
        power = np.zeros(12)
        power[i]            = 1.4
        power[(i + 7) % 12] = 1.0
        templates[f"{root}5"] = power / np.linalg.norm(power)

        # sus2 (root + 2nd + 5th)
        sus2 = np.zeros(12)
        sus2[i]            = 1.2
        sus2[(i + 2) % 12] = 0.9
        sus2[(i + 7) % 12] = 1.0
        templates[f"{root}sus2"] = sus2 / np.linalg.norm(sus2)

        # sus4 (root + 4th + 5th)
        sus4 = np.zeros(12)
        sus4[i]            = 1.2
        sus4[(i + 5) % 12] = 0.9
        sus4[(i + 7) % 12] = 1.0
        templates[f"{root}sus4"] = sus4 / np.linalg.norm(sus4)
    return templates

print("Initializing Chroma Chord Templates...")
CHORD_TEMPLATES = create_chord_templates()

# ==========================================
# Audio Processing — unified feature pipeline
# Mic path now matches offline extraction: HPSS + chroma_cqt.
# ==========================================
def process_audio_chunk(audio_data):
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]

    rms = np.sqrt(np.mean(audio_data**2))
    if rms < VOLUME_THRESHOLD:
        return {"chord": "--", "confidence": 0.0, "volume": float(rms)}

    # Harmonic-percussive separation strips pick/strum transients before analysis.
    # margin=3 matches extract_chords_from_file() for consistency.
    try:
        y_h = librosa.effects.harmonic(audio_data, margin=3)
    except Exception:
        y_h = audio_data

    chroma = librosa.feature.chroma_cqt(y=y_h, sr=SR, hop_length=HOP_LENGTH // 2)
    chroma_vector = np.mean(chroma, axis=1)

    norm = np.linalg.norm(chroma_vector)
    if norm == 0:
        return {"chord": "--", "confidence": 0.0, "volume": float(rms)}
    chroma_vector = chroma_vector / norm

    best_chord, best_score = "--", 0.0
    for name, template in CHORD_TEMPLATES.items():
        score = float(np.dot(chroma_vector, template))
        if score > best_score:
            best_score = score
            best_chord = name

    return {"chord": best_chord, "confidence": float(best_score), "volume": float(rms)}

# ==========================================
# YouTube Audio Downloader (yt-dlp)
# ==========================================
def _ytmusic_url(query):
    """
    Use ytmusicapi to find the best matching song on YouTube Music.
    Returns a youtube.com watch URL, or None if nothing found.
    Retries on SSL/network errors; falls through to yt-dlp's caller search if it fails.
    """
    import time as _time
    for filter_type in ('songs', 'videos'):
        for attempt in range(3):
            try:
                results = ytmusic.search(query, filter=filter_type, limit=3)
                for r in results:
                    vid = r.get('videoId')
                    if vid:
                        title = r.get('title', '')
                        artist = ''
                        artists = r.get('artists', [])
                        if artists:
                            artist = artists[0].get('name', '')
                        print(f"[YTMusic] Found ({filter_type}): {artist} — {title}  [{vid}]")
                        return f"https://www.youtube.com/watch?v={vid}"
                break  # got results but no videoIds — try next filter
            except Exception as e:
                print(f"[YTMusic] {filter_type} attempt {attempt+1}/3: {e}")
                _time.sleep(0.5 * (attempt + 1))
    return None


def download_audio(query, video_id=None):
    import time as _time, uuid as _uuid
    print(f"\nSearching YouTube Music for: '{query}'...")
    os.makedirs("static", exist_ok=True)

    # Use a UUID so filenames are always unique — avoids Windows file-lock
    # collisions when the browser is still holding the previous .wav open.
    # Old files are cleaned up here on a best-effort basis; failures are
    # ignored because Windows won't release a lock until the browser unloads.
    for f in glob.glob("static/downloaded_*.wav"):
        try:
            os.remove(f)
            print(f"[Cleanup] Deleted {f}")
        except OSError:
            print(f"[Cleanup] Skipped (locked): {f}")

    out_base = f"static/downloaded_{_uuid.uuid4().hex[:12]}"

    # If a specific video ID was chosen by the user, use it directly;
    # otherwise resolve via YouTube Music search, then fall back to generic search
    if video_id:
        url = f"https://www.youtube.com/watch?v={video_id}"
    else:
        url = _ytmusic_url(query) or f"ytsearch1:{query}"
    print(f"[Download] URL: {url}")

    # Use system ffmpeg in Docker (apt-installed), fall back to imageio bundle locally
    import shutil, subprocess
    ffmpeg_path = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()

    # Optional cookies file (HF admin can set YT_COOKIES_FILE in Space secrets)
    cookies_file = os.environ.get("YT_COOKIES_FILE")
    if cookies_file and not os.path.exists(cookies_file):
        cookies_file = None

    success = False
    title = "Unknown"

    # ── Strategy 1: yt-dlp with rotating player clients ─────────────────
    client_order = [
        ['tv_simply'], ['mediaconnect'], ['mweb'], ['ios'], ['tv'], ['web'],
    ]
    base_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{out_base}.%(ext)s',
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'ffmpeg_location': ffmpeg_path,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        },
        'socket_timeout': 30,
        'retries': 2,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }]
    }
    if cookies_file:
        base_opts['cookiefile'] = cookies_file

    for client in client_order:
        opts = dict(base_opts)
        opts['extractor_args'] = {'youtube': {'player_client': client}}
        print(f"[Download] yt-dlp player_client={client[0]}...")
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info:
                    if 'entries' in info and len(info['entries']) > 0:
                        title = info['entries'][0].get('title', title)
                    elif 'title' in info:
                        title = info.get('title', title)
                success = True
                print(f"[Download] OK via yt-dlp/{client[0]}: {title}")
                break
        except Exception as e:
            err_msg = str(e).lower()
            transient = (
                'sign in' in err_msg or 'bot' in err_msg or 'confirm' in err_msg
                or 'ssl' in err_msg or 'eof' in err_msg
                or 'unable to download' in err_msg or 'http error' in err_msg
                or 'timeout' in err_msg or 'connection' in err_msg
            )
            if transient:
                print(f"[Download] {client[0]} transient error, trying next client…")
                continue
            print(f"[Download] {client[0]} failed (non-transient): {e}")
            break

    # ── Strategy 2: pytubefix fallback (different YouTube API path) ─────
    if not success:
        print("[Download] yt-dlp exhausted, trying pytubefix...")
        try:
            from pytubefix import YouTube
            # Resolve URL → video id; pytubefix needs a watch URL
            target_url = url if url.startswith('http') else None
            if not target_url:
                # url was an ytsearch query — need to extract via yt-dlp's search
                with yt_dlp.YoutubeDL({'quiet': True, 'extract_flat': True}) as ydl:
                    s = ydl.extract_info(url, download=False)
                    if s and s.get('entries'):
                        target_url = f"https://www.youtube.com/watch?v={s['entries'][0]['id']}"
            if target_url:
                yt = YouTube(target_url)
                stream = yt.streams.get_audio_only()
                if stream is None:
                    raise RuntimeError("No audio stream available")
                tmp_name = f"{os.path.basename(out_base)}.m4a"
                stream.download(output_path="static", filename=tmp_name)
                tmp_path = f"static/{tmp_name}"
                wav_path = f"{out_base}.wav"
                subprocess.run(
                    [ffmpeg_path, "-y", "-i", tmp_path, "-ar", str(SR), wav_path],
                    check=True, capture_output=True,
                )
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                title = yt.title
                success = True
                print(f"[Download] OK via pytubefix: {title}")
        except Exception as e:
            print(f"[Download] pytubefix failed: {e}")

    if not success:
        # Friendly message for end users — no implementation details
        raise RuntimeError(
            "Couldn't fetch this song right now. "
            "YouTube is rate-limiting our server — please try a different song or try again in a moment."
        )

    downloaded = glob.glob(f"{out_base}.*")
    if not downloaded:
        print("Error: audio file was not saved.")
        return None

    return os.path.basename(downloaded[0])

# ==========================================
# FastAPI Server
# ==========================================
app = FastAPI()
active_connections: List[WebSocket] = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Browser-based mic detection.
    The client sends:
      - text '{"command":"START"}' / '{"command":"STOP"}' to toggle listening
      - binary frames of float32 PCM audio captured via getUserMedia
    The server processes each chunk and replies with JSON chord detection results.
    """
    await websocket.accept()
    active_connections.append(websocket)
    audio_buffer = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
    is_listening = False
    import time as _time
    last_predict_time = 0.0

    try:
        while True:
            msg = await websocket.receive()

            # Text message: START / STOP toggle
            if "text" in msg:
                data = msg["text"]
                if "START" in data:
                    is_listening = True
                    audio_buffer.fill(0)
                    last_predict_time = 0.0
                    print("Listening started (browser mic).")
                elif "STOP" in data:
                    is_listening = False
                    audio_buffer.fill(0)
                    print("Listening stopped.")
                continue

            # Binary message: raw float32 PCM from browser mic
            if "bytes" in msg and is_listening:
                pcm = np.frombuffer(msg["bytes"], dtype=np.float32)
                if len(pcm) == 0:
                    continue

                # Rolling buffer — append new audio, shift old out
                new_len = min(len(pcm), CHUNK_SAMPLES)
                audio_buffer[:] = np.roll(audio_buffer, -new_len)
                audio_buffer[-new_len:] = pcm[-new_len:]

                result = process_audio_chunk(audio_buffer.copy())
                if result:
                    now = _time.time()
                    if result['confidence'] > 0.60 and (now - last_predict_time > 1.0):
                        print(f"Detected: {result['chord']} ({result['confidence']:.2f})", flush=True)
                        await websocket.send_json(result)
                        last_predict_time = now

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)



app.mount("/static", StaticFiles(directory="static"), name="static")

# ==========================================
# REST API Endpoints
# ==========================================
def extract_chords_from_file(filepath):
    """
    Chord extraction pipeline for a full song.
    Uses Transformer model if trained; falls back to Chroma templates.
    Both paths apply HPSS + CQT for consistent preprocessing.
    """
    MAX_ANALYSIS_SEC = 210
    y, sr = librosa.load(filepath, sr=SR, duration=MAX_ANALYSIS_SEC)
    song_duration = float(len(y)) / SR

    y_harmonic = librosa.effects.harmonic(y, margin=3)

    FINE_HOP = HOP_LENGTH // 2
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=SR, hop_length=FINE_HOP)
    chroma_T = chroma.T
    T = chroma_T.shape[0]

    _, beat_frames = librosa.beat.beat_track(
        y=y_harmonic, sr=SR, hop_length=FINE_HOP, trim=False)
    beat_frames = np.asarray(beat_frames, dtype=int)
    beat_times  = librosa.frames_to_time(beat_frames, sr=SR, hop_length=FINE_HOP)
    beat_times  = np.append(beat_times, song_duration)

    tempo_bpm = (60.0 / float(np.median(np.diff(beat_times[:-1])))
                 if len(beat_times) > 2 else 100.0)
    print(f"[Beat] tempo={tempo_bpm:.1f} BPM, {len(beat_frames)} beats detected")

    sub_times = []
    for i in range(len(beat_times) - 1):
        sub_times.append(float(beat_times[i]))
        sub_times.append((float(beat_times[i]) + float(beat_times[i + 1])) / 2.0)
    sub_times.append(float(beat_times[-1]))
    sub_beat_times = np.array(sub_times)

    median_sub = float(np.median(np.diff(sub_beat_times))) if len(sub_beat_times) > 2 else 0.25
    max_snap   = median_sub * 0.55

    def nearest_beat(t):
        deltas = np.abs(sub_beat_times - t)
        idx    = int(np.argmin(deltas))
        return float(sub_beat_times[idx]) if deltas[idx] <= max_snap else float(t)

    if   tempo_bpm > 160: MIN_SEG = 0.25
    elif tempo_bpm > 120: MIN_SEG = 0.40
    elif tempo_bpm >  90: MIN_SEG = 0.60
    else:                 MIN_SEG = 0.80
    print(f"[Beat] MIN_SEG={MIN_SEG:.2f}s")

    WIN_FRAMES = SEQUENCE_LENGTH
    STRIDE     = WIN_FRAMES // 2
    secs_per_frame = FINE_HOP / SR

    if TRANSFORMER_MODEL is not None:
        starts = list(range(0, T - WIN_FRAMES + 1, STRIDE))
        if not starts:
            return []
        windows = np.array(
            [chroma_T[s : s + WIN_FRAMES] for s in starts],
            dtype=np.float32
        )
        probs        = TRANSFORMER_MODEL.predict(windows, verbose=0, batch_size=32)
        pred_classes = np.argmax(probs, axis=1)
        pred_confs   = np.max(probs, axis=1)
        raw = [
            (CHORD_CLASSES[pred_classes[i]],
             starts[i] * secs_per_frame,
             float(pred_confs[i]))
            for i in range(len(starts))
        ]
    else:
        sub_beat_frames = np.clip(
            np.round(sub_beat_times[:-1] * SR / FINE_HOP).astype(int), 0, T - 1
        )
        beat_chroma   = librosa.util.sync(chroma, sub_beat_frames, aggregate=np.median)
        beat_chroma_T = beat_chroma.T
        num_sub_beats = beat_chroma_T.shape[0]
        raw = []
        for i in range(num_sub_beats):
            win = beat_chroma_T[max(0, i - 1) : min(num_sub_beats, i + 2)]
            vec = np.median(win, axis=0)
            peak = np.max(vec)
            if peak <= 0:
                continue
            vec = np.where(vec > 0.38 * peak, vec, 0)
            norm = np.linalg.norm(vec)
            if norm == 0:
                continue
            vec = vec / norm
            best_chord, best_score = "--", 0.0
            for name, template in CHORD_TEMPLATES.items():
                score = float(np.dot(vec, template))
                if score > best_score:
                    best_score = score
                    best_chord = name
            if best_score >= 0.65:
                raw.append((best_chord, float(sub_beat_times[i]), best_score))

    timeline = []
    if raw:
        current_chord, current_start, _ = raw[0]
        for chord, t, conf in raw[1:]:
            if chord != current_chord:
                if current_chord != "--":
                    timeline.append({
                        "start": round(current_start, 3),
                        "end":   round(t, 3),
                        "chord": current_chord
                    })
                current_chord, current_start = chord, t
        if current_chord != "--":
            timeline.append({
                "start": round(current_start, 3),
                "end":   round(song_duration, 3),
                "chord": current_chord
            })

    if TRANSFORMER_MODEL is not None and timeline:
        snapped = []
        for seg in timeline:
            s = nearest_beat(seg["start"])
            e = nearest_beat(seg["end"])
            if e > s:
                snapped.append({"start": round(s, 3), "end": round(e, 3), "chord": seg["chord"]})
        timeline = snapped

    timeline = [s for s in timeline if (s["end"] - s["start"]) >= MIN_SEG]

    merged = []
    for seg in timeline:
        if merged and merged[-1]["chord"] == seg["chord"]:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg)
    timeline = merged

    return timeline


async def _build_song_response(filename: str):
    """Run extraction on a static-served file and assemble the JSON response."""
    loop = asyncio.get_running_loop()
    try:
        timeline = await loop.run_in_executor(
            None, extract_chords_from_file, f"static/{filename}"
        )
    except Exception as e:
        print(f"Chroma extraction error: {e}")
        raise HTTPException(status_code=500, detail="Chord extraction failed.")

    chord_dur = {}
    for seg in timeline:
        c = seg['chord']
        if c != '--':
            chord_dur[c] = chord_dur.get(c, 0) + (seg['end'] - seg['start'])
    total_dur = sum(chord_dur.values()) or 1.0
    main_chords = [
        c for c, d in sorted(chord_dur.items(), key=lambda x: -x[1])
        if d / total_dur >= 0.04
    ][:8]

    return {"audio_url": f"/static/{filename}", "timeline": timeline, "main_chords": main_chords}


@app.post("/api/search")
async def search_song(request: SearchQuery):
    loop = asyncio.get_running_loop()
    _q, _vid = request.query, request.video_id
    try:
        filename = await loop.run_in_executor(None, download_audio, _q, _vid)
    except Exception as exc:
        detail = str(exc)
        leak_markers = [
            'sign in', 'confirm you', 'not a bot',
            '[youtube]', 'yt-dlp', 'cookies', 'extractor',
            'download failed', 'ssl', 'eof', 'unable to',
            'http error', 'timeout', 'connection', 'pytubefix',
        ]
        if any(m in detail.lower() for m in leak_markers):
            detail = ("Couldn't fetch this song right now. "
                      "Please try a different song or try again in a moment.")
        raise HTTPException(status_code=400, detail=detail)
    if not filename:
        raise HTTPException(status_code=400, detail="Couldn't fetch this song. Please try another.")

    return await _build_song_response(filename)


@app.post("/api/upload")
async def upload_song(file: UploadFile = File(...)):
    """
    Accept an MP3/WAV/M4A/etc upload, save (and convert to wav), then run
    the same chord-extraction pipeline as /api/search.
    """
    import shutil, subprocess, uuid as _uuid
    os.makedirs("static", exist_ok=True)

    # Clean up old downloads same way as YouTube path
    for f in glob.glob("static/downloaded_*.wav"):
        try:
            os.remove(f)
        except OSError:
            pass

    src_ext = os.path.splitext(file.filename or "upload")[1].lower() or ".bin"
    uid = _uuid.uuid4().hex[:12]
    raw_path = f"static/upload_{uid}{src_ext}"
    wav_path = f"static/downloaded_{uid}.wav"

    # Stream the upload to disk
    with open(raw_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    # Convert to 22050 Hz mono wav (the format librosa.load expects)
    ffmpeg_path = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    try:
        subprocess.run(
            [ffmpeg_path, "-y", "-i", raw_path, "-ar", str(SR), "-ac", "1", wav_path],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        try: os.remove(raw_path)
        except OSError: pass
        raise HTTPException(
            status_code=400,
            detail="Couldn't read that audio file. Try MP3, WAV, or M4A."
        )

    try:
        os.remove(raw_path)
    except OSError:
        pass

    return await _build_song_response(os.path.basename(wav_path))

def _ytmusic_search_with_retry(query, limit=8, attempts=3):
    """ytmusicapi over HF cloud often hits SSL EOF — retry with backoff."""
    import time as _time
    last_err = None
    for i in range(attempts):
        try:
            return ytmusic.search(query, filter='songs', limit=limit)
        except Exception as e:
            last_err = e
            print(f"[YTMusic] search attempt {i+1}/{attempts} failed: {e}")
            _time.sleep(0.5 * (i + 1))
    raise last_err if last_err else RuntimeError("ytmusic search failed")


def _ytdlp_search_fallback(query, limit=8):
    """Fallback search using yt-dlp's flat extraction when ytmusicapi is down."""
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'skip_download': True,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        },
        'extractor_args': {'youtube': {'player_client': ['mediaconnect', 'tv', 'web']}},
    }
    suggestions = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
        if not info or 'entries' not in info:
            return []
        for r in info['entries']:
            vid = r.get('id')
            if not vid:
                continue
            thumb = ''
            thumbs = r.get('thumbnails') or []
            if thumbs:
                thumb = thumbs[-1].get('url', '')
            duration_sec = r.get('duration')
            duration_str = ''
            if duration_sec:
                m, s = divmod(int(duration_sec), 60)
                duration_str = f"{m}:{s:02d}"
            suggestions.append({
                'videoId': vid,
                'title':    r.get('title', ''),
                'artist':   r.get('uploader', '') or r.get('channel', ''),
                'duration': duration_str,
                'thumbnail': thumb,
            })
    return suggestions


@app.get("/api/suggest")
async def suggest_songs(q: str = ""):
    if not q:
        return []
    loop = asyncio.get_running_loop()

    def _search():
        try:
            results = _ytmusic_search_with_retry(q, limit=8)
            suggestions = []
            for r in results:
                vid = r.get('videoId')
                if not vid:
                    continue
                artists = r.get('artists') or []
                artist = artists[0].get('name', '') if artists else ''
                thumbnails = r.get('thumbnails') or []
                thumb = thumbnails[-1].get('url', '') if thumbnails else ''
                suggestions.append({
                    'videoId': vid,
                    'title':    r.get('title', ''),
                    'artist':   artist,
                    'duration': r.get('duration', ''),
                    'thumbnail': thumb,
                })
            if suggestions:
                return suggestions
        except Exception as e:
            print(f"[Suggest] ytmusic failed: {e}; falling back to yt-dlp search")

        # Fallback: yt-dlp's ytsearch
        try:
            return _ytdlp_search_fallback(q, limit=8)
        except Exception as e:
            print(f"[Suggest] yt-dlp fallback failed: {e}")
            return []

    return await loop.run_in_executor(None, _search)

@app.get("/api/autocomplete")
async def autocomplete(q: str = ""):
    if not q:
        return []
    loop = asyncio.get_running_loop()

    def _suggest():
        # ytmusicapi sometimes drops SSL on HF cloud — retry once, fail silently.
        import time as _time
        for attempt in range(2):
            try:
                return ytmusic.get_search_suggestions(q)
            except Exception as e:
                if attempt == 0:
                    _time.sleep(0.4)
                    continue
                # Final attempt failed — log once at debug level, return empty.
                print(f"[Autocomplete] suggestion fetch failed: {type(e).__name__}")
                return []
        return []

    return await loop.run_in_executor(None, _suggest)

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    os.makedirs("static", exist_ok=True)
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting FastAPI Server on port {port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port)
