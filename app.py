import asyncio
import html
import json
import os
import glob
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import numpy as np
import librosa
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import yt_dlp

import imageio_ffmpeg
import ytmusicapi

ytmusic = ytmusicapi.YTMusic()

class SearchQuery(BaseModel):
    query: str
    video_id: Optional[str] = None

class UrlQuery(BaseModel):
    url: str

SR = 22050
DURATION = 1.0
CHUNK_SAMPLES = int(SR * DURATION)
HOP_LENGTH = 512
VOLUME_THRESHOLD = 0.001

PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# 12 major then 12 minor — same order transformer_model.CHORD_NAMES uses.
CHORD_CLASSES = PITCH_CLASSES + [p + 'm' for p in PITCH_CLASSES]

# Offline analysis runs at half rate with a coarser hop than the mic path.
# chroma_cqt tops out at C8 (4186 Hz), well under the 5512 Hz Nyquist here, and
# 46 ms frames are still far finer than any chord change.
ANALYSIS_SR      = 11025
ANALYSIS_HOP     = 512
CQT_FMIN         = librosa.note_to_hz('C1')
CQT_OCTAVES      = 7
CQT_BINS_PER_OCT = 36
MAX_ANALYSIS_SEC = 210

# 2.32s window / 1.16s step, unchanged from the previous 200-frame @ 256/22050
# geometry so P_STAY stays tuned to the same stride.
WIN_SECONDS = 2.32

# Budget for the whole yt-dlp client walk, chosen to leave headroom under the
# ~180s HF Spaces gateway timeout so failures return JSON, not a 500 page.
DOWNLOAD_DEADLINE_SEC = 75

# Cloud containers frequently advertise IPv6 but have no working route to
# Google, which surfaces as [SSL: UNEXPECTED_EOF_WHILE_READING] partway through
# the handshake rather than as a clean connection error. Pinning to IPv4 is
# yt-dlp's --force-ipv4. Set YT_FORCE_IPV4=0 if a host needs v6.
FORCE_IPV4 = os.environ.get("YT_FORCE_IPV4", "1") != "0"
IPV4_OPTS = {'source_address': '0.0.0.0'} if FORCE_IPV4 else {}

# Jamendo is the search backend wherever YouTube is unreachable (HF Spaces
# black-holes TLS to www.youtube.com). It serves CC-licensed tracks and, unlike
# YouTube, hands back a direct MP3 URL - so results feed the same guarded
# fetch path as a pasted link. Free client_id: https://devportal.jamendo.com/
JAMENDO_CLIENT_ID = os.environ.get("JAMENDO_CLIENT_ID", "").strip()
JAMENDO_API       = "https://api.jamendo.com/v3.0"
JAMENDO_TIMEOUT   = 20
# Fetch a wider pool than we show, so the most-listened of the matches can rise
# to the top rather than whichever eight Jamendo happened to return first.
JAMENDO_POOL      = 30

def create_chord_templates():
    templates = {}
    for i, root in enumerate(PITCH_CLASSES):
        maj = np.zeros(12)
        maj[i]            = 1.3
        maj[(i + 4) % 12] = 0.9
        maj[(i + 7) % 12] = 1.0
        templates[f"{root}"] = maj / np.linalg.norm(maj)

        minor = np.zeros(12)
        minor[i]            = 1.3
        minor[(i + 3) % 12] = 0.9
        minor[(i + 7) % 12] = 1.0
        templates[f"{root}m"] = minor / np.linalg.norm(minor)

        power = np.zeros(12)
        power[i]            = 1.4
        power[(i + 7) % 12] = 1.0
        templates[f"{root}5"] = power / np.linalg.norm(power)

        sus2 = np.zeros(12)
        sus2[i]            = 1.2
        sus2[(i + 2) % 12] = 0.9
        sus2[(i + 7) % 12] = 1.0
        templates[f"{root}sus2"] = sus2 / np.linalg.norm(sus2)

        sus4 = np.zeros(12)
        sus4[i]            = 1.2
        sus4[(i + 5) % 12] = 0.9
        sus4[(i + 7) % 12] = 1.0
        templates[f"{root}sus4"] = sus4 / np.linalg.norm(sus4)
    return templates

print("Initializing Chroma Chord Templates...")
CHORD_TEMPLATES = create_chord_templates()

def process_audio_chunk(audio_data):
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]

    rms = np.sqrt(np.mean(audio_data**2))
    if rms < VOLUME_THRESHOLD:
        return {"chord": "--", "confidence": 0.0, "volume": float(rms)}

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

_COOKIE_FILE = None
_COOKIE_RESOLVED = False

def _cookie_file():
    """Resolve a Netscape cookie jar for yt-dlp, or None.

    HF Spaces exposes secrets as env vars rather than files, so YT_COOKIES may
    carry the jar's *contents*; materialise it to a 0600 temp file once.
    YT_COOKIES_FILE still takes a real path for local runs. The jar is a
    credential: it is never logged and never written under static/.
    """
    global _COOKIE_FILE, _COOKIE_RESOLVED
    if _COOKIE_RESOLVED:
        return _COOKIE_FILE
    _COOKIE_RESOLVED = True

    path = os.environ.get("YT_COOKIES_FILE")
    if path and os.path.exists(path):
        _COOKIE_FILE = path
        print("[Cookies] using jar at YT_COOKIES_FILE")
        return _COOKIE_FILE

    blob = os.environ.get("YT_COOKIES", "")
    if not blob.strip():
        print("[Cookies] none configured - YouTube will likely block a cloud IP")
        return None

    # Secret editors often turn real newlines into the two characters \ and n.
    if "\\n" in blob and "\n" not in blob.strip():
        blob = blob.replace("\\n", "\n")
    blob = blob.replace("\r\n", "\n").strip()
    if not blob.startswith("# Netscape"):
        blob = "# Netscape HTTP Cookie File\n" + blob

    try:
        import tempfile
        fd, tmp = tempfile.mkstemp(prefix="ytcookies_", suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(blob + "\n")
        os.chmod(tmp, 0o600)
        _COOKIE_FILE = tmp
        print(f"[Cookies] loaded jar from YT_COOKIES ({blob.count(chr(10)) + 1} lines)")
    except Exception as e:
        print(f"[Cookies] could not write jar: {type(e).__name__}")
    return _COOKIE_FILE


_FFMPEG_EXE = None

def _ffmpeg_exe():
    """Resolve ffmpeg once — imageio_ffmpeg.get_ffmpeg_exe() spawns the binary
    to verify it, which is not something to repeat on every request."""
    global _FFMPEG_EXE
    if _FFMPEG_EXE is None:
        _FFMPEG_EXE = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    return _FFMPEG_EXE


def _jamendo_get(path, params):
    """Call the Jamendo REST API and return its results list.

    Jamendo answers HTTP 200 even for auth failures, putting the real status in
    headers.status, so that has to be checked explicitly rather than trusting
    the status code.
    """
    from urllib.parse import urlencode
    if not JAMENDO_CLIENT_ID:
        raise RuntimeError("JAMENDO_CLIENT_ID is not set")

    query = dict(params, client_id=JAMENDO_CLIENT_ID)
    query.setdefault('format', 'json')
    url = f"{JAMENDO_API}/{path}/?{urlencode(query)}"
    req = urllib.request.Request(url, headers={'User-Agent': 'live-chord-ai/1.0'})
    with urllib.request.urlopen(req, timeout=JAMENDO_TIMEOUT) as resp:
        payload = json.loads(resp.read().decode('utf-8', 'replace'))

    head = payload.get('headers') or {}
    if head.get('status') != 'success':
        raise RuntimeError(head.get('error_message') or 'Jamendo API error')
    return payload.get('results') or []


def _jamendo_tracks(limit, retry=False, **params):
    """One /tracks/ call, optionally retrying an empty-but-successful reply.

    Jamendo throttles a shared cloud egress IP by returning status=success with
    zero results rather than an error, so the identical query alternates between
    hits and nothing.
    """
    attempts = 3 if retry else 1
    for i in range(attempts):
        rows = _jamendo_get('tracks', dict(params, limit=limit, audioformat='mp31'))
        if rows or i + 1 >= attempts:
            return rows
        print(f"[Jamendo] empty reply, retrying ({i+1}/{attempts-1})")
        time.sleep(0.4 * (i + 1))
    return []


def _listens(track):
    try:
        return int((track.get('stats') or {}).get('rate_listened_total') or 0)
    except (TypeError, ValueError):
        return 0


def _jamendo_search(query, limit=8):
    """Best-known tracks matching the query.

    Jamendo's own order=popularity_total does not mean "popular matches" - it
    disregards the search terms, so "Dazie Mae Move On" comes back as an
    unrelated chart track. Relevance is therefore left to the default ordering
    and popularity applied afterwards, over a wider pool than we display, using
    the listen counts that include=stats returns.
    """
    q = (query or '').strip()
    if not q:
        return []

    rows = _jamendo_tracks(JAMENDO_POOL, retry=True, search=q, include='stats')
    if not rows:
        rows = _jamendo_tracks(JAMENDO_POOL, name=q, include='stats')

    rows = [r for r in rows if r.get('audio') or r.get('audiodownload')]
    rows.sort(key=_listens, reverse=True)

    out = []
    for r in rows[:limit]:
        audio = r.get('audio') or r.get('audiodownload')
        secs = int(r.get('duration') or 0)
        # Jamendo returns HTML-escaped text, so an artist like "Dada & the
        # Weathermen" arrives as "Dada &amp; the Weathermen" and would render
        # literally - the UI sets these via textContent, not innerHTML.
        out.append({
            'trackId':   str(r.get('id') or ''),
            'title':     html.unescape(r.get('name') or ''),
            'artist':    html.unescape(r.get('artist_name') or ''),
            'duration':  f"{secs // 60}:{secs % 60:02d}" if secs else '',
            'thumbnail': r.get('album_image') or r.get('image') or '',
            'audio_url': audio,
        })
    return out


def _ytmusic_url(query):
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
                        print(f"[YTMusic] Found ({filter_type}): {artist} - {title}  [{vid}]")
                        return f"https://www.youtube.com/watch?v={vid}"
                break
            except Exception as e:
                print(f"[YTMusic] {filter_type} attempt {attempt+1}/3: {e}")
                _time.sleep(0.5 * (attempt + 1))
    return None


def download_audio(query, video_id=None):
    import time as _time, uuid as _uuid
    print(f"\nSearching YouTube Music for: '{query}'...")
    os.makedirs("static", exist_ok=True)

    # UUID filenames avoid Windows file-lock collisions when browser still holds previous file
    for f in glob.glob("static/downloaded_*"):
        try:
            os.remove(f)
            print(f"[Cleanup] Deleted {f}")
        except OSError:
            print(f"[Cleanup] Skipped (locked): {f}")

    out_base = f"static/downloaded_{_uuid.uuid4().hex[:12]}"

    if video_id:
        url = f"https://www.youtube.com/watch?v={video_id}"
    else:
        url = _ytmusic_url(query) or f"ytsearch1:{query}"
    print(f"[Download] URL: {url}")

    ffmpeg_path = _ffmpeg_exe()

    cookies_file = _cookie_file()

    success = False
    title = "Unknown"

    # mediaconnect first: as of yt-dlp 2026.03 it is the only client that still
    # returns an audio format for most tracks, so leading with anything else
    # burns a round trip per request. The rest stay as fallbacks since client
    # viability shifts with yt-dlp releases and with the server's IP.
    client_order = [
        ['mediaconnect'], ['tv_simply'], ['mweb'], ['ios'], ['tv'], ['web'],
    ]
    # Keep the compressed stream as downloaded: transcoding to WAV cost a
    # re-encode per request and shipped ~50 MB to the browser instead of ~4 MB.
    # m4a/AAC is preferred over webm/opus purely for Safari playback.
    base_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio/best',
        'outtmpl': f'{out_base}.%(ext)s',
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'ffmpeg_location': ffmpeg_path,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        },
        'socket_timeout': 15,
        'retries': 1,
        **IPV4_OPTS,
    }
    if cookies_file:
        base_opts['cookiefile'] = cookies_file

    # Walking all six clients at 30s/socket could exceed three minutes, long
    # enough that the platform gateway 500s before we can return a real message.
    # Stop early and surface our own error while the request is still alive.
    started = _time.monotonic()
    for client in client_order:
        if _time.monotonic() - started > DOWNLOAD_DEADLINE_SEC:
            print(f"[Download] deadline hit after {DOWNLOAD_DEADLINE_SEC}s, giving up")
            break
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
                or 'format is not available' in err_msg or 'requested format' in err_msg
            )
            if transient:
                print(f"[Download] {client[0]} transient error, trying next client…")
                continue
            print(f"[Download] {client[0]} failed (non-transient): {e}")
            break

    if not success:
        print("[Download] yt-dlp exhausted, trying pytubefix...")
        try:
            from pytubefix import YouTube
            target_url = url if url.startswith('http') else None
            if not target_url:
                with yt_dlp.YoutubeDL({'quiet': True, 'extract_flat': True}) as ydl:
                    s = ydl.extract_info(url, download=False)
                    if s and s.get('entries'):
                        target_url = f"https://www.youtube.com/watch?v={s['entries'][0]['id']}"
            if target_url:
                yt = YouTube(target_url)
                stream = yt.streams.get_audio_only()
                if stream is None:
                    raise RuntimeError("No audio stream available")
                # Served as-is; _decode_audio reads it without a WAV round trip.
                stream.download(output_path="static",
                                filename=f"{os.path.basename(out_base)}.m4a")
                title = yt.title
                success = True
                print(f"[Download] OK via pytubefix: {title}")
        except Exception as e:
            print(f"[Download] pytubefix failed: {e}")

    if not success:
        raise RuntimeError(
            "Couldn't fetch this song right now. "
            "YouTube is rate-limiting our server — please try a different song or try again in a moment."
        )

    downloaded = glob.glob(f"{out_base}.*")
    if not downloaded:
        print("Error: audio file was not saved.")
        return None

    return os.path.basename(downloaded[0])

MAX_URL_AUDIO_BYTES = 60 * 1024 * 1024
URL_FETCH_TIMEOUT   = 30

# Content-Type -> extension. The browser needs a truthful extension to pick a
# decoder; ffmpeg sniffs the container itself and ignores the name.
_CTYPE_EXT = {
    'audio/mpeg': '.mp3',  'audio/mp3':  '.mp3',  'audio/mp4':   '.m4a',
    'audio/x-m4a': '.m4a', 'audio/aac':  '.aac',  'audio/ogg':   '.ogg',
    'audio/opus': '.opus', 'audio/webm': '.webm', 'audio/wav':   '.wav',
    'audio/x-wav': '.wav', 'audio/flac': '.flac', 'audio/x-flac': '.flac',
    'video/mp4':  '.m4a',  'video/webm': '.webm',
}
_ALLOWED_EXT = set(_CTYPE_EXT.values())


def _assert_public_url(url):
    """Reject anything that could reach this host's own network.

    Fetching a user-supplied URL server-side is an SSRF primitive: unguarded, a
    request for http://169.254.169.254/... or a private address would be fetched
    with our credentials and handed back through the audio player. Every DNS
    answer must be a global address, and this runs again on each redirect hop.
    """
    import ipaddress
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        raise ValueError("Only http and https links are supported.")
    host = parsed.hostname
    if not host:
        raise ValueError("That doesn't look like a valid link.")

    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise ValueError("Couldn't resolve that address.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_multicast:
            raise ValueError("That link points to a private address.")
    return parsed


class _GuardedRedirect(urllib.request.HTTPRedirectHandler):
    """Re-validate on every hop; otherwise a public URL could 302 to localhost."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _assert_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_from_url(url):
    import uuid as _uuid
    from urllib.parse import urlparse, unquote

    url = (url or '').strip()
    if not url:
        raise ValueError("Paste a link to an audio file first.")
    _assert_public_url(url)

    os.makedirs("static", exist_ok=True)
    for f in glob.glob("static/downloaded_*"):
        try:
            os.remove(f)
        except OSError:
            print(f"[Cleanup] Skipped (locked): {f}")

    opener = urllib.request.build_opener(_GuardedRedirect())
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept': 'audio/*,video/*;q=0.9,*/*;q=0.8',
    })
    print(f"[URL] fetching {urlparse(url).netloc}...")
    try:
        resp = opener.open(req, timeout=URL_FETCH_TIMEOUT)
    except ValueError:
        raise
    except urllib.error.HTTPError as e:
        # Name the status: "403" tells the user the host refused us, which is a
        # different fix from a typo'd link.
        raise ValueError(f"That link returned HTTP {e.code} ({e.reason}).")
    except urllib.error.URLError as e:
        raise ValueError(f"Couldn't reach that link ({e.reason}).")
    except Exception as e:
        raise ValueError(f"Couldn't download that link ({type(e).__name__}).")

    with resp:
        ctype = (resp.headers.get('Content-Type') or '').split(';')[0].strip().lower()
        declared = resp.headers.get('Content-Length')
        if declared and int(declared) > MAX_URL_AUDIO_BYTES:
            raise ValueError(f"That file is larger than {MAX_URL_AUDIO_BYTES // (1024*1024)} MB.")

        ext = _CTYPE_EXT.get(ctype, '')
        if not ext:
            suffix = os.path.splitext(unquote(urlparse(url).path))[1].lower()
            ext = suffix if suffix in _ALLOWED_EXT else '.bin'

        out_path = f"static/downloaded_{_uuid.uuid4().hex[:12]}{ext}"
        total = 0
        with open(out_path, 'wb') as fh:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_URL_AUDIO_BYTES:
                    fh.close()
                    os.remove(out_path)
                    raise ValueError(
                        f"That file is larger than {MAX_URL_AUDIO_BYTES // (1024*1024)} MB.")
                fh.write(chunk)

    if total == 0:
        os.remove(out_path)
        raise ValueError("That link returned an empty file.")

    print(f"[URL] got {total/1e6:.1f} MB, content-type={ctype or 'unknown'}")
    return os.path.basename(out_path)


app = FastAPI()
active_connections: List[WebSocket] = []


def _warm_up_dsp():
    """Compile librosa's numba kernels on a synthetic clip at boot.

    Cold, the first extraction pays ~4s of JIT locally and noticeably more on a
    small cloud CPU. Runs on a daemon thread so the port still binds instantly.
    """
    try:
        t0 = time.time()
        y = np.zeros(ANALYSIS_SR * 3, dtype=np.float32)
        y[::64] = 0.1
        C = np.abs(librosa.cqt(
            y, sr=ANALYSIS_SR, hop_length=ANALYSIS_HOP, fmin=CQT_FMIN,
            n_bins=CQT_OCTAVES * CQT_BINS_PER_OCT,
            bins_per_octave=CQT_BINS_PER_OCT))
        onset_env = librosa.onset.onset_strength(
            S=librosa.amplitude_to_db(C, ref=np.max),
            sr=ANALYSIS_SR, hop_length=ANALYSIS_HOP)
        librosa.beat.beat_track(onset_envelope=onset_env, sr=ANALYSIS_SR,
                                hop_length=ANALYSIS_HOP, trim=False)
        C_h, _ = librosa.decompose.hpss(C, margin=3)
        librosa.feature.chroma_cqt(
            C=C_h, sr=ANALYSIS_SR, hop_length=ANALYSIS_HOP, fmin=CQT_FMIN,
            n_octaves=CQT_OCTAVES, bins_per_octave=CQT_BINS_PER_OCT)
        librosa.estimate_tuning(y=y, sr=ANALYSIS_SR)
        print(f"[Warmup] DSP kernels ready in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"[Warmup] skipped: {type(e).__name__}: {e}")


# `python app.py` runs this file as __main__ and then uvicorn imports it again as
# "app", so guard the thread or the JIT work is done twice, in parallel, for nothing.
if __name__ != "__main__":
    threading.Thread(target=_warm_up_dsp, daemon=True).start()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    audio_buffer = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
    is_listening = False
    import time as _time
    last_predict_time = 0.0

    try:
        while True:
            msg = await websocket.receive()

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

            if "bytes" in msg and is_listening:
                pcm = np.frombuffer(msg["bytes"], dtype=np.float32)
                if len(pcm) == 0:
                    continue

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

# Higher P_STAY = smoother sequence (fewer chord changes). Window step ~0.58s,
# so 0.9 favors chords that persist for a few windows.
P_STAY = 0.90

# BETA sharpens the per-window template scores into emission probabilities before
# Viterbi. Higher = more confident emissions (less smoothing pull from neighbors).
EMIT_BETA = 12.0

def _build_class_templates():
    """Major/minor chord templates aligned to CHORD_CLASSES order (24 x 12)."""
    mat = np.zeros((len(CHORD_CLASSES), 12), dtype=np.float64)
    for k, name in enumerate(CHORD_CLASSES):
        minor = name.endswith('m')
        root  = name[:-1] if minor else name
        i = PITCH_CLASSES.index(root)
        v = np.zeros(12)
        v[i]                                = 1.3
        v[(i + (3 if minor else 4)) % 12]   = 0.9
        v[(i + 7) % 12]                     = 1.0
        mat[k] = v / np.linalg.norm(v)
    return mat

CHORD_CLASS_TEMPLATES = _build_class_templates()

def template_emissions(windows):
    """Per-window emission probabilities from chroma-template similarity.

    windows: (N, WIN_FRAMES, 12) chroma. Returns (N, 24) softmaxed over chords.
    """
    vecs = windows.mean(axis=1)                      # (N, 12)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.where(norms == 0, 1.0, norms)
    scores = vecs @ CHORD_CLASS_TEMPLATES.T          # (N, 24)
    scores = EMIT_BETA * scores
    scores -= scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)

def viterbi_smooth(probs, p_stay=P_STAY):
    """First-order Markov (Viterbi) decode over per-window chord probabilities.

    Replaces per-window argmax with the globally most-likely chord path, using a
    diagonal-loaded transition matrix (self-transition p_stay, uniform otherwise).
    Returns an array of class indices, one per window.
    """
    probs = np.asarray(probs, dtype=np.float64)
    n_steps, n_states = probs.shape
    if n_steps == 0:
        return np.zeros(0, dtype=int)

    eps = 1e-12
    log_emit = np.log(probs + eps)

    p_other = (1.0 - p_stay) / (n_states - 1)
    log_trans = np.full((n_states, n_states), np.log(p_other))
    np.fill_diagonal(log_trans, np.log(p_stay))

    delta = log_emit[0].copy()
    backptr = np.zeros((n_steps, n_states), dtype=int)
    for t in range(1, n_steps):
        scores = delta[:, None] + log_trans       # (prev, curr)
        backptr[t] = np.argmax(scores, axis=0)
        delta = scores[backptr[t], np.arange(n_states)] + log_emit[t]

    path = np.zeros(n_steps, dtype=int)
    path[-1] = int(np.argmax(delta))
    for t in range(n_steps - 1, 0, -1):
        path[t - 1] = backptr[t, path[t]]
    return path

def _decode_audio(filepath, sr=ANALYSIS_SR, max_seconds=MAX_ANALYSIS_SEC):
    """Decode any container ffmpeg understands straight to mono float32.

    Replaces librosa.load, which needs a WAV on disk: soundfile has no webm/m4a
    backend and librosa's audioread fallback is gone as of 0.10.
    """
    ffmpeg_path = _ffmpeg_exe()
    proc = subprocess.run(
        [ffmpeg_path, "-v", "quiet", "-t", str(max_seconds), "-i", filepath,
         "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        check=True, capture_output=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def extract_chords_from_file(filepath):
    y = _decode_audio(filepath)
    if y.size == 0:
        return []
    song_duration = float(len(y)) / ANALYSIS_SR

    tuning = float(librosa.estimate_tuning(y=y, sr=ANALYSIS_SR))

    # One CQT feeds beat tracking, harmonic separation and chroma. Computing it
    # once here is what makes this ~15x faster than separating the waveform with
    # effects.harmonic and then running a second transform over the result.
    C = np.abs(librosa.cqt(
        y, sr=ANALYSIS_SR, hop_length=ANALYSIS_HOP, fmin=CQT_FMIN,
        n_bins=CQT_OCTAVES * CQT_BINS_PER_OCT,
        bins_per_octave=CQT_BINS_PER_OCT, tuning=tuning))

    # Beats come off the full CQT, not the percussive half: tonal onsets track
    # the quarter-note pulse here, where the drum-only envelope halves the tempo.
    onset_env = librosa.onset.onset_strength(
        S=librosa.amplitude_to_db(C, ref=np.max),
        sr=ANALYSIS_SR, hop_length=ANALYSIS_HOP)
    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=ANALYSIS_SR, hop_length=ANALYSIS_HOP, trim=False)

    # Median-filtering the CQT gives the same harmonic emphasis as
    # effects.harmonic without the STFT -> mask -> ISTFT round trip.
    C_harmonic, _ = librosa.decompose.hpss(C, margin=3)
    chroma = librosa.feature.chroma_cqt(
        C=C_harmonic, sr=ANALYSIS_SR, hop_length=ANALYSIS_HOP, fmin=CQT_FMIN,
        n_octaves=CQT_OCTAVES, bins_per_octave=CQT_BINS_PER_OCT)
    chroma_T = chroma.T
    T = chroma_T.shape[0]

    beat_frames = np.asarray(beat_frames, dtype=int)
    beat_times  = librosa.frames_to_time(
        beat_frames, sr=ANALYSIS_SR, hop_length=ANALYSIS_HOP)
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

    secs_per_frame = ANALYSIS_HOP / ANALYSIS_SR
    WIN_FRAMES = max(1, int(round(WIN_SECONDS / secs_per_frame)))
    STRIDE     = max(1, WIN_FRAMES // 2)

    # Chroma-template emissions + Viterbi smoothing. The trained transformer was
    # minor-biased (confidently mislabelled major chords as minor), so we decode
    # straight from the harmonically-grounded chroma templates instead.
    starts = list(range(0, T - WIN_FRAMES + 1, STRIDE))
    if not starts:
        return []
    windows = np.array(
        [chroma_T[s : s + WIN_FRAMES] for s in starts],
        dtype=np.float32
    )
    emis         = template_emissions(windows)
    pred_classes = viterbi_smooth(emis, P_STAY)
    pred_confs   = emis[np.arange(len(pred_classes)), pred_classes]
    raw = [
        (CHORD_CLASSES[pred_classes[i]],
         starts[i] * secs_per_frame,
         float(pred_confs[i]))
        for i in range(len(starts))
    ]

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

    if timeline:
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


@app.post("/api/from-url")
async def search_from_url(request: UrlQuery):
    """Analyse audio at a direct link, bypassing YouTube entirely."""
    loop = asyncio.get_running_loop()
    try:
        filename = await loop.run_in_executor(None, download_from_url, request.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=400, detail="Couldn't fetch audio from that link.")

    try:
        return await _build_song_response(filename)
    except HTTPException:
        # Reaching here means the bytes downloaded but ffmpeg could not decode
        # them - usually an HTML error page served with an audio content type.
        raise HTTPException(
            status_code=400,
            detail="That link didn't contain audio we can read. Use a direct link to an audio file.")


def _ytmusic_search_with_retry(query, limit=8, attempts=3):
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
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'skip_download': True,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        },
        'extractor_args': {'youtube': {'player_client': ['mediaconnect', 'tv', 'web']}},
        **IPV4_OPTS,
    }
    if _cookie_file():
        ydl_opts['cookiefile'] = _cookie_file()
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
        # Jamendo first when configured: it works from hosts that cannot reach
        # YouTube, and its results carry a directly playable audio_url.
        if JAMENDO_CLIENT_ID:
            try:
                hits = _jamendo_search(q, limit=8)
                if hits:
                    return hits
                print("[Jamendo] no matches")
            except Exception as e:
                print(f"[Jamendo] search failed: {e}")
            return []

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
        import time as _time
        # Jamendo has no dedicated suggest endpoint, so build phrases from the
        # track search itself. Deduped because one artist often fills the page.
        if JAMENDO_CLIENT_ID:
            try:
                seen, out = set(), []
                for r in _jamendo_search(q, limit=6):
                    phrase = f"{r['artist']} {r['title']}".strip() if r['artist'] else r['title']
                    if phrase and phrase.lower() not in seen:
                        seen.add(phrase.lower())
                        out.append(phrase)
                return out
            except Exception as e:
                print(f"[Autocomplete] jamendo failed: {e}")
                return []

        for attempt in range(2):
            try:
                suggestions = ytmusic.get_search_suggestions(q)
                if suggestions:
                    return suggestions
                break
            except Exception as e:
                if attempt == 0:
                    _time.sleep(0.4)
                    continue
                print(f"[Autocomplete] suggestion fetch failed: {type(e).__name__}")

        # Fallback: yt-dlp flat search bypasses ytmusic entirely
        # (ytmusic hits SSL/EOF errors on HF cloud).
        try:
            results = _ytdlp_search_fallback(q, limit=5)
            seen = set()
            out = []
            for r in results:
                title  = (r.get('title') or '').strip()
                artist = (r.get('artist') or '').strip()
                if not title:
                    continue
                phrase = f"{artist} {title}".strip() if artist else title
                key = phrase.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(phrase)
            return out
        except Exception as e:
            print(f"[Autocomplete] search fallback failed: {type(e).__name__}")
            return []

    return await loop.run_in_executor(None, _suggest)

@app.get("/api/diag")
async def diag():
    """Report whether this host can reach YouTube at all.

    Exists because the Space failed with a bare SSL EOF and nothing in the
    request path could tell us why. Booleans and versions only - never cookie
    contents or filesystem paths. Safe to delete once downloads are healthy.
    """
    import socket

    def _tls(host, family):
        import ssl as _ssl
        try:
            infos = socket.getaddrinfo(host, 443, family, socket.SOCK_STREAM)
        except Exception as e:
            return f"dns: {type(e).__name__}"
        if not infos:
            return "no address"
        af, socktype, proto, _, addr = infos[0]
        sock = None
        try:
            sock = socket.socket(af, socktype, proto)
            sock.settimeout(8)
            sock.connect(addr)
            with _ssl.create_default_context().wrap_socket(
                    sock, server_hostname=host) as tls:
                return f"ok ({tls.version()})"
        except Exception as e:
            return f"{type(e).__name__}: {str(e)[:70]}"
        finally:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

    def _tcp(host, port=443):
        """TCP-only reach test, to tell a blocked port from a blocked handshake."""
        try:
            s = socket.create_connection((host, port), timeout=8)
            s.close()
            return "ok"
        except Exception as e:
            return f"{type(e).__name__}: {str(e)[:50]}"

    # huggingface.co and pypi.org are controls: if those fail too, egress is
    # broken generally rather than YouTube being singled out. The rest are
    # candidate audio sources - yt-dlp already has extractors for all of them.
    HOSTS = ["www.youtube.com", "music.youtube.com",
             "huggingface.co", "pypi.org",
             "bandcamp.com", "soundcloud.com", "api-v2.soundcloud.com",
             "archive.org", "api.jamendo.com", "freemusicarchive.org",
             "ccmixter.org", "opengameart.org"]

    def _jamendo_probe():
        """Run a real search and report Jamendo's own headers block.

        The Space gets status=success with zero results for a query that
        returns plenty locally, so the interesting data is in results_count,
        warnings and error_message - not in whether the socket opened.
        """
        if not JAMENDO_CLIENT_ID:
            return "not configured"
        from urllib.parse import urlencode
        url = f"{JAMENDO_API}/tracks/?" + urlencode({
            'client_id': JAMENDO_CLIENT_ID, 'format': 'json',
            'limit': 3, 'search': 'acoustic',
        })
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'live-chord-ai/1.0'})
            with urllib.request.urlopen(req, timeout=JAMENDO_TIMEOUT) as r:
                body = json.loads(r.read().decode('utf-8', 'replace'))
            head = body.get('headers') or {}
            return {
                "status":        head.get('status'),
                "code":          head.get('code'),
                "results_count": head.get('results_count'),
                "error_message": head.get('error_message') or None,
                "warnings":      head.get('warnings') or None,
                "first_track":   (body.get('results') or [{}])[0].get('name'),
            }
        except Exception as e:
            return f"{type(e).__name__}: {str(e)[:90]}"

    def _probe():
        out = {
            "jamendo_configured": bool(JAMENDO_CLIENT_ID),
            "jamendo_probe": _jamendo_probe(),
            "search_backend": "jamendo" if JAMENDO_CLIENT_ID else "youtube",
            "cookies_configured": bool(_cookie_file()),
            "force_ipv4": FORCE_IPV4,
            "yt_dlp": getattr(yt_dlp.version, "__version__", "unknown"),
            "ffmpeg": bool(_ffmpeg_exe()),
            "ipv6": _tls("www.youtube.com", socket.AF_INET6),
        }
        for h in HOSTS:
            out[h] = {"tcp": _tcp(h), "tls": _tls(h, socket.AF_INET)}
        return out

    return await asyncio.get_running_loop().run_in_executor(None, _probe)


@app.get("/")
def read_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    os.makedirs("static", exist_ok=True)
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting FastAPI Server on port {port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port)
