"""
Worker média pour Ask Tariq / CONFOWEB
voix (audio)  ->  transcription mot par mot  ->  sous-titres animés  ->  short vertical 1080x1920

Routes :
  GET  /health             état du service (sans clé)
  GET  /assets             liste les fonds, musiques et polices disponibles
  POST /assets/{kind}      envoie un fichier dans backgrounds | music | fonts
  POST /transcribe         audio -> JSON (texte + mots horodatés)
  POST /render             audio -> vidéo MP4 sous-titrée
"""
import json
import os
import random
import re
import secrets
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

# --------------------------------------------------------------------------
# Configuration (variables d'environnement)
# --------------------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
ASSETS = DATA_DIR / "assets"
JOBS = DATA_DIR / "jobs"
MODELS = DATA_DIR / "models"
for _p in (ASSETS / "backgrounds", ASSETS / "music", ASSETS / "fonts", JOBS, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("API_KEY", "")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "fr")

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MUSIC_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".aac"}

app = FastAPI(title="Media worker", version="1.0")

# Un seul travail lourd à la fois : protège la RAM du VPS partagé.
WORK = threading.Semaphore(1)
_model = None
_model_lock = threading.Lock()


def check_key(x_api_key: str = Header(default="")):
    """Si API_KEY est définie, tout appel doit fournir l'en-tête X-API-Key."""
    if API_KEY and not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Clé API invalide")


# --------------------------------------------------------------------------
# Utilitaires
# --------------------------------------------------------------------------
def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", Path(name or "file").name) or "file"


def save_upload(upload: UploadFile, folder: Path, stem: str) -> Path:
    suffix = Path(upload.filename or "").suffix.lower() or ".bin"
    dest = folder / f"{stem}{suffix}"
    with open(dest, "wb") as out:
        shutil.copyfileobj(upload.file, out, length=1024 * 1024)
    return dest


def probe_duration(path: Path) -> float:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    try:
        return float(res.stdout.strip())
    except ValueError:
        raise HTTPException(status_code=400, detail="Fichier audio illisible")


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel
            _model = WhisperModel(
                WHISPER_MODEL, device="cpu", compute_type="int8",
                download_root=str(MODELS),
            )
        return _model


def transcribe_file(path: Path, language: str, prompt: str) -> dict:
    lang = None if language in ("", "auto") else language
    segments, info = get_model().transcribe(
        str(path), language=lang, word_timestamps=True,
        vad_filter=True, initial_prompt=prompt or None, beam_size=5,
    )
    words, texts = [], []
    for seg in segments:
        texts.append(seg.text.strip())
        for w in seg.words or []:
            token = w.word.strip()
            if token:
                words.append({"w": token, "start": round(w.start, 2), "end": round(w.end, 2)})
    return {
        "language": info.language,
        "duration": round(info.duration, 2),
        "text": " ".join(texts),
        "words": words,
    }


def pick_asset(kind: str, name: str, exts: set, default: str):
    """name : '' (=default), 'random', 'none' ou un nom de fichier."""
    folder = ASSETS / kind
    choice = name or default
    if choice == "none":
        return None
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    if choice == "random":
        return random.choice(files) if files else None
    path = folder / Path(choice).name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{kind}/{choice} introuvable")
    return path


# --------------------------------------------------------------------------
# Sous-titres ASS : 3 mots à l'écran, le mot prononcé en surbrillance
# --------------------------------------------------------------------------
ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,DejaVu Sans,78,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,7,3,2,70,70,470,1
Style: Title,DejaVu Sans,88,&H0000E6FF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,7,3,8,70,70,230,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
HIGHLIGHT = r"{\c&H00E6FF&}"   # jaune (BBGGRR)
NORMAL = r"{\c&HFFFFFF&}"      # blanc


def ass_time(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def clean_token(token: str, upper: bool) -> str:
    token = token.replace("{", "(").replace("}", ")").replace("\\", "")
    return token.upper() if upper else token


def chunk_words(words, max_words=3, max_chars=18, max_gap=0.6):
    chunks, cur = [], []
    for w in words:
        if cur:
            gap = w["start"] - cur[-1]["end"]
            chars = sum(len(x["w"]) + 1 for x in cur) + len(w["w"])
            if len(cur) >= max_words or chars > max_chars or gap > max_gap:
                chunks.append(cur)
                cur = []
        cur.append(w)
        if re.search(r"[.!?…:;]$", w["w"]):
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def build_ass(words, title: str, upper: bool, duration: float) -> str:
    lines = [ASS_HEADER]
    if title.strip():
        end = min(3.5, duration)
        text = clean_token(title.strip(), upper)
        lines.append(
            f"Dialogue: 1,{ass_time(0)},{ass_time(end)},Title,,0,0,0,,{{\\fad(150,200)}}{text}\n"
        )
    for chunk in chunk_words(words):
        tokens = [clean_token(w["w"], upper) for w in chunk]
        for i, w in enumerate(chunk):
            start = w["start"]
            end = chunk[i + 1]["start"] if i + 1 < len(chunk) else w["end"] + 0.15
            parts = [
                f"{HIGHLIGHT}{tok}{NORMAL}" if j == i else tok
                for j, tok in enumerate(tokens)
            ]
            lines.append(
                f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Sub,,0,0,0,,{' '.join(parts)}\n"
            )
    return "".join(lines)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "whisper_model": WHISPER_MODEL,
        "auth": bool(API_KEY),
    }


@app.get("/assets", dependencies=[Depends(check_key)])
def list_assets():
    return {
        kind: sorted(p.name for p in (ASSETS / kind).iterdir() if p.is_file())
        for kind in ("backgrounds", "music", "fonts")
    }


@app.post("/assets/{kind}", dependencies=[Depends(check_key)])
def upload_asset(kind: str, file: UploadFile = File(...)):
    if kind not in ("backgrounds", "music", "fonts"):
        raise HTTPException(status_code=400, detail="kind doit être backgrounds, music ou fonts")
    dest = ASSETS / kind / safe_name(file.filename)
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out, length=1024 * 1024)
    return {"saved": f"{kind}/{dest.name}"}


@app.post("/transcribe", dependencies=[Depends(check_key)])
def transcribe(
    audio: UploadFile = File(...),
    language: str = Form(DEFAULT_LANGUAGE),
    initial_prompt: str = Form(""),
):
    job = JOBS / uuid.uuid4().hex
    job.mkdir(parents=True)
    try:
        path = save_upload(audio, job, "voice")
        with WORK:
            return transcribe_file(path, language, initial_prompt)
    finally:
        shutil.rmtree(job, ignore_errors=True)


@app.post("/render", dependencies=[Depends(check_key)])
def render(
    audio: UploadFile = File(...),
    background: str = Form(""),          # '', 'random', 'none' ou nom de fichier
    music: str = Form(""),               # '', 'random', 'none' ou nom de fichier
    music_volume: float = Form(0.15),    # poids de la musique face à la voix (1.0)
    title: str = Form(""),               # texte d'accroche affiché 3,5 s en haut
    language: str = Form(DEFAULT_LANGUAGE),
    initial_prompt: str = Form(""),      # noms et termes à reconnaître (Starlink, SpaceX...)
    uppercase: bool = Form(True),
    words_json: str = Form(""),          # optionnel : mots déjà transcrits (saute Whisper)
):
    job = JOBS / uuid.uuid4().hex
    job.mkdir(parents=True)
    try:
        voice = save_upload(audio, job, "voice")
        duration = probe_duration(voice)

        with WORK:
            if words_json.strip():
                words = json.loads(words_json)
                words = words["words"] if isinstance(words, dict) else words
            else:
                words = transcribe_file(voice, language, initial_prompt)["words"]
            if not words:
                raise HTTPException(status_code=422, detail="Aucune parole détectée")

            (job / "subs.ass").write_text(build_ass(words, title, uppercase, duration), encoding="utf-8")

            bg = pick_asset("backgrounds", background, VIDEO_EXT | IMAGE_EXT, "random")
            mus = pick_asset("music", music, MUSIC_EXT, "none")
            total = duration + 0.4

            cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
            if bg is None:
                cmd += ["-f", "lavfi", "-i", "color=c=0x0b1020:s=1080x1920:r=30"]
            elif bg.suffix.lower() in IMAGE_EXT:
                cmd += ["-loop", "1", "-framerate", "30", "-i", str(bg)]
            else:
                cmd += ["-stream_loop", "-1", "-i", str(bg)]
            cmd += ["-i", str(voice)]
            if mus:
                cmd += ["-stream_loop", "-1", "-i", str(mus)]

            fonts = ASSETS / "fonts"
            fontsdir = f":fontsdir={fonts}" if any(fonts.iterdir()) else ""
            vf = (
                "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
                f"crop=1080:1920,fps=30,format=yuv420p,ass=subs.ass{fontsdir}[v]"
            )
            voice_chain = (
                "highpass=f=80,"
                "acompressor=threshold=0.125:ratio=3:attack=5:release=80"
            )
            loud = "loudnorm=I=-14:TP=-1.5:LRA=11"
            if mus:
                af = (
                    f"[1:a]{voice_chain}[voice];"
                    f"[voice][2:a]amix=inputs=2:duration=first:weights='1 {music_volume}':"
                    f"dropout_transition=0,{loud}[a]"
                )
            else:
                af = f"[1:a]{voice_chain},{loud}[a]"

            cmd += [
                "-filter_complex", f"{vf};{af}",
                "-map", "[v]", "-map", "[a]",
                "-t", f"{total:.2f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-r", "30",
                "-c:a", "aac", "-b:a", "160k", "-ar", "44100",
                "-movflags", "+faststart", "short.mp4",
            ]
            res = subprocess.run(cmd, cwd=job, capture_output=True, text=True, timeout=900)
            if res.returncode != 0:
                raise HTTPException(status_code=500, detail=f"FFmpeg a échoué : {res.stderr[-1500:]}")

        return FileResponse(
            job / "short.mp4", media_type="video/mp4", filename="short.mp4",
            background=BackgroundTask(shutil.rmtree, job, ignore_errors=True),
        )
    except Exception:
        shutil.rmtree(job, ignore_errors=True)
        raise
