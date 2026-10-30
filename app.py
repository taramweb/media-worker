"""
Worker média / by TaRaM / CONFOWEB
voix (audio)  ->  transcription mot par mot  ->  sous-titres animés  ->  short vertical 1080x1920

Routes :
  GET  /health             état du service (sans clé)
  GET  /assets             liste les fonds, musiques et polices disponibles
  POST /assets/{kind}      envoie un fichier dans backgrounds | music | fonts
  POST /transcribe         audio -> JSON (texte + mots horodatés)
  POST /render             audio -> vidéo MP4 sous-titrée (fond unique)
  POST /align              audio + script -> minutage de chaque plan (diagnostic)
  POST /render_story       audio + script + images -> histoire montée (Ken Burns,
                           fondus, sous-titres arabes, musique avec ducking)
"""
import difflib
import gc
import json
import math
import os
import random
import re
import secrets
import shutil
import subprocess
import threading
import unicodedata
import uuid
from pathlib import Path
from typing import List

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
# 1 = le modèle Whisper reste en RAM (rapide) ; 0 = libéré après chaque transcription (VPS partagé)
KEEP_MODEL = os.getenv("KEEP_MODEL", "1") == "1"
# Facteur de sur-échantillonnage des images avant zoompan (anti-tremblement)
KB_SCALE = int(os.getenv("KB_SCALE", "3"))

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MUSIC_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".aac"}

app = FastAPI(title="Media worker", version="1.1")

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
    result = {
        "language": info.language,
        "duration": round(info.duration, 2),
        "text": " ".join(texts),
        "words": words,
    }
    if not KEEP_MODEL:
        global _model
        with _model_lock:
            _model = None
        gc.collect()
    return result


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
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,@FONT@,72,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,7,3,5,60,60,0,1
Style: Title,@FONT@,88,&H0000E6FF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,7,3,8,70,70,230,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
HIGHLIGHT = r"{\c&H00E6FF&}"   # jaune (BBGGRR)
NORMAL = r"{\c&HFFFFFF&}"      # blanc


def shade(tag: str, k: int) -> str:
    """Variante quasi identique d'une couleur (k=0 ou 1) pour empêcher la fusion de blocs."""
    return tag if k == 0 else tag.replace("&}", "&}").replace("&H00E6FF&", "&H00E6FE&").replace("&HFFFFFF&", "&HFEFEFE&")


def ass_time(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def clean_token(token: str, upper: bool) -> str:
    token = token.replace("{", "(").replace("}", ")").replace("\\", "")
    return token.upper() if upper else token


def chunk_words(words, max_words=3, max_chars=16, max_gap=0.6):
    chunks, cur = [], []
    for w in words:
        if cur:
            gap = w["start"] - cur[-1]["end"]
            chars = sum(len(x["w"]) + 1 for x in cur) + len(w["w"])
            if len(cur) >= max_words or chars > max_chars or gap > max_gap:
                chunks.append(cur)
                cur = []
        cur.append(w)
        if re.search(r"[.!?…:;؟؛،]$", w["w"]):
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def fit_font_size(tokens, base_size: int, upper: bool, max_width: int = 960,
                  factor: float = 0.0) -> int:
    """Estime la largeur du texte et réduit la taille pour qu'il tienne sur une seule ligne."""
    factor = factor or (0.74 if upper else 0.64)
    width = (sum(len(t) for t in tokens) + 0.5 * (len(tokens) - 1)) * factor * base_size
    if width <= max_width:
        return base_size
    return max(40, int(base_size * max_width / width))


def build_ass(words, title: str, upper: bool, duration: float,
              font_size: int = 72, sub_y: int = 1450, max_words: int = 3,
              font_name: str = "DejaVu Sans", max_chars: int = 16,
              char_factor: float = 0.0, rtl: bool = False) -> str:
    lines = [ASS_HEADER.replace("@FONT@", font_name)]
    if title.strip():
        end = min(3.5, duration)
        text = clean_token(title.strip(), upper)
        lines.append(
            f"Dialogue: 1,{ass_time(0)},{ass_time(end)},Title,,0,0,0,,{{\\fad(150,200)}}{text}\n"
        )
    chunks = chunk_words(words, max_words=max_words, max_chars=max_chars)
    for ci, chunk in enumerate(chunks):
        next_start = chunks[ci + 1][0]["start"] if ci + 1 < len(chunks) else None
        tokens = [clean_token(w["w"], upper) for w in chunk]
        fs = fit_font_size(tokens, font_size, upper, factor=char_factor)
        # \an5 + \pos : le centre du texte reste EXACTEMENT au même endroit pour tous les mots
        prefix = f"{{\\an5\\pos(540,{sub_y})\\fs{fs}}}"
        for i, w in enumerate(chunk):
            start = w["start"]
            if i + 1 < len(chunk):
                end = chunk[i + 1]["start"]
            else:
                end = w["end"] + 0.15
                if next_start is not None:      # jamais de chevauchement avec le groupe suivant
                    end = min(end, next_start)
            end = max(end, start + 0.05)
            if rtl:
                # libass lit chaque bloc de couleur identique d'un seul tenant (droite -> gauche)
                # et pose les blocs distincts de gauche à droite, dans l'ordre d'écriture.
                # Pour garantir l'ordre arabe : chaque mot, chaque signe de ponctuation et
                # chaque espace forme SON bloc, écrit dans l'ordre VISUEL (dernier mot d'abord).
                # Deux blocs voisins ne partagent jamais la même valeur de couleur
                # (nuances alternées, invisibles à l'œil), sinon libass les fusionne.
                runs = []
                for j in range(len(tokens) - 1, -1, -1):
                    base = HIGHLIGHT if j == i else NORMAL
                    m = re.match(r"^(\W*)(.*?)(\W*)$", tokens[j], re.S)
                    lead, core, trail = m.group(1), m.group(2), m.group(3)
                    if trail:                      # fin de mot -> s'affiche à GAUCHE du mot
                        runs.append((base, trail))
                    runs.append((base, core or tokens[j]))
                    if lead:                       # début de mot -> s'affiche à DROITE du mot
                        runs.append((base, lead))
                    if j > 0:
                        runs.append((NORMAL, " "))
                body = ""
                for k, (base, text) in enumerate(runs):
                    body += shade(base, k % 2) + text
            else:
                body = " ".join(
                    f"{HIGHLIGHT}{tok}{NORMAL}" if j == i else tok
                    for j, tok in enumerate(tokens)
                )
            lines.append(
                f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Sub,,0,0,0,,{prefix}{body}\n"
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
    font_size: int = Form(72),           # taille des sous-titres (réduite auto si trop large)
    sub_y: int = Form(1450),             # hauteur fixe des sous-titres (0 = haut, 1920 = bas)
    max_words: int = Form(3),            # mots affichés en même temps
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

            (job / "subs.ass").write_text(build_ass(words, title, uppercase, duration, font_size, sub_y, max_words), encoding="utf-8")

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

# ==========================================================================
# MODE « HISTOIRE » : voix + script (plans) + images -> vidéo montée
# ==========================================================================
W, H, FPS = 1080, 1920, 30
MOTIONS = ["zoom_in", "pan_right", "zoom_out", "pan_left"]

_AR_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u0640]")   # tashkeel + tatweel
_STAGE_DIR = re.compile(r"\[[^\]]*\]")                      # [pause], [ton grave]...


def norm_token(token: str) -> str:
    """Forme de comparaison : sans voyelles brèves, hamzas/alef unifiés, sans ponctuation."""
    t = unicodedata.normalize("NFKC", token)
    t = _DIACRITICS.sub("", t).translate(_AR_MAP)
    return re.sub(r"[^\w]", "", t).lower()


def display_token(token: str) -> str:
    """Forme affichée dans les sous-titres : sans tashkeel."""
    return _DIACRITICS.sub("", unicodedata.normalize("NFKC", token)).strip()


def parse_script(script_json: str) -> dict:
    try:
        data = json.loads(script_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"script_json invalide : {e}")
    shots = data.get("shots") if isinstance(data, dict) else None
    if not shots or not isinstance(shots, list):
        raise HTTPException(status_code=422, detail="script_json doit contenir une liste 'shots'")
    clean = []
    for i, s in enumerate(shots):
        text = _STAGE_DIR.sub(" ", str(s.get("text", ""))).strip()
        if not any(norm_token(t) for t in text.split()):
            raise HTTPException(status_code=422, detail=f"Plan {i + 1} : texte vide")
        motion = s.get("motion") or MOTIONS[i % len(MOTIONS)]
        if motion not in MOTIONS:
            raise HTTPException(status_code=422, detail=f"Plan {i + 1} : motion '{motion}' inconnu ({', '.join(MOTIONS)})")
        clean.append({"text": text, "motion": motion})
    return {"title": str(data.get("title", "")).strip(), "shots": clean}


def align_script(script: dict, heard: list, audio_dur: float):
    """Associe chaque mot du script à un instant de la voix.

    heard : mots entendus par Whisper [{"w","start","end"}] (leurs fautes d'orthographe
    importent peu). Les mots du script sans correspondance sont interpolés entre leurs
    voisins reconnus. Retourne (mots_minutés, ratio_de_correspondance).
    """
    toks = []
    for si, shot in enumerate(script["shots"]):
        for raw in shot["text"].split():
            n = norm_token(raw)
            if n:
                toks.append({"shot": si, "w": display_token(raw), "n": n})
    heard = [h for h in heard if norm_token(h["w"])]
    if not heard:
        raise HTTPException(status_code=422, detail="Aucune parole détectée dans l'audio")

    sm = difflib.SequenceMatcher(None, [t["n"] for t in toks], [norm_token(h["w"]) for h in heard], autojunk=False)
    anchors = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            anchors[blk.a + k] = blk.b + k
    ratio = len(anchors) / len(toks)

    n = len(toks)
    for i, j in anchors.items():
        toks[i]["start"], toks[i]["end"] = heard[j]["start"], heard[j]["end"]

    known = sorted(anchors)
    t_first, t_last = heard[0]["start"], max(audio_dur, heard[-1]["end"])
    prev_i, prev_end = -1, t_first
    nxt = iter(known + [n])
    for q in nxt:
        gap_start = prev_end
        gap_end = toks[q]["start"] if q < n else t_last
        gap = max(0.0, gap_end - gap_start)
        count = q - prev_i - 1
        for k in range(count):
            t = toks[prev_i + 1 + k]
            t["start"] = round(gap_start + gap * k / count, 3)
            t["end"] = round(gap_start + gap * (k + 1) / count, 3)
        if q < n:
            prev_i, prev_end = q, toks[q]["end"]
    return toks, ratio


def shot_starts(toks: list, n_shots: int, lead: float) -> list:
    first = {}
    for t in toks:
        first.setdefault(t["shot"], t["start"])
    starts = [0.0]
    for i in range(1, n_shots):
        starts.append(max(starts[-1] + 0.8, first[i] - lead))   # coupe un peu avant le mot
    return starts


def compute_timing(voice: Path, duration: float, script: dict, language: str,
                   words_json: str, lead: float) -> dict:
    if words_json.strip():
        heard = json.loads(words_json)
        heard = heard["words"] if isinstance(heard, dict) else heard
    else:
        heard = transcribe_file(voice, language, "")["words"]
    toks, ratio = align_script(script, heard, duration)
    total = duration + 0.6
    fallback = ratio < 0.5
    if fallback:
        # Correspondance trop faible : on répartit les mots sur la période où la voix
        # parle réellement, au prorata du nombre de lettres de chaque mot.
        t0 = heard[0]["start"] if isinstance(heard[0], dict) else 0.0
        t1 = max(h["end"] for h in heard)
        weights = [len(t["n"]) + 1 for t in toks]
        cum, tot = 0, float(sum(weights))
        for t, wgt in zip(toks, weights):
            t["start"] = round(t0 + (t1 - t0) * cum / tot, 3)
            cum += wgt
            t["end"] = round(t0 + (t1 - t0) * cum / tot, 3)
    starts = shot_starts(toks, len(script["shots"]), lead)
    return {"toks": toks, "ratio": ratio, "starts": starts, "total": total,
            "duration": duration, "fallback": fallback}


def ken_burns_clip(img: Path, out: Path, seconds: float, motion: str, strength: float):
    """Image fixe -> clip 1080x1920 avec zoom/panoramique doux (ease in-out)."""
    frames = max(2, math.ceil(seconds * FPS))
    p = f"(on/{frames})"
    e = f"({p}*{p}*(3-2*{p}))"
    pan = strength * 1.4
    cy = "ih/2-ih/zoom/2"
    if motion == "zoom_out":
        z, x = f"{1 + strength}-{strength}*{e}", "iw/2-iw/zoom/2"
    elif motion == "pan_left":
        z, x = f"{1 + pan}", f"(iw-iw/zoom)*(1-{e})"
    elif motion == "pan_right":
        z, x = f"{1 + pan}", f"(iw-iw/zoom)*{e}"
    else:  # zoom_in
        z, x = f"1+{strength}*{e}", "iw/2-iw/zoom/2"
    big_w, big_h = W * KB_SCALE, H * KB_SCALE
    vf = (
        f"scale={big_w}:{big_h}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={big_w}:{big_h},setsar=1,"
        f"zoompan=z='{z}':x='{x}':y='{cy}':d={frames}:s={W}x{H}:fps={FPS},format=yuv420p"
    )
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(img), "-vf", vf,
           "-frames:v", str(frames), "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12",
           "-pix_fmt", "yuv420p", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if res.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Clip {out.name} : {res.stderr[-1200:]}")


@app.post("/align", dependencies=[Depends(check_key)])
def align(
    audio: UploadFile = File(...),
    script_json: str = Form(...),
    language: str = Form("ar"),
    words_json: str = Form(""),
    lead: float = Form(0.12),
):
    """Diagnostic : montre comment la voix se découpe en plans, sans rien rendre."""
    script = parse_script(script_json)
    job = JOBS / uuid.uuid4().hex
    job.mkdir(parents=True)
    try:
        voice = save_upload(audio, job, "voice")
        duration = probe_duration(voice)
        with WORK:
            t = compute_timing(voice, duration, script, language, words_json, lead)
        starts, total = t["starts"], t["total"]
        shots = []
        for i, s in enumerate(script["shots"]):
            end = starts[i + 1] if i + 1 < len(starts) else total
            shots.append({"shot": i + 1, "start": round(starts[i], 2), "end": round(end, 2),
                          "seconds": round(end - starts[i], 2), "motion": s["motion"], "text": s["text"]})
        return {"audio_seconds": round(duration, 2), "match_ratio": round(t["ratio"], 3),
                "fallback_proportional": t["fallback"],
                "warning": "Faible correspondance : minutage réparti au prorata du texte. Relis le script tel quel ou essaie WHISPER_MODEL=medium"
                if t["fallback"] else None, "shots": shots}
    finally:
        shutil.rmtree(job, ignore_errors=True)


@app.post("/render_story", dependencies=[Depends(check_key)])
def render_story(
    audio: UploadFile = File(...),
    images: List[UploadFile] = File(...),   # un fichier par plan, triés par nom (shot_01, shot_02...)
    script_json: str = Form(...),           # {"title": "...", "shots": [{"text": "...", "motion": "zoom_in"}]}
    language: str = Form("ar"),
    words_json: str = Form(""),             # optionnel : saute Whisper
    music: str = Form("none"),              # 'none', 'random' ou nom de fichier
    music_volume: float = Form(0.25),       # niveau de base, abaissé automatiquement sous la voix
    font_name: str = Form("Tajawal"),       # police uploadée dans /assets/fonts (sinon repli automatique)
    font_size: int = Form(76),
    sub_y: int = Form(1400),
    max_words: int = Form(3),
    transition: float = Form(0.4),
    kb_strength: float = Form(0.12),
    lead: float = Form(0.12),
    crf: int = Form(20),
    preset: str = Form("medium"),
    denoise: bool = Form(True),
    show_title: bool = Form(True),
    max_mb: float = Form(45.0),             # taille maximale visée (limite des bots Telegram : 50 Mo)
):
    script = parse_script(script_json)
    shots = script["shots"]
    imgs = sorted(images, key=lambda u: u.filename or "")
    if len(imgs) != len(shots):
        raise HTTPException(status_code=422, detail=f"{len(shots)} plans dans le script mais {len(imgs)} images reçues")

    job = JOBS / uuid.uuid4().hex
    job.mkdir(parents=True)
    try:
        voice = save_upload(audio, job, "voice")
        img_paths = [save_upload(u, job, f"img_{i:03d}") for i, u in enumerate(imgs)]
        duration = probe_duration(voice)

        with WORK:
            t = compute_timing(voice, duration, script, language, words_json, lead)
            starts, total, toks = t["starts"], t["total"], t["toks"]
            n = len(shots)
            durs = [(starts[i + 1] if i + 1 < n else total) - starts[i] for i in range(n)]
            T = max(0.1, min(transition, min(durs)))

            # 1) un clip par plan (zoom/pan doux), chevauchement pour les fondus
            clips = []
            for i, (img, shot) in enumerate(zip(img_paths, shots)):
                pre = 0.0 if i == 0 else T / 2
                post = 0.0 if i == n - 1 else T / 2
                clip = job / f"clip_{i:03d}.mp4"
                ken_burns_clip(img, clip, durs[i] + pre + post, shot["motion"], kb_strength)
                clips.append(clip)

            # 2) sous-titres : le TEXTE du script, le TIMING de la voix
            words = [{"w": x["w"], "start": x["start"], "end": x["end"]} for x in toks]
            title = script["title"] if show_title else ""
            (job / "subs.ass").write_text(
                build_ass(words, title, False, total, font_size, sub_y, max_words,
                          font_name=font_name, max_chars=24, char_factor=0.5, rtl=True),
                encoding="utf-8")

            mus = pick_asset("music", music, MUSIC_EXT, "none")
            fonts = ASSETS / "fonts"
            fontsdir = f":fontsdir={fonts}" if any(fonts.iterdir()) else ""

            cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
            for c in clips:
                cmd += ["-i", c.name]
            cmd += ["-i", str(voice)]
            if mus:
                cmd += ["-stream_loop", "-1", "-i", str(mus)]
            vi, mi = n, n + 1

            # 3) fondus enchaînés : la transition n°i est centrée sur le début du plan i
            if n == 1:
                chain, last = "", "[0:v]"
            else:
                parts, last = [], "[0:v]"
                for i in range(1, n):
                    off = starts[i] - T / 2
                    out = f"[x{i}]"
                    parts.append(f"{last}[{i}:v]xfade=transition=fade:duration={T:.3f}:offset={off:.3f}{out}")
                    last = out
                chain = ";".join(parts) + ";"
            vgraph = (
                f"{chain}{last}eq=contrast=1.03:saturation=1.06,noise=alls=2:allf=t,"
                f"ass=subs.ass{fontsdir},fade=t=out:st={total - 0.5:.2f}:d=0.5,format=yuv420p[v]"
            )

            # 4) voix nettoyée + musique qui s'efface sous la voix + loudness réseaux sociaux
            voice_chain = ("highpass=f=80," + ("afftdn=nr=10:nf=-40," if denoise else "") +
                           "acompressor=threshold=0.125:ratio=3:attack=5:release=80,"
                           f"apad=whole_dur={total:.2f}")
            loud = "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=44100"
            if mus:
                agraph = (
                    f"[{vi}:a]{voice_chain},asplit=2[vm][vs];"
                    f"[{mi}:a]volume={music_volume}[m0];"
                    f"[m0][vs]sidechaincompress=threshold=0.02:ratio=8:attack=15:release=450[m];"
                    f"[vm][m]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
                    f"afade=t=out:st={total - 1.5:.2f}:d=1.5,{loud}[a]"
                )
            else:
                agraph = f"[{vi}:a]{voice_chain},afade=t=out:st={total - 1.0:.2f}:d=1.0,{loud}[a]"

            # débit vidéo plafonné pour que le fichier reste sous max_mb (audio 192 kb/s inclus)
            vkbps = int(max(800, min(12000, max_mb * 8192 / total - 200)))
            cmd += ["-filter_complex", f"{vgraph};{agraph}", "-map", "[v]", "-map", "[a]",
                    "-t", f"{total:.2f}", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                    "-maxrate", f"{vkbps}k", "-bufsize", f"{2 * vkbps}k", "-profile:v", "high", "-r", str(FPS),
                    "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
                    "-movflags", "+faststart", "story.mp4"]
            res = subprocess.run(cmd, cwd=job, capture_output=True, text=True, timeout=1800)
            if res.returncode != 0:
                raise HTTPException(status_code=500, detail=f"FFmpeg a échoué : {res.stderr[-1800:]}")

        return FileResponse(
            job / "story.mp4", media_type="video/mp4", filename="story.mp4",
            headers={"X-Align-Ratio": f"{t['ratio']:.3f}", "X-Duration": f"{total:.2f}"},
            background=BackgroundTask(shutil.rmtree, job, ignore_errors=True),
        )
    except Exception:
        shutil.rmtree(job, ignore_errors=True)
        raise
