# core/audio_utils.py
import subprocess, tempfile, os
from pathlib import Path

def ensure_wav_pcm16_16k(input_path: Path, *, loudnorm: bool=False) -> Path:
    out = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)
    af = ["aformat=sample_fmts=s16:sample_rates=16000:channel_layouts=mono"]
    if loudnorm:
        af.insert(0, "loudnorm=I=-16:TP=-1.5:LRA=11")
    cmd = ["ffmpeg","-y","-i",str(input_path),"-vn","-af",",".join(af),"-c:a","pcm_s16le",str(out)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out

def ensure_bytes_pcm16_16k(audio_bytes: bytes, *, loudnorm: bool=False) -> bytes:
    src = Path(tempfile.NamedTemporaryFile(delete=False).name)
    dst = None
    try:
        src.write_bytes(audio_bytes)
        dst = ensure_wav_pcm16_16k(src, loudnorm=loudnorm)
        return dst.read_bytes()
    finally:
        try: os.remove(src)
        except: pass
        if dst:
            try: os.remove(dst)
            except: pass

def is_probably_wav(path: Path) -> bool:
    b = path.read_bytes()[:12]
    return b.startswith(b"RIFF") and b[8:12] == b"WAVE"
