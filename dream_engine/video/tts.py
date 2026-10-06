"""Narration with the Windows speech engine (System.Speech): free, offline.

All sentences are synthesised in one PowerShell call, one WAV per sentence, so the
video can time each subtitle to its spoken sentence. Returns None where Windows
speech is unavailable; the video then uses reading-speed timing and no audio.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import wave
from pathlib import Path

PS_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$job = Get-Content -Raw -Encoding UTF8 $args[0] | ConvertFrom-Json
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
if ($job.voice) { $s.SelectVoice($job.voice) }
$s.Rate = [int]$job.rate
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
for ($i = 0; $i -lt $job.sentences.Count; $i++) {
  $s.SetOutputToWaveFile((Join-Path $job.dir ("s{0:D3}.wav" -f $i)), $fmt)
  $s.Speak([string]$job.sentences[$i])
}
$s.Dispose()
"""


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def _powershell() -> str | None:
    found = shutil.which("powershell") or shutil.which("pwsh")
    if found:
        return found
    # not always on PATH (e.g. when started from Git Bash); Windows keeps it in a fixed place
    root = os.environ.get("SystemRoot", r"C:\Windows")
    candidate = Path(root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    return str(candidate) if candidate.exists() else None


def synthesize(sentences: list[str], out_dir: Path, voice: str | None, rate: int) -> list[Path] | None:
    shell = _powershell()
    if not voice or not shell or not sentences:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    job = out_dir / "job.json"
    job.write_text(json.dumps({"voice": voice, "rate": rate, "dir": str(out_dir.resolve()), "sentences": sentences}),
                   encoding="utf-8")
    script = out_dir / "speak.ps1"
    script.write_text(PS_SCRIPT, encoding="utf-8")
    r = subprocess.run([shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), str(job)],
                       capture_output=True, text=True, timeout=600)
    wavs = [out_dir / f"s{i:03d}.wav" for i in range(len(sentences))]
    if r.returncode != 0 or not all(w.exists() for w in wavs):
        raise RuntimeError(f"Windows speech failed: {(r.stderr or r.stdout).strip()[:300]}")
    return wavs


def assemble(parts: list[tuple[Path | None, float]], out: Path) -> None:
    """Concatenate WAVs and silences: ``(wav, 0)`` plays a file, ``(None, seconds)`` is silence."""
    params = None
    for p, _ in parts:
        if p is not None:
            with wave.open(str(p), "rb") as w:
                params = w.getparams()
            break
    if params is None:
        return
    with wave.open(str(out), "wb") as o:
        o.setparams(params)
        silence_frame = b"\x00" * params.sampwidth * params.nchannels
        for p, secs in parts:
            if p is None:
                o.writeframes(silence_frame * int(round(secs * params.framerate)))
            else:
                with wave.open(str(p), "rb") as w:
                    o.writeframes(w.readframes(w.getnframes()))
