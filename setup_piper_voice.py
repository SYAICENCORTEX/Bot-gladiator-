from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "id_ID-news_tts-medium.onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/id/id_ID/news_tts/medium/id_ID-news_tts-medium.onnx",
    "id_ID-news_tts-medium.onnx.json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/id/id_ID/news_tts/medium/id_ID-news_tts-medium.onnx.json",
}


def download(url: str, dest: Path, *, retries: int = 3) -> None:
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"✅ Sudah ada: {dest}")
        return

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        print(f"⬇️ Download {dest.name} (percobaan {attempt}/{retries})")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "discord-voice-ai-bot/1.0"})
            with urllib.request.urlopen(req, timeout=180) as response:
                total = response.headers.get("Content-Length")
                total_int = int(total) if total and total.isdigit() else None
                downloaded = 0
                with tmp.open("wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_int:
                            percent = downloaded * 100 / total_int
                            print(f"   {percent:5.1f}%", end="\r", flush=True)
                tmp.replace(dest)
                print(f"✅ Selesai: {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
                return
        except Exception as exc:  # noqa: BLE001
            print(f"❌ Gagal: {type(exc).__name__}: {exc}")
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            if attempt < retries:
                time.sleep(2)

    print(f"❌ Download {dest.name} gagal total. Coba jalankan lagi nanti.")
    sys.exit(1)


def main() -> None:
    for filename, url in FILES.items():
        download(url, MODEL_DIR / filename)

    print("\n✅ Model Piper sudah siap. Tidak perlu ubah .env kalau /say sudah dipaksa lokal.")
    print("Sekarang langsung restart/start server lalu tes: /say text: halo tes suara lokal")


if __name__ == "__main__":
    main()
