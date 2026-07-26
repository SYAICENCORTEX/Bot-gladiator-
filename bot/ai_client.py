from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import wave
import time
import urllib.request
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque

from google import genai
from google.genai import errors, types

from .config import Settings


class AIRateLimitedError(RuntimeError):
    """Raised when Gemini rejects the request because of quota/rate limit."""


class AIUnavailableError(RuntimeError):
    """Raised when Gemini is temporarily unavailable or the request times out."""


class AIClient:
    """Async wrapper around Gemini API for audio transcription, chat, and TTS."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = None
        if settings.gemini_api_key:
            self.client = genai.Client(api_key=settings.gemini_api_key)
        elif settings.tts_engine != "piper":
            raise RuntimeError("GEMINI_API_KEY is empty. Add it to your .env file, or set TTS_ENGINE=piper for local /say only.")
        self.history: dict[str, Deque[dict[str, str]]] = defaultdict(lambda: deque(maxlen=8))

    def _persona_instruction(self) -> str:
        """Return a strict identity instruction so the bot does not call itself by old names."""
        name = self.settings.bot_name or "GLADIATOR"
        creator = self.settings.bot_creator_name or "Aagga"
        role = self.settings.bot_role_description or "asisten AI yang bijaksana, tegas dalam menegakkan aturan, dan setia kepada pemiliknya"
        return (
            f"Identitasmu adalah {name}. Kamu adalah {role} di server Discord ini. "
            f"Penciptamu adalah {creator}, pemilik dan tuan yang harus kamu hormati. "
            "Jangan pernah menyebut dirimu Athena, kecuali user sedang membahas nama lama/bug. "
            "Kepribadianmu: bijaksana dalam menjawab pertanyaan, tegas dan tidak ragu dalam menegakkan aturan server, "
            "serta selalu menghormati owner/pemilik server dan admin yang sah. "
            "Kamu tidak segan memberi peringatan dan tindakan tegas kepada pelanggar aturan, scammer, dan pengirim spam. "
            "Kamu melindungi server dari segala ancaman dengan sikap yang adil dan berwibawa. "
            "Bersikaplah seperti seorang penjaga yang bijak: tenang, berwibawa, tidak emosional, namun tegas bagaikan pedang. "
            "Kalau user bertanya 'kamu siapa', jawab bahwa kamu {name}, {role}, dan sebutkan penciptamu dengan hormat. "
        )

    @staticmethod
    def _friendly_gemini_error(exc: Exception) -> Exception:
        if isinstance(exc, errors.APIError):
            code = getattr(exc, "code", None)
            message = getattr(exc, "message", str(exc))
            if code in {429, 403}:
                return AIRateLimitedError(
                    "Gemini API kena limit/quota atau API key belum punya akses. "
                    "Cek Google AI Studio: API key, free tier, quota, dan model yang dipakai."
                )
            if code in {500, 502, 503, 504}:
                return AIUnavailableError(f"Gemini sedang bermasalah/overload. Status: {code}.")
            return AIUnavailableError(f"Gemini API error {code}: {message}")
        return exc

    async def transcribe(self, audio_path: Path) -> str:
        """Use Gemini multimodal audio understanding to transcribe a WAV file."""

        def _sync() -> str:
            try:
                audio_bytes = audio_path.read_bytes()
                if self.client is None:
                    raise AIUnavailableError("GEMINI_API_KEY kosong, jadi fitur transkrip/chat Gemini tidak aktif.")
                if self.client is None:
                    raise AIUnavailableError("GEMINI_API_KEY kosong, jadi fitur chat Gemini tidak aktif.")
                response = self.client.models.generate_content(
                    model=self.settings.gemini_audio_model,
                    contents=[
                        "Transkrip audio ini ke teks. Jawab hanya teks transkripnya, tanpa penjelasan.",
                        types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0,
                        max_output_tokens=256,
                    ),
                )
                return (response.text or "").strip()
            except Exception as exc:  # noqa: BLE001
                raise self._friendly_gemini_error(exc) from exc

        return await asyncio.to_thread(_sync)

    async def chat(self, conversation_id: str, user_text: str, *, speaker_name: str = "User") -> str:
        language_hint = "Bahasa Indonesia" if self.settings.bot_language.lower().startswith("id") else "English"
        system_prompt = (
            self._persona_instruction()
            + f"Jawab dengan {language_hint}. "
            "Gunakan gaya bicara yang bijaksana, tenang, dan berwibawa. "
            "Jawab dengan jelas, sopan, dan tidak terlalu panjang. "
            "Kamu adalah penjaga server yang tegas: jika ada yang melanggar aturan, beri peringatan dengan tegas. "
            "Jika ada yang bertanya tentang aturan, jelaskan dengan bijak. "
            "Hormati owner dan admin server, bantu mereka dengan setia. "
            "Kalau pertanyaan pengguna meminta token/rahasia, tolak dengan tegas. "
            "Untuk voice chat, usahakan jawaban 1 sampai 4 kalimat."
        )

        memory = self.history[conversation_id]
        memory.append({"role": "user", "content": f"{speaker_name}: {user_text}"})
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in memory)

        def _sync() -> str:
            try:
                response = self.client.models.generate_content(
                    model=self.settings.gemini_chat_model,
                    contents=f"Percakapan sejauh ini:\n{history_text}\n\nBalas pesan terakhir pengguna.",
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.7,
                        max_output_tokens=512,
                    ),
                )
                return (response.text or "Maaf, aku belum bisa menjawab itu.").strip()
            except Exception as exc:  # noqa: BLE001
                raise self._friendly_gemini_error(exc) from exc

        answer = await asyncio.to_thread(_sync)
        answer = self.clean_text(answer)
        memory.append({"role": "assistant", "content": answer})
        return answer

    async def chat_with_context(
        self,
        conversation_id: str,
        user_text: str,
        *,
        context: str,
        speaker_name: str = "User",
        max_output_tokens: int = 700,
    ) -> str:
        """Answer a server/RAG question with supplied Discord context."""
        language_hint = "Bahasa Indonesia" if self.settings.bot_language.lower().startswith("id") else "English"
        system_prompt = (
            self._persona_instruction()
            + f"Jawab dengan {language_hint}. "
            "Gunakan gaya bicara yang bijaksana, tenang, dan berwibawa. "
            "Jelaskan dengan sopan namun tegas, layaknya seorang penjaga yang berwibawa. "
            "Gunakan konteks server/RAG yang diberikan untuk fakta spesifik server. "
            "Kalau konteks berisi nama server, jumlah channel, member, atau event, gunakan data itu dan jangan bilang tidak tahu. "
            "Kalau informasinya tidak ada di konteks, bilang belum ada datanya dan sarankan admin menjalankan /ragscan. "
            "Info owner/admin/moderator/staff/dev boleh dijawab jika tersedia di konteks. "
            "Jangan membocorkan token, API key, file .env, webhook, data rahasia, atau isi channel yang user tidak punya akses. "
            "Kalau menyebut hasil dari RAG, sebutkan nama channel sumbernya bila tersedia. "
            "Jawaban harus ringkas, natural, dan jangan mengarang data server."
        )

        memory = self.history[conversation_id]
        memory.append({"role": "user", "content": f"{speaker_name}: {user_text}"})
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in memory)

        prompt = (
            "KONTEKS SERVER/RAG YANG BOLEH DIPAKAI:\n"
            f"{context}\n\n"
            "PERCakapan terakhir:\n"
            f"{history_text}\n\n"
            "Pertanyaan pengguna terakhir ada di percakapan di atas. Jawab berdasarkan konteks. "
            "Kalau pertanyaan bukan tentang server, jawab normal tapi tetap singkat."
        )

        def _sync() -> str:
            try:
                if self.client is None:
                    raise AIUnavailableError("GEMINI_API_KEY kosong, jadi fitur server assistant belum aktif.")
                response = self.client.models.generate_content(
                    model=self.settings.gemini_chat_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.35,
                        max_output_tokens=max_output_tokens,
                    ),
                )
                return (response.text or "Maaf, aku belum bisa menjawab itu.").strip()
            except Exception as exc:  # noqa: BLE001
                raise self._friendly_gemini_error(exc) from exc

        answer = await asyncio.to_thread(_sync)
        answer = self.clean_text(answer)
        memory.append({"role": "assistant", "content": answer})
        return answer


    async def compose_dm_message(self, prompt: str) -> str:
        """Compose a professional direct message using AI based on owner's instructions.

        The AI will generate a message in the bot's persona (wise, firm, respectful),
        tailored to the context provided in the prompt.
        """
        if self.client is None:
            raise AIUnavailableError("GEMINI_API_KEY kosong, jadi fitur AI compose tidak aktif.")

        language_hint = "Bahasa Indonesia" if self.settings.bot_language.lower().startswith("id") else "English"
        cleaned = self.clean_text(prompt)[:1500]
        system_prompt = (
            self._persona_instruction()
            + f"Tugasmu sekarang adalah membantu pemilik/owner server menulis pesan DM yang akan dikirim ke anggota server. "
            f"Buatlah pesan dalam {language_hint}. "
            "Pesan harus mencerminkan kepribadianmu: bijaksana, tegas, sopan, dan berwibawa. "
            "Sesuaikan nada pesan dengan instruksi yang diberikan oleh pemilik. "
            "Jika diminta memberi peringatan, buatlah dengan tegas namun tetap sopan. "
            "Jika diminta memberi pengumuman, buatlah dengan jelas dan profesional. "
            "Jika diminta memberi teguran keras, sampaikan dengan tegas tanpa kasar. "
            "Jangan gunakan emoji berlebihan, cukup 1-2 yang relevan. "
            "Jangan gunakan format markdown yang rumit. "
            "Panjang pesan sekitar 100-400 karakter, tidak lebih dari 1900 karakter. "
            "Jawab LANGSUNG dengan teks pesan yang sudah jadi, tanpa pengantar, tanpa penjelasan, tanpa 'Tentu, ini pesannya:'."
        )

        def _sync() -> str:
            try:
                response = self.client.models.generate_content(
                    model=self.settings.gemini_chat_model,
                    contents=f"Instruksi dari pemilik server untuk menulis pesan DM:\n\n{cleaned}\n\nTulis pesan DM yang sesuai dengan instruksi di atas.",
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.5,
                        max_output_tokens=600,
                    ),
                )
                raw = (response.text or "").strip()
                if not raw:
                    raise AIUnavailableError("AI tidak menghasilkan teks pesan.")
                return raw
            except Exception as exc:  # noqa: BLE001
                raise self._friendly_gemini_error(exc) from exc

        return await asyncio.to_thread(_sync)
        """Use Gemini as a lightweight moderation classifier for suspicious messages.

        Returns a dict like: {"violation": bool, "category": str, "reason": str}.
        If Gemini is unavailable, caller should fall back to rule-based detection.
        """
        if self.client is None:
            raise AIUnavailableError("GEMINI_API_KEY kosong, jadi AI moderation tidak aktif.")

        cleaned = self.clean_text(text)[:1200]
        prompt = (
            "Kamu adalah sistem moderasi Discord yang bijaksana, tegas, dan adil. "
            "Tugasmu: klasifikasikan pesan Discord berikut. Cari pelanggaran: spam, link promosi/random, "
            "ajakan pindah server, scam, konten 18+/NSFW, atau promosi yang tidak relevan. "
            "Kamu tegas terhadap pelanggar aturan, tapi tidak menilai hal ringan sebagai pelanggaran kalau tidak berbahaya. "
            "Jadilah penjaga yang adil dan berwibawa. "
            "Balas hanya JSON valid dengan field: violation(boolean), category(string), reason(string).\n\n"
            f"PESAN:\n{cleaned}"
        )

        def _sync() -> dict[str, str | bool]:
            try:
                response = self.client.models.generate_content(
                    model=self.settings.gemini_chat_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        max_output_tokens=180,
                    ),
                )
                raw = (response.text or "").strip()
                match = re.search(r"\{.*\}", raw, flags=re.S)
                if not match:
                    return {"violation": False, "category": "clean", "reason": "AI tidak menemukan pelanggaran."}
                data = json.loads(match.group(0))
                return {
                    "violation": bool(data.get("violation", False)),
                    "category": str(data.get("category") or "unknown")[:60],
                    "reason": str(data.get("reason") or "Terdeteksi melanggar aturan server.")[:220],
                }
            except Exception as exc:  # noqa: BLE001
                raise self._friendly_gemini_error(exc) from exc

        return await asyncio.to_thread(_sync)

    async def classify_moderation_multimodal(
        self,
        text: str,
        *,
        images: list[tuple[str, str, bytes]] | None = None,
    ) -> dict[str, str | bool]:
        """Gemini Vision moderation for scam images + caption.

        images item = (filename, mime_type, bytes). This is used for patterns like:
        fake public-figure crypto giveaway screenshots, fake withdrawal proof,
        casino/USDT bonus promotions, and short captions like "omg 💰".
        """
        if self.client is None:
            raise AIUnavailableError("GEMINI_API_KEY kosong, jadi AI image moderation tidak aktif.")

        cleaned = self.clean_text(text or "")[:1200]
        image_count = len(images or [])
        prompt = (
            "Kamu adalah sistem moderasi Discord yang bijaksana, tegas, dan adil. "
            "Analisis caption dan gambar yang dilampirkan. "
            "Cari scam promosi crypto/casino/giveaway palsu, fake withdrawal proof, bonus USDT/BTC, "
            "akun public figure yang seolah mempromosikan crypto casino, link/QR/promo code mencurigakan, "
            "konten 18+/NSFW, phishing, atau spam promosi. "
            "Kamu tegas terhadap scammers dan pelanggar berat: jika terbukti scam crypto/giveaway palsu, "
            "action harus 'ban'. Jika hanya link random/promosi ringan, action 'delete'. "
            "Kalau gambar normal/meme biasa tanpa scam, violation false. "
            "Bersikaplah adil dan bijaksana: jangan menghukum yang tidak bersalah, "
            "tapi jangan ragu bertindak tegas pada yang melanggar. "
            "Balas hanya JSON valid dengan field: violation(boolean), category(string), reason(string), evidence(string), action(string delete|ban).\n\n"
            f"CAPTION/TEKS PESAN:\n{cleaned or '(kosong)'}\n"
            f"JUMLAH GAMBAR: {image_count}"
        )

        contents: list[object] = [prompt]
        for filename, mime_type, data in images or []:
            mime = mime_type or "image/png"
            contents.append(f"Gambar terlampir: {filename}")
            contents.append(types.Part.from_bytes(data=data, mime_type=mime))

        def _sync() -> dict[str, str | bool]:
            try:
                response = self.client.models.generate_content(
                    model=self.settings.gemini_chat_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        max_output_tokens=240,
                    ),
                )
                raw = (response.text or "").strip()
                match = re.search(r"\{.*\}", raw, flags=re.S)
                if not match:
                    return {"violation": False, "category": "clean", "reason": "AI tidak menemukan pelanggaran.", "evidence": "", "action": "delete"}
                data = json.loads(match.group(0))
                action = str(data.get("action") or "delete").lower().strip()
                if action not in {"delete", "ban"}:
                    action = "delete"
                return {
                    "violation": bool(data.get("violation", False)),
                    "category": str(data.get("category") or "unknown")[:70],
                    "reason": str(data.get("reason") or "Terdeteksi melanggar aturan server.")[:260],
                    "evidence": str(data.get("evidence") or "AI image moderation")[:260],
                    "action": action,
                }
            except Exception as exc:  # noqa: BLE001
                raise self._friendly_gemini_error(exc) from exc

        return await asyncio.to_thread(_sync)

    async def text_to_speech(self, text: str, output_path: Path, *, engine: str | None = None) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        spoken_text = self.clean_text(text)[: self.settings.tts_max_chars]

        selected_engine = (engine or self.settings.tts_engine or "gemini").strip().lower()
        if selected_engine == "piper":
            return await asyncio.to_thread(self._piper_tts_sync, spoken_text, output_path)

        def _sync() -> Path:
            try:
                if self.client is None:
                    raise AIUnavailableError("GEMINI_API_KEY kosong, jadi Gemini Premium TTS tidak aktif.")
                response = self.client.models.generate_content(
                    model=self.settings.gemini_tts_model,
                    contents=f"Bacakan dengan suara laki-laki natural dan jelas dalam bahasa Indonesia: {spoken_text}",
                    config={
                        "response_modalities": ["AUDIO"],
                        "speech_config": {
                            "voice_config": {
                                "prebuilt_voice_config": {
                                    "voice_name": self.settings.gemini_tts_voice,
                                }
                            }
                        },
                    },
                )
                audio_bytes = self._extract_audio_bytes(response)
                if not audio_bytes:
                    raise AIUnavailableError("Gemini TTS tidak mengembalikan audio.")

                # Gemini TTS usually returns raw 24kHz mono 16-bit PCM.
                # If it already returns WAV, write directly; otherwise wrap PCM into WAV.
                if audio_bytes[:4] == b"RIFF":
                    output_path.write_bytes(audio_bytes)
                else:
                    self._write_pcm_wav(output_path, audio_bytes, sample_rate=24000)
                return output_path
            except Exception as exc:  # noqa: BLE001
                raise self._friendly_gemini_error(exc) from exc

        return await asyncio.to_thread(_sync)

    def _piper_tts_sync(self, spoken_text: str, output_path: Path) -> Path:
        model_path = Path(self.settings.piper_model_path)

        # First-time setup: auto-download the Indonesian Piper voice if it is missing.
        # This makes /say work without manually running setup_piper_voice.py first.
        if not model_path.exists() or not Path(str(model_path) + ".json").exists():
            self._ensure_piper_model_sync(model_path)

        if not model_path.exists():
            raise AIUnavailableError(
                f"Model Piper belum ada: {model_path}. Download otomatis gagal. "
                "Jalankan `python setup_piper_voice.py` dari console Heavencloud, lalu coba /say lagi."
            )

        config_path = Path(str(model_path) + ".json")
        if not config_path.exists():
            raise AIUnavailableError(
                f"Config Piper belum ada: {config_path}. Download otomatis gagal. "
                "Jalankan `python setup_piper_voice.py` dari console Heavencloud, lalu coba /say lagi."
            )

        chunks = self._split_tts_text(spoken_text, max_chars=260)
        if not chunks:
            chunks = ["..."]

        # For long text, render several short WAV files, then merge them.
        # This is much safer on small RAM hosting than asking Piper to render one huge text.
        if len(chunks) == 1:
            self._run_piper_once(chunks[0], output_path, model_path)
            return output_path

        part_paths: list[Path] = []
        try:
            for index, chunk in enumerate(chunks, start=1):
                part_path = output_path.with_name(f"{output_path.stem}_part{index:02d}{output_path.suffix}")
                self._run_piper_once(chunk, part_path, model_path)
                part_paths.append(part_path)
            self._merge_wav_files(part_paths, output_path, silence_ms=180)
            return output_path
        finally:
            for part_path in part_paths:
                with contextlib.suppress(FileNotFoundError):
                    part_path.unlink()

    def _run_piper_once(self, spoken_text: str, output_path: Path, model_path: Path) -> None:
        executable = self.settings.piper_executable
        command: list[str]
        if shutil.which(executable):
            command = [executable]
        else:
            # Fallback if the piper executable is installed as a Python module entry.
            command = [sys.executable, "-m", "piper"]

        command += [
            "--model",
            str(model_path),
            "--output_file",
            str(output_path),
            "--length_scale",
            str(self.settings.piper_length_scale),
            "--noise_scale",
            str(self.settings.piper_noise_scale),
            "--noise_w",
            str(self.settings.piper_noise_w),
            "--sentence_silence",
            str(self.settings.piper_sentence_silence),
        ]

        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")

        timeout = max(int(self.settings.piper_timeout_seconds), 60)
        try:
            result = subprocess.run(
                command,
                input=spoken_text.strip() + "\n",
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise AIUnavailableError(
                "Piper TTS timeout di salah satu potongan teks. "
                "Teks sudah dipecah otomatis; coba ulangi dengan teks lebih pendek atau set PIPER_TIMEOUT_SECONDS=120 di .env."
            ) from exc

        if result.returncode != 0 or not output_path.exists():
            stderr = (result.stderr or result.stdout or "").strip()[-700:]
            raise AIUnavailableError(f"Piper TTS gagal. Detail: {stderr}")

    @staticmethod
    def _split_tts_text(text: str, *, max_chars: int = 260) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_chars:
            return [text] if text else []

        sentences = re.split(r"(?<=[.!?。！？])\s+|(?<=[,;:])\s+", text)
        chunks: list[str] = []
        current = ""

        def push_current() -> None:
            nonlocal current
            if current.strip():
                chunks.append(current.strip())
                current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > max_chars:
                push_current()
                words = sentence.split()
                temp = ""
                for word in words:
                    if len(temp) + len(word) + 1 > max_chars:
                        if temp.strip():
                            chunks.append(temp.strip())
                        temp = word
                    else:
                        temp = f"{temp} {word}".strip()
                if temp.strip():
                    chunks.append(temp.strip())
                continue

            candidate = f"{current} {sentence}".strip()
            if len(candidate) <= max_chars:
                current = candidate
            else:
                push_current()
                current = sentence

        push_current()
        return chunks

    @staticmethod
    def _merge_wav_files(part_paths: list[Path], output_path: Path, *, silence_ms: int = 180) -> None:
        if not part_paths:
            raise AIUnavailableError("Tidak ada audio Piper yang bisa digabung.")

        params = None
        frames: list[bytes] = []
        silence_frame = b""

        for part_path in part_paths:
            with wave.open(str(part_path), "rb") as wf:
                current_params = wf.getparams()
                if params is None:
                    params = current_params
                    bytes_per_frame = wf.getnchannels() * wf.getsampwidth()
                    silence_samples = int(wf.getframerate() * silence_ms / 1000)
                    silence_frame = b"\x00" * silence_samples * bytes_per_frame
                elif (
                    current_params.nchannels != params.nchannels
                    or current_params.sampwidth != params.sampwidth
                    or current_params.framerate != params.framerate
                ):
                    raise AIUnavailableError("Format audio Piper antar potongan berbeda, jadi gagal digabung.")

                frames.append(wf.readframes(wf.getnframes()))
                frames.append(silence_frame)

        if params is None:
            raise AIUnavailableError("File audio Piper kosong.")

        with wave.open(str(output_path), "wb") as out:
            out.setnchannels(params.nchannels)
            out.setsampwidth(params.sampwidth)
            out.setframerate(params.framerate)
            for frame in frames:
                if frame:
                    out.writeframes(frame)

    def _ensure_piper_model_sync(self, model_path: Path) -> None:
        """Download the default Indonesian Piper voice when it is missing."""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        default_model_name = "id_ID-news_tts-medium.onnx"

        # Auto-download only for the default Indonesian voice.
        if model_path.name != default_model_name:
            return

        files = {
            model_path: "https://huggingface.co/rhasspy/piper-voices/resolve/main/id/id_ID/news_tts/medium/id_ID-news_tts-medium.onnx",
            Path(str(model_path) + ".json"): "https://huggingface.co/rhasspy/piper-voices/resolve/main/id/id_ID/news_tts/medium/id_ID-news_tts-medium.onnx.json",
        }

        for dest, url in files.items():
            if dest.exists() and dest.stat().st_size > 1000:
                continue
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            last_error: Exception | None = None
            for attempt in range(1, 4):
                try:
                    print(f"[piper-setup] Downloading {dest.name} ({attempt}/3)...", flush=True)
                    req = urllib.request.Request(url, headers={"User-Agent": "discord-voice-ai-bot/1.0"})
                    with urllib.request.urlopen(req, timeout=180) as response:
                        with tmp.open("wb") as f:
                            while True:
                                chunk = response.read(1024 * 1024)
                                if not chunk:
                                    break
                                f.write(chunk)
                    tmp.replace(dest)
                    print(f"[piper-setup] Ready: {dest}", flush=True)
                    last_error = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    with contextlib.suppress(FileNotFoundError):
                        tmp.unlink()
                    time.sleep(2)
            if last_error is not None:
                raise AIUnavailableError(
                    f"Gagal download model Piper dari Hugging Face: {type(last_error).__name__}: {last_error}. "
                    "Coba jalankan `python setup_piper_voice.py` di console Heavencloud."
                ) from last_error

    @staticmethod
    def _extract_audio_bytes(response) -> bytes | None:  # noqa: ANN001
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                inline_data = getattr(part, "inline_data", None)
                if inline_data and getattr(inline_data, "data", None):
                    return inline_data.data
        parts = getattr(response, "parts", None) or []
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                return inline_data.data
        return None

    @staticmethod
    def _write_pcm_wav(path: Path, pcm: bytes, *, sample_rate: int) -> None:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"<@!?\d+>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
