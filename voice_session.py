from __future__ import annotations

import asyncio
import contextlib
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import discord

from .ai_client import AIClient
from .config import Settings

if TYPE_CHECKING:
    from main import VoiceAIBot


class VoiceSession:
    """Stable voice session for Discord.py DAVE/E2EE voice.

    This patch intentionally uses the normal discord.VoiceClient only.
    Direct mic receive/listening is disabled because the old voice-receive
    extension does not handle Discord's DAVE/E2EE voice path reliably and was
    causing 4017 / corrupted-stream loops.
    """

    def __init__(
        self,
        bot: "VoiceAIBot",
        guild: discord.Guild,
        text_channel: discord.abc.Messageable,
        ai: AIClient,
        settings: Settings,
    ):
        self.bot = bot
        self.guild = guild
        self.text_channel = text_channel
        self.ai = ai
        self.settings = settings
        self.voice_client: discord.VoiceClient | None = None
        self.processing_lock = asyncio.Lock()
        self.playback_lock = asyncio.Lock()
        self.connect_lock = asyncio.Lock()
        self.temp_dir = Path(tempfile.gettempdir()) / "discord_voice_ai_bot" / str(guild.id)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.listening = False
        self.speech_enabled = True

    def is_connected(self) -> bool:
        return bool(self.voice_client and self.voice_client.is_connected())

    def is_listening(self) -> bool:
        return False

    def is_playing(self) -> bool:
        if not self.voice_client:
            return False
        with contextlib.suppress(Exception):
            return bool(self.voice_client.is_playing())
        return False

    async def connect(self, channel: discord.VoiceChannel | discord.StageChannel, *, receive: bool = False) -> None:
        """Connect to voice with DAVE-capable discord.py.

        If the server had a stale voice session, `/resetvoice` + this method
        force-cleans the old client first. Reconnect is disabled so discord.py
        does not loop forever when Discord rejects the handshake.
        """
        async with self.connect_lock:
            if self.voice_client and self.voice_client.is_connected():
                if self.voice_client.channel.id != channel.id:
                    await self.voice_client.move_to(channel)
                return

            # Clean any stale voice client stored by discord.py for this guild.
            existing = discord.utils.get(self.bot.voice_clients, guild=self.guild)
            if existing:
                with contextlib.suppress(Exception):
                    if existing.is_playing():
                        existing.stop()
                with contextlib.suppress(Exception):
                    await existing.disconnect(force=True)
                await asyncio.sleep(1.5)

            # Send a null voice state update to clear Discord-side stale state.
            with contextlib.suppress(Exception):
                await self.guild.change_voice_state(channel=None)
            await asyncio.sleep(1.0)

            try:
                self.voice_client = await channel.connect(
                    cls=discord.VoiceClient,
                    timeout=30.0,
                    reconnect=False,
                    self_deaf=False,
                    self_mute=False,
                )
            except asyncio.TimeoutError as exc:
                self.voice_client = None
                raise RuntimeError(
                    "Timeout connect voice. Ini biasanya voice state Discord masih nyangkut. "
                    "Stop server 10 detik, start lagi, lalu coba `/join`."
                ) from exc
            except discord.errors.ConnectionClosed as exc:
                self.voice_client = None
                code = getattr(exc, "code", None)
                if code == 4017:
                    raise RuntimeError(
                        "Discord menolak voice dengan kode 4017 (DAVE/E2EE). "
                        "Jalankan `pip install --upgrade --force-reinstall -r requirements.txt` "
                        "supaya discord.py dan davey ter-update."
                    ) from exc
                raise RuntimeError(f"Discord menutup koneksi voice. Code: {code}. Detail: {exc}") from exc
            except Exception as exc:  # noqa: BLE001
                self.voice_client = None
                raise RuntimeError(f"Gagal connect voice: {exc}") from exc

    async def start_listening(self) -> None:
        raise RuntimeError(
            "mic langsung dinonaktifkan dulu. Penyebab error 4017 adalah Discord DAVE/E2EE; "
            "patch ini menstabilkan /join, /say, dan /talk dulu."
        )

    async def stop_listening(self) -> None:
        self.listening = False

    async def disconnect(self) -> None:
        await self.stop_listening()
        vc = self.voice_client or discord.utils.get(self.bot.voice_clients, guild=self.guild)
        if vc:
            with contextlib.suppress(Exception):
                if vc.is_playing():
                    vc.stop()
            with contextlib.suppress(Exception):
                await vc.disconnect(force=True)
        with contextlib.suppress(Exception):
            await self.guild.change_voice_state(channel=None)
        self.voice_client = None

    async def process_user_pcm(self, member: discord.Member, pcm: bytes) -> None:  # kept for compatibility
        return

    async def speak(self, text: str, *, force: bool = False, premium: bool = False) -> None:
        if not self.voice_client or not self.voice_client.is_connected():
            await self._safe_send("Aku belum masuk voice channel. Pakai `/join` dulu.")
            return
        if not self.speech_enabled and not force:
            return

        async with self.playback_lock:
            timestamp = int(time.time() * 1000)
            wav_path = self.temp_dir / f"tts_{timestamp}.wav"
            try:
                await self.ai.text_to_speech(text, wav_path, engine="gemini" if premium else "piper")
                await self._play_audio_file(wav_path)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    wav_path.unlink()

    async def stop_speaking(self) -> None:
        if self.voice_client and self.voice_client.is_connected():
            with contextlib.suppress(Exception):
                if self.voice_client.is_playing():
                    self.voice_client.stop()

    async def _play_audio_file(self, audio_path: Path) -> None:
        if not self.voice_client:
            return

        while self.voice_client.is_playing():
            await asyncio.sleep(0.1)

        done = asyncio.Event()

        def after(error: Exception | None) -> None:
            if error:
                print(f"[playback-error] {self.guild.id}: {error}")
            self.bot.loop.call_soon_threadsafe(done.set)

        source = discord.FFmpegPCMAudio(str(audio_path))
        source = discord.PCMVolumeTransformer(source, volume=0.95)
        self.voice_client.play(source, after=after)
        await done.wait()

    async def _safe_send(self, content: str) -> None:
        with contextlib.suppress(Exception):
            await self.text_channel.send(content)
