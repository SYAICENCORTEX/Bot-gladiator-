from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on", "enable", "enabled"}


def _get_csv(name: str) -> tuple[str, ...]:
    value = os.getenv(name, "").strip()
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _get_csv_int(name: str) -> tuple[int, ...]:
    result: list[int] = []
    for part in _get_csv(name):
        if part.isdigit():
            result.append(int(part))
    return tuple(result)


@dataclass(frozen=True)
class Settings:
    discord_token: str
    gemini_api_key: str
    discord_guild_id: int | None
    bot_prefix: str
    bot_name: str
    bot_creator_name: str
    bot_role_description: str
    bot_language: str
    gemini_chat_model: str
    gemini_audio_model: str
    gemini_tts_model: str
    gemini_tts_voice: str
    gemini_timeout_seconds: int
    max_voice_seconds: int
    min_voice_seconds: float
    tts_max_chars: int
    chat_cooldown_seconds: int
    voice_reply_cooldown_seconds: int
    tts_engine: str
    piper_executable: str
    piper_model_path: str
    piper_timeout_seconds: int
    piper_length_scale: float
    piper_noise_scale: float
    piper_noise_w: float
    piper_sentence_silence: float
    rag_scan_limit: int
    rag_top_k: int
    rag_max_context_chars: int
    admin_role_names: tuple[str, ...]
    admin_role_ids: tuple[int, ...]
    bot_owner_ids: tuple[int, ...]
    public_ai_commands: bool
    public_ai_voice_reply: bool
    moderation_enabled: bool
    moderation_ai_enabled: bool
    moderation_delete_links: bool
    moderation_dm_warnings: bool
    mod_log_channel_id: int | None
    mod_log_channel_names: tuple[str, ...]
    mod_allowed_link_domains: tuple[str, ...]
    mod_bypass_role_names: tuple[str, ...]
    mod_bypass_role_ids: tuple[int, ...]
    mod_spam_window_seconds: int
    mod_spam_message_limit: int
    mod_duplicate_message_limit: int
    mod_max_mentions: int
    moderation_image_ai_enabled: bool
    moderation_ban_crypto_scams: bool
    moderation_ban_dm_warnings: bool
    moderation_image_max_bytes: int
    mrbeast_scam_detection: bool
    mrbeast_scam_ban: bool
    celebrity_scam_detection: bool


def load_settings() -> Settings:
    guild_id_raw = os.getenv("DISCORD_GUILD_ID", "").strip()
    guild_id = int(guild_id_raw) if guild_id_raw.isdigit() else None

    mod_log_id_raw = os.getenv("MOD_LOG_CHANNEL_ID", "").strip()
    mod_log_channel_id = int(mod_log_id_raw) if mod_log_id_raw.isdigit() else None

    default_log_names = ("mod-log", "logs", "log-server", "server-log", "bot-log", "guard-log", "moderation-log")
    mod_log_names = _get_csv("MOD_LOG_CHANNEL_NAMES") or default_log_names

    default_bypass_names = ("Owner", "Admin", "Administrator", "Moderator", "Staff", "Developer", "Dev", "Bot Guard", "Guard", "GDev")
    mod_bypass_names = _get_csv("MOD_BYPASS_ROLE_NAMES") or default_bypass_names

    # Identity/persona. BOT_PERSONA_NAME is preferred so old .env values like
    # BOT_NAME=Athena Voice AI do not accidentally leak into the bot persona.
    bot_name = (
        os.getenv("BOT_PERSONA_NAME", "").strip()
        or os.getenv("BOT_DISPLAY_NAME", "").strip()
        or os.getenv("BOT_NAME", "").strip()
        or "GLADIATOR"
    )
    if bot_name.strip().lower().startswith("athena"):
        bot_name = "GLADIATOR"

    return Settings(
        discord_token=os.getenv("DISCORD_TOKEN", "").strip(),
        gemini_api_key=(
            os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
            or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY", "").strip()
        ),
        discord_guild_id=guild_id,
        bot_prefix=os.getenv("BOT_PREFIX", "!").strip() or "!",
        bot_name=bot_name,
        bot_creator_name=os.getenv("BOT_CREATOR_NAME", "Aagga").strip() or "Aagga",
        bot_role_description=os.getenv("BOT_ROLE_DESCRIPTION", "asisten AI yang bijaksana, tegas dalam menegakkan aturan, dan setia kepada pemiliknya").strip() or "asisten AI yang bijaksana, tegas dalam menegakkan aturan, dan setia kepada pemiliknya",
        bot_language=os.getenv("BOT_LANGUAGE", "id").strip() or "id",
        gemini_chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash",
        gemini_audio_model=os.getenv("GEMINI_AUDIO_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash",
        gemini_tts_model=os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts").strip() or "gemini-2.5-flash-preview-tts",
        gemini_tts_voice=os.getenv("GEMINI_TTS_VOICE", "Charon").strip() or "Charon",
        gemini_timeout_seconds=_get_int("GEMINI_TIMEOUT_SECONDS", 25),
        max_voice_seconds=_get_int("MAX_VOICE_SECONDS", 12),
        min_voice_seconds=_get_float("MIN_VOICE_SECONDS", 1.2),
        tts_max_chars=_get_int("TTS_MAX_CHARS", 1200),
        chat_cooldown_seconds=_get_int("CHAT_COOLDOWN_SECONDS", 8),
        voice_reply_cooldown_seconds=_get_int("VOICE_REPLY_COOLDOWN_SECONDS", 8),
        # TTS_ENGINE can be "gemini" or "piper". Piper runs locally and does not use Gemini quota.
        tts_engine=os.getenv("TTS_ENGINE", "piper").strip().lower() or "piper",
        piper_executable=os.getenv("PIPER_EXECUTABLE", "piper").strip() or "piper",
        piper_model_path=os.getenv("PIPER_MODEL_PATH", "models/id_ID-news_tts-medium.onnx").strip() or "models/id_ID-news_tts-medium.onnx",
        piper_timeout_seconds=_get_int("PIPER_TIMEOUT_SECONDS", 120),
        piper_length_scale=_get_float("PIPER_LENGTH_SCALE", 1.0),
        piper_noise_scale=_get_float("PIPER_NOISE_SCALE", 0.667),
        piper_noise_w=_get_float("PIPER_NOISE_W", 0.8),
        piper_sentence_silence=_get_float("PIPER_SENTENCE_SILENCE", 0.25),
        rag_scan_limit=_get_int("RAG_SCAN_LIMIT", 200),
        rag_top_k=_get_int("RAG_TOP_K", 6),
        rag_max_context_chars=_get_int("RAG_MAX_CONTEXT_CHARS", 4500),
        # Role/user access control.
        # Owner server, Administrator, and Manage Server always pass.
        # Add role names/IDs here for Moderator, Staff, etc.
        admin_role_names=_get_csv("ADMIN_ROLE_NAMES"),
        admin_role_ids=_get_csv_int("ADMIN_ROLE_IDS"),
        bot_owner_ids=_get_csv_int("BOT_OWNER_IDS"),
        # true = member biasa tetap bisa tanya AI text.
        # false = semua command AI hanya untuk role tinggi/admin.
        public_ai_commands=_get_bool("PUBLIC_AI_COMMANDS", True),
        # false = member biasa yang pakai /talk tidak otomatis dibacakan di voice.
        public_ai_voice_reply=_get_bool("PUBLIC_AI_VOICE_REPLY", False),
        # Auto moderation / guard system.
        moderation_enabled=_get_bool("MODERATION_ENABLED", True),
        moderation_ai_enabled=_get_bool("MODERATION_AI_ENABLED", True),
        moderation_delete_links=_get_bool("MODERATION_DELETE_LINKS", True),
        moderation_dm_warnings=_get_bool("MODERATION_DM_WARNINGS", True),
        mod_log_channel_id=mod_log_channel_id,
        mod_log_channel_names=mod_log_names,
        # Empty allowlist = normal members cannot post links. Example: discord.com,youtube.com
        mod_allowed_link_domains=_get_csv("MOD_ALLOWED_LINK_DOMAINS"),
        # Bypass roles can post links/promos without auto-delete. Admin/Manage Server always bypass too.
        mod_bypass_role_names=mod_bypass_names,
        mod_bypass_role_ids=_get_csv_int("MOD_BYPASS_ROLE_IDS"),
        mod_spam_window_seconds=_get_int("MOD_SPAM_WINDOW_SECONDS", 12),
        mod_spam_message_limit=_get_int("MOD_SPAM_MESSAGE_LIMIT", 6),
        mod_duplicate_message_limit=_get_int("MOD_DUPLICATE_MESSAGE_LIMIT", 3),
        mod_max_mentions=_get_int("MOD_MAX_MENTIONS", 6),
        moderation_image_ai_enabled=_get_bool("MODERATION_IMAGE_AI_ENABLED", True),
        moderation_ban_crypto_scams=_get_bool("MODERATION_BAN_CRYPTO_SCAMS", True),
        moderation_ban_dm_warnings=_get_bool("MODERATION_BAN_DM_WARNINGS", True),
        moderation_image_max_bytes=_get_int("MODERATION_IMAGE_MAX_BYTES", 4_000_000),
        # Mr. Beast / Celebrity scam detection (marak 2026)
        mrbeast_scam_detection=_get_bool("MRBEAST_SCAM_DETECTION", True),
        mrbeast_scam_ban=_get_bool("MRBEAST_SCAM_BAN", True),
        celebrity_scam_detection=_get_bool("CELEBRITY_SCAM_DETECTION", True),
    )
