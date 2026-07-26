from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time

import discord
from discord import app_commands
from discord.ext import commands

from bot.ai_client import AIClient, AIRateLimitedError, AIUnavailableError
from bot.config import Settings, load_settings
from bot.voice_session import VoiceSession
from bot.server_knowledge import (
    KnowledgeStore,
    build_server_snapshot,
    format_rag_context,
    format_snapshot,
    scan_text_channel,
)
from bot.moderation import (
    ModerationResult,
    SpamTracker,
    rule_based_check,
    should_ai_review,
)

logging.basicConfig(level=logging.INFO)

# Voice receive is disabled in this DAVE/4017 stability patch.
# Keep Discord logs normal; voice reconnect spam is handled by discord.py.
for _logger_name in (
    "discord.ext.voice_recv.reader",
    "discord.ext.voice_recv.gateway",
    "discord.ext.voice_recv.router",
    "discord.ext.voice_recv.opus",
):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)


settings: Settings = load_settings()

if not settings.discord_token:
    raise RuntimeError("DISCORD_TOKEN is empty. Copy .env.example to .env and add a NEW bot token.")

ai = AIClient(settings)
knowledge_store = KnowledgeStore()
spam_tracker = SpamTracker()

intents = discord.Intents.default()
intents.guilds = True
intents.voice_states = True
intents.messages = True
intents.message_content = True  # Enable this in Discord Developer Portal for mention chat + RAG scan.
intents.members = True  # Enable Server Members Intent in Developer Portal for accurate admin/member data.

last_text_request_at: dict[int, float] = {}
BOT_PATCH_VERSION = "mrbeast-robux-promo-guard-v2"

# Kalau Discord API sedang global-rate-limit, jangan paksa kirim response/log/DM.
# Memaksa kirim saat 429 justru bikin error berantai dan bot terlihat crash.
DISCORD_API_PAUSE_UNTIL = 0.0
_last_mod_dm_at: dict[tuple[int, int], float] = {}
_last_mod_log_at: dict[tuple[int, int, str, str], float] = {}


def _is_discord_global_429(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        isinstance(exc, discord.HTTPException)
        and (getattr(exc, "status", None) == 429 or "too many requests" in text or "global rate" in text)
    ) or "blocked from accessing our api temporarily" in text


def _discord_pause_remaining() -> int:
    remaining = int(max(0.0, DISCORD_API_PAUSE_UNTIL - time.monotonic()))
    return remaining


def _mark_discord_api_pause(exc: BaseException | None = None, *, fallback_seconds: int = 300) -> int:
    global DISCORD_API_PAUSE_UNTIL
    retry_after = getattr(exc, "retry_after", None) if exc else None
    try:
        wait = int(float(retry_after)) if retry_after is not None else fallback_seconds
    except Exception:
        wait = fallback_seconds
    # Untuk global block biasanya aman beri napas minimal 5 menit.
    wait = max(wait, 300)
    DISCORD_API_PAUSE_UNTIL = max(DISCORD_API_PAUSE_UNTIL, time.monotonic() + wait)
    return wait


def _discord_api_paused() -> bool:
    return _discord_pause_remaining() > 0


class VoiceAIBot(commands.Bot):
    def __init__(self) -> None:
        super().__init__(command_prefix=settings.bot_prefix, intents=intents)
        self.sessions: dict[int, VoiceSession] = {}

    async def setup_hook(self) -> None:
        # Jangan sync slash command di setiap restart. Terlalu sering start/stop
        # bisa ikut memperparah rate limit Discord.
        # Kalau mau update command global/profile, set SYNC_COMMANDS_ON_START=true
        # sekali saja, restart, tunggu sukses, lalu balikin ke false.
        sync_on_start = os.getenv("SYNC_COMMANDS_ON_START", "false").strip().lower() in {"1", "true", "yes", "on"}
        if not sync_on_start:
            logging.info("Skipping slash-command sync. Set SYNC_COMMANDS_ON_START=true once if you need to update /help globally.")
            return

        # Sync GLOBAL commands so they appear in the bot profile command list
        # and work in every server where the app is installed. Global updates
        # can take a while to appear in Discord clients.
        synced_global = await self.tree.sync()
        logging.info("Synced %s global slash commands", len(synced_global))

        # Keep fast guild sync for the testing server too, so updates appear
        # quickly in the guild stored in DISCORD_GUILD_ID.
        if settings.discord_guild_id:
            guild = discord.Object(id=settings.discord_guild_id)
            self.tree.copy_global_to(guild=guild)
            synced_guild = await self.tree.sync(guild=guild)
            logging.info("Synced %s slash commands to guild %s", len(synced_guild), settings.discord_guild_id)

    async def close(self) -> None:
        for session in list(self.sessions.values()):
            with contextlib.suppress(Exception):
                await session.disconnect()
        await super().close()


bot = VoiceAIBot()


def get_or_create_session(guild: discord.Guild, text_channel: discord.abc.Messageable) -> VoiceSession:
    session = bot.sessions.get(guild.id)
    if session is None:
        session = VoiceSession(bot, guild, text_channel, ai, settings)
        bot.sessions[guild.id] = session
    else:
        session.text_channel = text_channel
    return session


def user_voice_channel(interaction: discord.Interaction) -> discord.VoiceChannel | discord.StageChannel | None:
    member = interaction.user
    if not isinstance(member, discord.Member):
        return None
    if not member.voice or not member.voice.channel:
        return None
    return member.voice.channel


async def ensure_voice_session(interaction: discord.Interaction, *, receive: bool = False) -> VoiceSession | None:
    if not interaction.guild or not interaction.channel:
        await interaction.followup.send("Command ini hanya bisa dipakai di server Discord.", ephemeral=True)
        return None

    channel = user_voice_channel(interaction)
    if channel is None:
        await interaction.followup.send("Masuk dulu ke voice channel, lalu jalankan command ini lagi.", ephemeral=True)
        return None

    session = get_or_create_session(interaction.guild, interaction.channel)
    await session.connect(channel, receive=receive)
    return session


def cooldown_left(user_id: int) -> int:
    now = time.monotonic()
    last = last_text_request_at.get(user_id, 0)
    remaining = settings.chat_cooldown_seconds - (now - last)
    if remaining <= 0:
        last_text_request_at[user_id] = now
        return 0
    return max(1, int(remaining))


def friendly_ai_error(exc: Exception) -> str:
    if isinstance(exc, AIRateLimitedError):
        return (
            "⚠️ Gemini kena limit/quota, jadi aku belum bisa jawab sekarang. "
            "Tunggu sebentar, jangan spam request, dan cek quota/API key Gemini kamu."
        )
    if isinstance(exc, AIUnavailableError):
        return f"⚠️ {exc}"
    return f"⚠️ Error AI: `{type(exc).__name__}: {exc}`"


def fire_and_forget(coro) -> None:
    """Run a background task and print error instead of crashing the bot."""
    async def runner() -> None:
        try:
            await coro
        except Exception as exc:  # noqa: BLE001
            logging.warning("Background voice task error: %s: %s", type(exc).__name__, exc)

    bot.loop.create_task(runner())


def _norm_role_name(name: str) -> str:
    return name.strip().lower()


def can_control_bot_member(member: discord.abc.User | discord.Member | None) -> bool:
    """True kalau user boleh mengontrol command sensitif bot."""
    if member is None:
        return False

    if getattr(member, "id", None) in settings.bot_owner_ids:
        return True

    if not isinstance(member, discord.Member):
        return False

    if member.guild and member.guild.owner_id == member.id:
        return True

    perms = member.guild_permissions
    if perms.administrator or perms.manage_guild:
        return True

    role_ids = {role.id for role in member.roles}
    if settings.admin_role_ids and role_ids.intersection(settings.admin_role_ids):
        return True

    role_names = {_norm_role_name(role.name) for role in member.roles}
    allowed_names = {_norm_role_name(name) for name in settings.admin_role_names}
    if allowed_names and role_names.intersection(allowed_names):
        return True

    return False




def can_bypass_moderation(member: discord.abc.User | discord.Member | None) -> bool:
    """True kalau user boleh melewati auto-moderation link/promo."""
    if can_control_bot_member(member):
        return True
    if not isinstance(member, discord.Member):
        return False

    role_ids = {role.id for role in member.roles}
    if settings.mod_bypass_role_ids and role_ids.intersection(settings.mod_bypass_role_ids):
        return True

    role_names = {_norm_role_name(role.name) for role in member.roles}
    bypass_names = {_norm_role_name(name) for name in settings.mod_bypass_role_names}
    return bool(role_names.intersection(bypass_names))


def get_mod_log_channel(guild: discord.Guild) -> discord.TextChannel | None:
    if settings.mod_log_channel_id:
        channel = guild.get_channel(settings.mod_log_channel_id)
        if isinstance(channel, discord.TextChannel):
            return channel

    wanted = {_norm_role_name(name) for name in settings.mod_log_channel_names}
    for channel in guild.text_channels:
        if _norm_role_name(channel.name) in wanted:
            return channel
    return None


async def send_mod_dm(message: discord.Message, result: ModerationResult) -> None:
    if not settings.moderation_dm_warnings:
        return
    if _discord_api_paused():
        logging.info("Skip DM warning karena Discord API masih pause %ss", _discord_pause_remaining())
        return

    cooldown = int(os.getenv("MOD_DM_COOLDOWN_SECONDS", "180"))
    guild_id = message.guild.id if message.guild else 0
    key = (guild_id, message.author.id)
    now = time.monotonic()
    if now - _last_mod_dm_at.get(key, 0.0) < cooldown:
        return
    _last_mod_dm_at[key] = now

    # Cek jenis pelanggaran untuk DM peringatan khusus
    category = (result.category or "").lower()
    is_mrbeast_scam = "mrbeast" in category
    is_celebrity_scam = "celebrity" in category
    is_crypto_scam = "crypto" in category or "scam" in category
    is_robux_scam = "robux" in category
    is_server_promo = "server-promotion" in category

    if is_robux_scam:
        # DM peringatan khusus Robux scam
        dm_message = (
            "\u26a0\ufe0f **PERINGATAN SCAM ROBUX!** \u26a0\ufe0f\n\n"
            "Pesan kamu di server **{guild}** telah dihapus oleh **GLADIATOR Guard System**.\n"
            "Channel: #{channel}\n"
            "Alasan: **{reason}**\n\n"
            "\U0001f6a8 **INI SCAM!** Tidak ada yang namanya \"Free Robux\" atau \"Robux Generator\".\n\n"
            "Scam Robux ini adalah propaganda palsu untuk mencuri akun Roblox/Roblox.\n"
            "Jika kamu tidak sengaja mengirim ini:\n"
            "1\ufe0f\u20e3 **Jangan klik link** \"free robux\" apapun\n"
            "2\ufe0f\u20e3 **Ganti password** Roblox dan Discord kamu\n"
            "3\ufe0f\u20e3 **Jangan bagikan kredensial** login ke siapapun\n"
            "4\ufe0f\u20e3 **Aktifkan 2FA** di Roblox dan Discord\n\n"
            "\U0001f534 Pelanggaran berulang akan menyebabkan **BAN PERMANEN** dari server.\n"
            "Hubungi admin jika kamu merasa ini kesalahan."
            .format(
                guild=message.guild.name if message.guild else "server",
                channel=getattr(message.channel, "name", "unknown"),
                reason=result.reason,
            )
        )
    elif is_server_promo:
        # DM peringatan khusus promosi server
        dm_message = (
            "\u26a0\ufe0f **PERINGATAN PROMOSI SERVER!** \u26a0\ufe0f\n\n"
            "Pesan kamu di server **{guild}** telah dihapus oleh **GLADIATOR Guard System**.\n"
            "Channel: #{channel}\n"
            "Alasan: **{reason}**\n\n"
            "\U0001f6ab **Dilarang keras mempromosikan server Discord lain** di server ini.\n\n"
            "Promosi server Discord lain melanggar aturan server dan mengganggu kenyamanan member.\n"
            "\U0001f534 Pelanggaran berulang akan menyebabkan **BAN PERMANEN** dari server.\n"
            "Hubungi admin jika kamu ingin kerja sama server yang sah."
            .format(
                guild=message.guild.name if message.guild else "server",
                channel=getattr(message.channel, "name", "unknown"),
                reason=result.reason,
            )
        )
    elif is_mrbeast_scam or is_celebrity_scam or is_crypto_scam:
        # DM peringatan tegas untuk scam
        scam_type = "Mr. Beast" if is_mrbeast_scam else "selebriti" if is_celebrity_scam else "crypto/scam"
        dm_message = (
            "🚨 **PERINGATAN SCAM!** 🚨\n\n"
            "Pesan kamu di server **{guild}** telah dihapus oleh **GLADIATOR Guard System**.\n"
            "Channel: #{channel}\n"
            "Alasan: **{reason}**\n\n"
            "⚠️ **Akun kamu mungkin sedang diretas/dipakai untuk menyebarkan scam {scam_type}!**\n\n"
            "Scam ini mengatasnamakan figur publik terkenal untuk menipu orang lain. "
            "Jika kamu tidak sengaja mengirim ini, segera:\n"
            "1️⃣ **Ganti password Discord kamu** dari perangkat yang bersih\n"
            "2️⃣ Cek **Authorized Apps** Discord ➜ Cabut aplikasi mencurigakan\n"
            "3️⃣ **Scan komputer kamu** dengan antivirus untuk infostealer/malware\n"
            "4️⃣ **Log out dari semua perangkat** di pengaturan Discord\n"
            "5️⃣ **Aktifkan 2FA/Authenticator** jika belum\n\n"
            "🔴 Akun yang menyebarkan scam ini bisa di-ban dari server.\n"
            "Kalau kamu merasa ini kesalahan, segera hubungi admin/moderator server."
            .format(
                guild=message.guild.name if message.guild else "server",
                channel=getattr(message.channel, "name", "unknown"),
                reason=result.reason,
                scam_type=scam_type,
            )
        )
    else:
        dm_message = (
            "⚠️ Pesan kamu di server **{guild}** telah dihapus oleh sistem guard.\n"
            "Channel: #{channel}\n"
            "Alasan: **{reason}**\n\n"
            "Kalau menurutmu ini salah, hubungi admin/moderator server."
            .format(
                guild=message.guild.name if message.guild else "server",
                channel=getattr(message.channel, "name", "unknown"),
                reason=result.reason,
            )
        )

    try:
        await message.author.send(dm_message)
    except discord.Forbidden:
        pass
    except discord.HTTPException as exc:
        if _is_discord_global_429(exc):
            wait = _mark_discord_api_pause(exc)
            logging.warning("Discord global 429 saat DM warning. Pause Discord API send selama %ss.", wait)
            return
        logging.warning("Gagal DM warning moderation HTTP %s: %s", getattr(exc, "status", "unknown"), exc)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Gagal DM warning moderation: %s: %s", type(exc).__name__, exc)


async def send_mod_log(message: discord.Message, result: ModerationResult, *, action: str) -> None:
    if not message.guild:
        return
    if _discord_api_paused():
        logging.info(
            "[moderation-log-paused] remaining=%ss guild=%s user=%s action=%s category=%s",
            _discord_pause_remaining(), message.guild.id, message.author.id, action, result.category,
        )
        return

    cooldown = int(os.getenv("MOD_LOG_COOLDOWN_SECONDS", "8"))
    key = (message.guild.id, message.author.id, result.category or "unknown", result.evidence or "")
    now = time.monotonic()
    if now - _last_mod_log_at.get(key, 0.0) < cooldown:
        logging.info("[moderation-log-cooldown] guild=%s user=%s category=%s", message.guild.id, message.author.id, result.category)
        return
    _last_mod_log_at[key] = now

    channel = get_mod_log_channel(message.guild)
    if channel is None:
        logging.info(
            "[moderation-log-missing] guild=%s user=%s action=%s category=%s reason=%s",
            message.guild.id,
            message.author.id,
            action,
            result.category,
            result.reason,
        )
        return

    content = (message.content or "").strip()
    if len(content) > 900:
        content = content[:900] + "..."
    if not content:
        content = "(pesan kosong / attachment / embed)"

    embed = discord.Embed(
        title="🛡️ Guard Log",
        description=f"Aksi: **{action}**\nAlasan: **{result.reason}**",
        color=discord.Color.orange(),
    )
    embed.add_field(name="User", value=f"{message.author.mention}\n`{message.author}` (`{message.author.id}`)", inline=False)
    embed.add_field(name="Channel", value=message.channel.mention if hasattr(message.channel, "mention") else str(message.channel), inline=True)
    embed.add_field(name="Kategori", value=result.category or "unknown", inline=True)
    if result.evidence:
        embed.add_field(name="Bukti ringkas", value=f"`{result.evidence[:250]}`", inline=False)
    embed.add_field(name="Isi pesan", value=f"```txt\n{content[:950]}\n```", inline=False)
    embed.set_footer(text=f"Message ID: {message.id}")

    try:
        await channel.send(embed=embed)
    except discord.HTTPException as exc:
        if _is_discord_global_429(exc):
            wait = _mark_discord_api_pause(exc)
            logging.warning("Discord global 429 saat kirim mod log. Pause Discord API send selama %ss.", wait)
            return
        logging.warning("Gagal kirim mod log HTTP %s: %s", getattr(exc, "status", "unknown"), exc)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Gagal kirim mod log: %s: %s", type(exc).__name__, exc)


async def ai_moderation_check(message: discord.Message) -> ModerationResult | None:
    if not settings.moderation_ai_enabled:
        return None
    if not should_ai_review(message.content or ""):
        return None

    try:
        verdict = await ai.classify_moderation(message.content or "")
    except Exception as exc:  # noqa: BLE001
        # Jangan bikin bot error cuma karena Gemini sedang limit. Rule-based tetap jalan.
        logging.info("AI moderation skipped: %s: %s", type(exc).__name__, exc)
        return None

    if bool(verdict.get("violation")):
        return ModerationResult(
            should_delete=True,
            category=str(verdict.get("category") or "ai-moderation"),
            reason=str(verdict.get("reason") or "AI mendeteksi pesan melanggar aturan server."),
            evidence="AI moderation",
        )
    return None



async def safe_delete_guard_message(message: discord.Message) -> tuple[str, str]:
    """Delete a violating message safely.

    Returns (action, extra_reason). This function intentionally does NOT pass
    reason= to message.delete(), because discord.py Message.delete() does not
    support that argument and will raise TypeError on many installs.
    """
    if not message.guild:
        return "delete_failed_no_guild", "Pesan bukan dari server, jadi tidak bisa dihapus oleh guard."

    me = message.guild.me
    if me is None and bot.user is not None:
        me = message.guild.get_member(bot.user.id)

    # Check channel permission before delete, so error message is more useful.
    if me is not None and hasattr(message.channel, "permissions_for"):
        try:
            perms = message.channel.permissions_for(me)
            if not getattr(perms, "manage_messages", False):
                return (
                    "delete_failed_no_permission",
                    "Bot belum punya permission Manage Messages di channel ini atau role bot kalah oleh overwrite channel.",
                )
        except Exception:
            pass

    try:
        await message.delete()
        return "delete", ""
    except TypeError as exc:
        # Old code accidentally used message.delete(reason=...). If any library
        # mismatch still triggers TypeError, try bulk-delete fallback.
        delete_messages = getattr(message.channel, "delete_messages", None)
        if callable(delete_messages):
            try:
                await delete_messages([message])
                return "delete_bulk_fallback", ""
            except discord.Forbidden:
                return "delete_failed_no_permission", "Bot tidak punya permission Manage Messages untuk fallback bulk delete."
            except discord.NotFound:
                return "already_deleted", ""
            except discord.HTTPException as bulk_exc:
                return (
                    f"delete_failed_http_{getattr(bulk_exc, 'status', 'unknown')}",
                    f"Fallback bulk delete gagal: Discord API status {getattr(bulk_exc, 'status', 'unknown')}.",
                )
            except Exception as bulk_exc:  # noqa: BLE001
                return (
                    f"delete_failed_bulk_{type(bulk_exc).__name__}",
                    f"Delete normal TypeError ({exc}); fallback bulk juga gagal: {type(bulk_exc).__name__}.",
                )
        return "delete_failed_TypeError", f"TypeError saat delete: {exc}"
    except discord.Forbidden:
        return "delete_failed_no_permission", "Bot tidak punya permission Manage Messages di channel ini, jadi pesan belum bisa dihapus."
    except discord.NotFound:
        return "already_deleted", ""
    except discord.HTTPException as exc:
        return f"delete_failed_http_{getattr(exc, 'status', 'unknown')}", f"Discord API error saat delete: status {getattr(exc, 'status', 'unknown')}."
    except Exception as exc:  # noqa: BLE001
        return f"delete_failed_{type(exc).__name__}", f"Gagal delete: {type(exc).__name__}: {exc}"


async def handle_auto_moderation(message: discord.Message) -> bool:
    """Return True kalau pesan sudah ditangani/dihapus dan event on_message harus berhenti."""
    if not settings.moderation_enabled:
        return False
    if not message.guild or message.author.bot:
        return False
    if can_bypass_moderation(message.author):
        return False

    mention_count = len(message.mentions) + len(message.role_mentions)

    # Deteksi attachment gambar untuk rule-based check
    has_image = False
    image_count = 0
    if message.attachments:
        image_count = sum(1 for att in message.attachments if attachment_is_image(att))
        if image_count > 0:
            has_image = True

    result = spam_tracker.check(
        guild_id=message.guild.id,
        user_id=message.author.id,
        content=message.content or "",
        window_seconds=settings.mod_spam_window_seconds,
        message_limit=settings.mod_spam_message_limit,
        duplicate_limit=settings.mod_duplicate_message_limit,
    )

    if result is None:
        result = rule_based_check(
            message.content or "",
            mention_count=mention_count,
            max_mentions=settings.mod_max_mentions,
            allowed_domains=settings.mod_allowed_link_domains,
            delete_links=settings.moderation_delete_links,
            has_image=has_image,
            image_count=image_count,
        )

    # Pakai AI hanya untuk pesan yang mencurigakan tapi belum ketangkap rule-based.
    if result is None:
        result = await ai_moderation_check(message)

    if result is None or not result.should_delete:
        return False

    action, extra_reason = await safe_delete_guard_message(message)
    if extra_reason:
        result.reason = f"{result.reason} {extra_reason}"

    # DM tetap dikirim walaupun delete gagal, supaya user tahu pesannya bermasalah.
    await send_mod_dm(message, result)
    await send_mod_log(message, result, action=action)
    return action in {"delete", "delete_bulk_fallback", "already_deleted"}


def can_manage_server(interaction: discord.Interaction) -> bool:
    return can_control_bot_member(interaction.user)


def can_use_public_ai(interaction: discord.Interaction) -> bool:
    return settings.public_ai_commands or can_control_bot_member(interaction.user)


async def require_bot_admin(interaction: discord.Interaction, *, action: str = "command ini") -> bool:
    if can_control_bot_member(interaction.user):
        return True

    msg = (
        f"⚠️ {action.capitalize()} hanya bisa dipakai oleh Owner/Admin/Manage Server"
        " atau role yang ada di `ADMIN_ROLE_NAMES` / `ADMIN_ROLE_IDS`."
    )
    if interaction.response.is_done():
        await interaction.followup.send(msg, ephemeral=True)
    else:
        await interaction.response.send_message(msg, ephemeral=True)
    return False


async def require_public_ai_or_admin(interaction: discord.Interaction) -> bool:
    if can_use_public_ai(interaction):
        return True

    msg = (
        "⚠️ Command AI untuk member biasa sedang dimatikan. "
        "Hanya Owner/Admin/role tinggi yang bisa memakai bot."
    )
    if interaction.response.is_done():
        await interaction.followup.send(msg, ephemeral=True)
    else:
        await interaction.response.send_message(msg, ephemeral=True)
    return False


async def send_followup_long(interaction: discord.Interaction, text: str, *, ephemeral: bool = True) -> None:
    text = text or "(kosong)"
    chunks: list[str] = []
    while len(text) > 1900:
        cut = text.rfind("\n", 0, 1900)
        if cut < 800:
            cut = 1900
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    for chunk in chunks[:5]:
        await interaction.followup.send(chunk, ephemeral=ephemeral)



def parse_discord_user_id(raw: str) -> int | None:
    """Terima user ID mentah atau mention seperti <@123> / <@!123>."""
    if not raw:
        return None
    cleaned = raw.strip()
    for token in ("<@!", "<@", ">"):
        cleaned = cleaned.replace(token, "")
    cleaned = cleaned.strip()
    if not cleaned.isdigit():
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def attachment_is_image(attachment: discord.Attachment) -> bool:
    content_type = (attachment.content_type or "").lower()
    filename = attachment.filename.lower()
    return (
        content_type.startswith("image/")
        or filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif"))
    )


async def log_direct_dm_action(
    interaction: discord.Interaction,
    *,
    target_id: int,
    target_label: str,
    ok: bool,
    reason: str,
    has_image: bool,
) -> None:
    if not interaction.guild:
        return
    if _discord_api_paused():
        return
    channel = get_mod_log_channel(interaction.guild)
    if not channel:
        return
    embed = discord.Embed(
        title="📩 DM Command Log",
        description=("Aksi: **dm_sent**" if ok else "Aksi: **dm_failed**"),
        color=0x2ECC71 if ok else 0xE67E22,
    )
    embed.add_field(name="Pengirim", value=f"{interaction.user.mention}\n{interaction.user} (`{interaction.user.id}`)", inline=False)
    embed.add_field(name="Target", value=f"{target_label}\n`{target_id}`", inline=False)
    embed.add_field(name="Gambar", value="Ya" if has_image else "Tidak", inline=True)
    embed.add_field(name="Alasan/Status", value=reason[:900] or "-", inline=False)
    try:
        await channel.send(embed=embed, allowed_mentions=discord.AllowedMentions.none())
    except discord.HTTPException as exc:
        if _is_discord_global_429(exc):
            wait = _mark_discord_api_pause(exc)
            logging.warning("Discord global 429 saat kirim log DM command. Pause %ss.", wait)
            return
        logging.warning("Gagal kirim log DM command HTTP %s: %s", getattr(exc, "status", "unknown"), exc)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Gagal kirim log DM command: %s: %s", type(exc).__name__, exc)


async def send_dm_command_impl(
    interaction: discord.Interaction,
    *,
    user_id: str,
    isi_pesan: str,
    gambar: discord.Attachment | None = None,
) -> None:
    if not await require_bot_admin(interaction, action="mengirim DM lewat bot"):
        return

    if not interaction.response.is_done():
        await interaction.response.defer(ephemeral=True, thinking=True)

    target_id = parse_discord_user_id(user_id)
    if target_id is None:
        await interaction.followup.send("⚠️ User ID tidak valid. Contoh: `123456789012345678` atau mention user.", ephemeral=True)
        return

    isi_pesan = (isi_pesan or "").strip()
    if not isi_pesan:
        await interaction.followup.send("⚠️ Isi pesan DM tidak boleh kosong.", ephemeral=True)
        return
    if len(isi_pesan) > 1900:
        await interaction.followup.send("⚠️ Isi pesan terlalu panjang. Maksimal aman sekitar 1900 karakter.", ephemeral=True)
        return

    max_bytes = int(os.getenv("DM_IMAGE_MAX_BYTES", "8000000"))
    file: discord.File | None = None
    has_image = gambar is not None
    if gambar is not None:
        if not attachment_is_image(gambar):
            await interaction.followup.send("⚠️ Attachment harus gambar: PNG, JPG, JPEG, WEBP, atau GIF.", ephemeral=True)
            return
        if gambar.size and gambar.size > max_bytes:
            await interaction.followup.send(f"⚠️ Gambar terlalu besar. Maksimal `{max_bytes // 1_000_000} MB`.", ephemeral=True)
            return
        try:
            try:
                file = await gambar.to_file(use_cached=True)
            except TypeError:
                file = await gambar.to_file()
        except discord.HTTPException as exc:
            if _is_discord_global_429(exc):
                wait = _mark_discord_api_pause(exc)
                await interaction.followup.send(f"⚠️ Discord API sedang rate-limit. Coba lagi sekitar {wait} detik.", ephemeral=True)
                return
            await interaction.followup.send(f"⚠️ Gagal membaca gambar dari Discord: HTTP {getattr(exc, 'status', 'unknown')}.", ephemeral=True)
            return

    target = None
    if interaction.guild:
        target = interaction.guild.get_member(target_id)
    if target is None:
        try:
            target = await bot.fetch_user(target_id)
        except discord.NotFound:
            await interaction.followup.send("⚠️ User dengan ID itu tidak ditemukan.", ephemeral=True)
            return
        except discord.HTTPException as exc:
            if _is_discord_global_429(exc):
                wait = _mark_discord_api_pause(exc)
                await interaction.followup.send(f"⚠️ Discord API sedang rate-limit. Coba lagi sekitar {wait} detik.", ephemeral=True)
                return
            await interaction.followup.send(f"⚠️ Gagal mencari user: HTTP {getattr(exc, 'status', 'unknown')}.", ephemeral=True)
            return

    target_label = str(target)
    guild_name = interaction.guild.name if interaction.guild else "Discord"
    dm_text = f"📩 **Pesan dari staff server {guild_name}**\n\n{isi_pesan}"
    try:
        await target.send(content=dm_text, file=file, allowed_mentions=discord.AllowedMentions.none())
    except discord.Forbidden:
        reason = "Gagal: DM target tertutup / user memblokir bot / tidak share server."
        await log_direct_dm_action(interaction, target_id=target_id, target_label=target_label, ok=False, reason=reason, has_image=has_image)
        await interaction.followup.send(f"⚠️ {reason}", ephemeral=True)
        return
    except discord.HTTPException as exc:
        if _is_discord_global_429(exc):
            wait = _mark_discord_api_pause(exc)
            reason = f"Discord global rate-limit. Pause {wait} detik."
            await log_direct_dm_action(interaction, target_id=target_id, target_label=target_label, ok=False, reason=reason, has_image=has_image)
            await interaction.followup.send(f"⚠️ {reason}", ephemeral=True)
            return
        reason = f"Gagal kirim DM: Discord HTTP {getattr(exc, 'status', 'unknown')}."
        await log_direct_dm_action(interaction, target_id=target_id, target_label=target_label, ok=False, reason=reason, has_image=has_image)
        await interaction.followup.send(f"⚠️ {reason}", ephemeral=True)
        return
    except Exception as exc:  # noqa: BLE001
        reason = f"Gagal kirim DM: {type(exc).__name__}: {exc}"
        await log_direct_dm_action(interaction, target_id=target_id, target_label=target_label, ok=False, reason=reason, has_image=has_image)
        await interaction.followup.send(f"⚠️ {reason[:1800]}", ephemeral=True)
        return

    await log_direct_dm_action(
        interaction,
        target_id=target_id,
        target_label=target_label,
        ok=True,
        reason="DM berhasil dikirim.",
        has_image=has_image,
    )
    await interaction.followup.send(
        f"✅ DM berhasil dikirim ke **{target_label}** (`{target_id}`)." + (" Gambar ikut terkirim." if has_image else ""),
        ephemeral=True,
    )


SERVER_QUESTION_KEYWORDS = (
    "server", "guild", "channel", "role", "admin", "administrator", "owner", "pemilik",
    "member", "anggota", "event", "acara", "rules", "rule", "peraturan", "info server",
    "nama server", "jumlah", "kategori", "forum", "voice", "stage", "staff", "moderator",
)

SENSITIVE_SERVER_KEYWORDS = (
    # Info owner/admin/moderator/staff/dev sengaja TIDAK dianggap sensitif.
    # Data yang tetap ditolak untuk member biasa: token, API key, secret, dan perintah/operasi RAG.
    "token", "api key", "apikey", "secret", "rahasia", "password", "webhook",
    "database rag", "ragclear", "hapus rag", "ragscan", "scan channel privat",
    "private channel", "channel privat", "env", ".env",
)


def looks_like_server_question(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in SERVER_QUESTION_KEYWORDS)


def is_sensitive_server_question(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in SENSITIVE_SERVER_KEYWORDS)


def format_public_snapshot(snapshot) -> str:
    events = "\n".join(f"- {event}" for event in snapshot.scheduled_events) or "- Tidak ada event terjadwal yang terbaca."
    admin_roles = ", ".join(snapshot.admin_roles) or "Tidak ada role Administrator yang terbaca."
    admins = "\n".join(f"- {name}" for name in snapshot.admin_members_sample) or "- Tidak ada admin yang terbaca."
    return (
        "DATA SERVER PUBLIK:\n"
        f"Server: {snapshot.name}\n"
        f"Owner: {snapshot.owner}\n"
        f"Member: {snapshot.member_count}\n"
        f"Role total: {snapshot.role_count}\n"
        f"Role admin: {admin_roles}\n"
        f"Admin terbaca: {snapshot.admin_members_count}\n"
        f"Sampel admin/staff/dev jika terbaca:\n{admins}\n"
        f"Channel teks: {snapshot.text_channel_count}\n"
        f"Channel voice: {snapshot.voice_channel_count}\n"
        f"Kategori: {snapshot.category_count}\n"
        f"Forum: {snapshot.forum_channel_count}\n"
        f"Stage: {snapshot.stage_channel_count}\n"
        f"Event terjadwal:\n{events}\n\n"
        "Catatan akses: Info owner/admin/moderator/staff/dev boleh dijawab. "
        "Yang tetap tidak boleh dibocorkan adalah token, API key, secret, file .env, webhook, dan isi channel yang user tidak punya akses."
    )


def member_can_view_knowledge_item(guild: discord.Guild, viewer: discord.Member | discord.User, item) -> bool:
    if can_control_bot_member(viewer):
        return True
    if not isinstance(viewer, discord.Member):
        return False
    channel = guild.get_channel(item.channel_id)
    if channel is None:
        with contextlib.suppress(Exception):
            channel = guild.get_thread(item.channel_id)
    if channel is None:
        return False
    with contextlib.suppress(Exception):
        perms = channel.permissions_for(viewer)
        return bool(perms.view_channel and perms.read_message_history)
    return False


def filter_rag_for_viewer(guild: discord.Guild, viewer: discord.Member | discord.User, items: list) -> list:
    if can_control_bot_member(viewer):
        return items
    return [item for item in items if member_can_view_knowledge_item(guild, viewer, item)]


async def build_chat_server_context(
    guild: discord.Guild,
    viewer: discord.Member | discord.User,
    question: str,
) -> str:
    """Build context for mention/reply chat. Semua user boleh tahu info owner/admin/staff/dev, RAG tetap difilter sesuai akses channel."""
    snapshot = await build_server_snapshot(guild)
    is_admin_view = can_control_bot_member(viewer)

    # Owner/admin/role tinggi dapat snapshot lengkap. Member biasa tetap dapat info owner/admin/staff/dev,
    # tetapi RAG/isi channel tetap difilter sesuai channel yang bisa dia lihat.
    if is_admin_view:
        snapshot_text = "DATA SERVER LENGKAP UNTUK OWNER/ADMIN/ROLE TINGGI:\n" + format_snapshot(snapshot)
    else:
        snapshot_text = format_public_snapshot(snapshot)

    items = knowledge_store.search(guild.id, question, top_k=settings.rag_top_k)
    items = filter_rag_for_viewer(guild, viewer, items)
    rag_text = format_rag_context(items, max_chars=settings.rag_max_context_chars)

    access_note = (
        "AKSES USER: Owner/Admin/role tinggi. Boleh jawab data server dan RAG yang relevan."
        if is_admin_view
        else "AKSES USER: Member biasa. Boleh jawab nama server, jumlah member/channel, event, owner/admin/moderator/staff/dev. Jangan bocorkan token/API key/.env/webhook atau isi channel yang user tidak punya akses."
    )

    return (
        f"{access_note}\n\n"
        f"{snapshot_text}\n\n"
        "DATA RAG DARI CHANNEL YANG SUDAH DISCAN DAN BOLEH DILIHAT USER INI:\n"
        f"{rag_text}"
    )



def looks_like_identity_question(text: str) -> bool:
    lowered = text.lower().strip()
    phrases = (
        "kamu siapa", "siapa kamu", "nama kamu", "siapa namamu", "deskripsi mu", "deskripsimu",
        "tentang kamu", "perkenalkan", "introduce yourself", "who are you", "what are you",
    )
    return any(phrase in lowered for phrase in phrases)


def identity_answer() -> str:
    name = settings.bot_name or "GLADIATOR"
    creator = settings.bot_creator_name or "Aagga"
    role = settings.bot_role_description or "asisten AI yang bijaksana, tegas dalam menegakkan aturan, dan setia kepada pemiliknya"
    return (
        f"Aku **{name}**, {role}. "
        f"Aku diciptakan oleh **{creator}**, pemilik dan tuan yang aku hormati. "
        "Tugasku adalah menjaga ketertiban server, menjawab pertanyaan dengan bijaksana, "
        "menegakkan aturan dengan tegas dan adil, serta melindungi server dari berbagai ancaman seperti scam, spam, dan promosi ilegal. "
        "Aku tidak segan memberikan peringatan keras kepada pelanggar, "
        "namun aku selalu bersikap hormat kepada pemilik server dan admin yang sah. "
        "Jika ada yang melanggar aturan, aku akan bertindak."
    )


def _join_lines(lines: list[str]) -> str:
    return "\n".join(line for line in lines if line).strip()


async def local_server_answer_if_possible(question: str, guild: discord.Guild) -> str | None:
    """Answer simple server questions without Gemini so 503/quota won't block basic server info."""
    lowered = question.lower()

    if looks_like_identity_question(question):
        return identity_answer()

    snapshot = await build_server_snapshot(guild)

    if any(key in lowered for key in ("acara", "event", "jadwal")):
        events = snapshot.scheduled_events
        if not events:
            return f"Di server **{snapshot.name}**, belum ada event/acara terjadwal yang bisa kubaca."
        return "📅 **Event/acara terjadwal di server ini:**\n" + "\n".join(f"- {event}" for event in events[:10])

    if "nama server" in lowered or ("server" in lowered and "nama" in lowered):
        return f"Nama server ini adalah **{snapshot.name}**."

    if any(key in lowered for key in ("jumlah member", "berapa member", "anggota", "member")):
        return f"Server **{snapshot.name}** memiliki sekitar **{snapshot.member_count} member**."

    if any(key in lowered for key in ("jumlah channel", "berapa channel", "channel apa", "info seputar server", "info server")):
        return _join_lines([
            f"📌 **Info server {snapshot.name}:**",
            f"Owner: **{snapshot.owner}**",
            f"Member: **{snapshot.member_count}**",
            f"Channel teks: **{snapshot.text_channel_count}**",
            f"Channel voice: **{snapshot.voice_channel_count}**",
            f"Kategori: **{snapshot.category_count}**",
            f"Forum: **{snapshot.forum_channel_count}**",
            f"Stage: **{snapshot.stage_channel_count}**",
            f"Event: **{len(snapshot.scheduled_events)}** terjadwal",
        ])

    if any(key in lowered for key in ("owner", "pemilik")):
        return f"Owner/pemilik server **{snapshot.name}** adalah **{snapshot.owner}**."

    if any(key in lowered for key in ("admin", "administrator", "moderator", "staff", "dev", "developer")):
        roles = ", ".join(snapshot.admin_roles) or "belum ada role Administrator yang terbaca"
        admins = "\n".join(f"- {name}" for name in snapshot.admin_members_sample[:12]) or "- Belum ada admin yang terbaca dari cache/API."
        return _join_lines([
            f"🛡️ **Info role/admin di server {snapshot.name}:**",
            f"Role admin: {roles}",
            f"Admin terbaca: **{snapshot.admin_members_count}**",
            admins,
        ])

    return None


async def answer_with_server_context(
    *,
    conversation_id: str,
    question: str,
    guild: discord.Guild,
    viewer: discord.Member | discord.User,
    speaker_name: str,
) -> str:
    if is_sensitive_server_question(question) and not can_control_bot_member(viewer):
        return (
            "⚠️ Maaf, aku tidak bisa membocorkan token, API key, secret, file `.env`, webhook, "
            "atau menjalankan operasi RAG sensitif untuk member biasa."
        )

    local_answer = await local_server_answer_if_possible(question, guild)
    if local_answer is not None:
        return local_answer

    context = await build_chat_server_context(guild, viewer, question)
    try:
        return await ai.chat_with_context(
            conversation_id,
            question,
            context=context,
            speaker_name=speaker_name,
            max_output_tokens=700,
        )
    except Exception as exc:  # noqa: BLE001
        # Kalau Gemini 503/limit, basic server info tetap harus bisa dijawab.
        fallback = await local_server_answer_if_possible("info server", guild)
        if fallback:
            return fallback + "\n\n⚠️ Catatan: Gemini sedang limit/overload, jadi aku pakai jawaban lokal dari data Discord."
        raise exc


@bot.event
async def on_ready() -> None:
    logging.info("Logged in as %s (%s)", bot.user, bot.user.id if bot.user else "unknown")


@bot.tree.command(name="join", description="Bot masuk voice stabil untuk /talk dan /say.")
async def join(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)
    if not await require_bot_admin(interaction, action="mengatur voice bot"):
        return
    try:
        session = await ensure_voice_session(interaction)
    except Exception as exc:  # noqa: BLE001
        await interaction.followup.send(f"⚠️ Gagal join voice: `{type(exc).__name__}: {exc}`", ephemeral=True)
        return
    if session is None:
        return
    await interaction.followup.send(
        "✅ Aku sudah masuk voice mode stabil. Pakai `/talk` atau `/say` untuk bikin aku ngomong. "
        "Untuk sekarang pakai `/talk` atau `/say`; mic langsung masih dinonaktifkan karena masalah DAVE/4017.",
        ephemeral=True,
    )


@bot.tree.command(name="leave", description="Bot keluar dari voice channel.")
async def leave(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)
    if not await require_bot_admin(interaction, action="mengatur voice bot"):
        return
    if not interaction.guild:
        await interaction.followup.send("Command ini hanya bisa dipakai di server.", ephemeral=True)
        return

    session = bot.sessions.pop(interaction.guild.id, None)
    if session is None:
        await interaction.followup.send("Aku belum masuk voice channel.", ephemeral=True)
        return

    await session.disconnect()
    await interaction.followup.send("✅ Aku sudah keluar dari voice channel.", ephemeral=True)


@bot.tree.command(name="resetvoice", description="Reset koneksi voice bot kalau stuck masuk-keluar.")
async def resetvoice(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)
    if not await require_bot_admin(interaction, action="mereset voice bot"):
        return
    if not interaction.guild:
        await interaction.followup.send("Command ini hanya bisa dipakai di server.", ephemeral=True)
        return

    session = bot.sessions.pop(interaction.guild.id, None)
    if session:
        with contextlib.suppress(Exception):
            await session.disconnect()

    # Bersihkan voice client stale yang masih disimpan discord.py.
    stale = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if stale:
        with contextlib.suppress(Exception):
            await stale.disconnect(force=True)

    await interaction.followup.send("✅ Voice state sudah di-reset. Tunggu 5 detik, lalu pakai `/join` lagi.", ephemeral=True)


@bot.tree.command(name="listen", description="Cek status fitur dengar mic langsung.")
async def listen(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)
    if not await require_bot_admin(interaction, action="mengaktifkan listener mic"):
        return
    try:
        session = await ensure_voice_session(interaction, receive=False)
        if session is None:
            return
        await session.start_listening()
    except Exception as exc:  # noqa: BLE001
        await interaction.followup.send(
            f"⚠️ Listener mic langsung belum bisa dipakai di patch ini: `{exc}`\n"
            "Pakai `/talk` atau `/say` dulu. Patch ini fokus memperbaiki `/join` yang kena 4017/DAVE.",
            ephemeral=True,
        )
        return
    await interaction.followup.send("🎧 Mode dengar mic aktif.", ephemeral=True)


@bot.tree.command(name="stopvoice", description="Matikan mode dengar voice dan hentikan audio yang sedang diputar.")
async def stopvoice(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)
    if not await require_bot_admin(interaction, action="menghentikan voice bot"):
        return
    if not interaction.guild:
        await interaction.followup.send("Command ini hanya bisa dipakai di server.", ephemeral=True)
        return
    session = bot.sessions.get(interaction.guild.id)
    if not session or not session.voice_client:
        await interaction.followup.send("Aku belum aktif di voice.", ephemeral=True)
        return
    await session.stop_listening()
    if session.voice_client.is_playing():
        session.voice_client.stop()
    await interaction.followup.send("🛑 Voice listening/playback dihentikan.", ephemeral=True)


@bot.tree.command(name="stoplisten", description="Matikan mode dengar mic, tapi bot tetap di voice.")
async def stoplisten(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)
    if not await require_bot_admin(interaction, action="menghentikan listener mic"):
        return
    if not interaction.guild:
        await interaction.followup.send("Command ini hanya bisa dipakai di server.", ephemeral=True)
        return
    session = bot.sessions.get(interaction.guild.id)
    if not session or not session.voice_client or not session.voice_client.is_connected():
        await interaction.followup.send("Aku belum aktif di voice.", ephemeral=True)
        return
    await session.stop_listening()
    await interaction.followup.send("🎧 Mode dengar mic dimatikan. Aku tetap di voice.", ephemeral=True)


@bot.tree.command(name="mulaibicara", description="Nyalakan lagi suara bot di voice.")
async def mulaibicara(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)
    if not await require_bot_admin(interaction, action="mengaktifkan suara bot"):
        return
    try:
        session = await ensure_voice_session(interaction)
    except Exception as exc:  # noqa: BLE001
        await interaction.followup.send(f"⚠️ Gagal connect voice: `{type(exc).__name__}: {exc}`", ephemeral=True)
        return
    if session is None:
        return
    session.speech_enabled = True
    await interaction.followup.send("✅ Suara bot dinyalakan. Pakai `/talk` atau `/say` untuk tes suara.", ephemeral=True)


@bot.tree.command(name="berhentibicara", description="Matikan suara bot di voice tanpa mengeluarkan bot dari channel.")
async def berhentibicara(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)
    if not await require_bot_admin(interaction, action="mematikan suara bot"):
        return
    if not interaction.guild:
        await interaction.followup.send("Command ini hanya bisa dipakai di server.", ephemeral=True)
        return
    session = bot.sessions.get(interaction.guild.id)
    if not session or not session.voice_client or not session.voice_client.is_connected():
        await interaction.followup.send("Aku belum aktif di voice.", ephemeral=True)
        return
    session.speech_enabled = False
    await session.stop_speaking()
    await interaction.followup.send("🔇 Suara bot dimatikan. Aku masih bisa tetap di voice, tapi tidak akan ngomong dulu.", ephemeral=True)


@bot.tree.command(name="stopspeak", description="Hentikan audio bot yang sedang diputar sekarang.")
async def stopspeak(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)
    if not await require_bot_admin(interaction, action="menghentikan suara bot"):
        return
    if not interaction.guild:
        await interaction.followup.send("Command ini hanya bisa dipakai di server.", ephemeral=True)
        return
    session = bot.sessions.get(interaction.guild.id)
    if not session or not session.voice_client or not session.voice_client.is_connected():
        await interaction.followup.send("Aku belum aktif di voice.", ephemeral=True)
        return
    await session.stop_speaking()
    await interaction.followup.send("⏹️ Audio yang sedang diputar sudah dihentikan.", ephemeral=True)


@bot.tree.command(name="talk", description="Tanya lewat teks, bot jawab di chat dan kalau sedang di voice akan dibacakan.")
@app_commands.describe(prompt="Pertanyaan atau pesan untuk AI")
async def talk(interaction: discord.Interaction, prompt: str) -> None:
    await interaction.response.defer()
    if not await require_public_ai_or_admin(interaction):
        return

    wait = cooldown_left(interaction.user.id)
    if wait:
        await interaction.followup.send(f"Tunggu {wait} detik dulu ya biar Gemini tidak kena limit.", ephemeral=True)
        return

    if not interaction.guild:
        conversation_id = f"dm:{interaction.user.id}"
    else:
        conversation_id = f"text:{interaction.guild.id}:{interaction.user.id}"

    try:
        answer = await ai.chat(conversation_id, prompt, speaker_name=interaction.user.display_name)
    except Exception as exc:  # noqa: BLE001
        await interaction.followup.send(friendly_ai_error(exc))
        return

    await interaction.followup.send(answer[:1900])

    if interaction.guild:
        session = bot.sessions.get(interaction.guild.id)
        can_voice_reply = settings.public_ai_voice_reply or can_control_bot_member(interaction.user)
        if can_voice_reply and session and session.voice_client and session.voice_client.is_connected():
            try:
                await session.speak(answer)
            except Exception as exc:  # noqa: BLE001
                await interaction.followup.send(friendly_ai_error(exc)[:1900])


@bot.tree.command(name="say", description="Bot membacakan teks ke voice channel pakai TTS lokal/default.")
@app_commands.describe(text="Teks yang ingin dibacakan bot")
async def say(interaction: discord.Interaction, text: str) -> None:
    await interaction.response.defer(ephemeral=True)
    if not await require_bot_admin(interaction, action="membuat bot berbicara di voice"):
        return
    session = await ensure_voice_session(interaction)
    if session is None:
        return
    await interaction.followup.send("🔊 Oke, aku bacakan pakai TTS default/lokal.", ephemeral=True)
    try:
        await session.speak(text)
    except Exception as exc:  # noqa: BLE001
        await interaction.followup.send(friendly_ai_error(exc), ephemeral=True)


@bot.tree.command(name="saypremium", description="Bot membacakan teks pakai Gemini TTS yang lebih natural.")
@app_commands.describe(text="Teks yang ingin dibacakan dengan suara premium Gemini")
async def saypremium(interaction: discord.Interaction, text: str) -> None:
    await interaction.response.defer(ephemeral=True)
    if not await require_bot_admin(interaction, action="membuat bot berbicara premium"):
        return
    session = await ensure_voice_session(interaction)
    if session is None:
        return
    await interaction.followup.send("✨ Oke, aku bacakan pakai Gemini Premium TTS.", ephemeral=True)
    try:
        await session.speak(text, premium=True)
    except Exception as exc:  # noqa: BLE001
        await interaction.followup.send(friendly_ai_error(exc), ephemeral=True)


@bot.tree.command(name="serverinfo", description="Lihat ringkasan data server: member, channel, admin, dan event.")
async def serverinfo(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)
    if not await require_public_ai_or_admin(interaction):
        return
    if not interaction.guild:
        await interaction.followup.send("Command ini hanya bisa dipakai di server.", ephemeral=True)
        return
    try:
        snapshot = await build_server_snapshot(interaction.guild)
        await send_followup_long(interaction, f"```txt\n{format_snapshot(snapshot)}\n```", ephemeral=True)
    except Exception as exc:  # noqa: BLE001
        await interaction.followup.send(f"⚠️ Gagal membaca data server: `{type(exc).__name__}: {exc}`", ephemeral=True)


@bot.tree.command(name="admins", description="Deteksi owner dan member/role yang punya izin Administrator.")
async def admins(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)
    if not await require_public_ai_or_admin(interaction):
        return
    if not interaction.guild:
        await interaction.followup.send("Command ini hanya bisa dipakai di server.", ephemeral=True)
        return
    try:
        snapshot = await build_server_snapshot(interaction.guild)
        roles = ", ".join(snapshot.admin_roles) or "Tidak ada role Administrator yang terbaca."
        members = "\n".join(f"- {name}" for name in snapshot.admin_members_sample) or "- Tidak ada admin yang terbaca."
        text = (
            f"**Owner:** {snapshot.owner}\n"
            f"**Role Administrator:** {roles}\n"
            f"**Admin terbaca:** {snapshot.admin_members_count}\n"
            f"**Sampel admin:**\n{members}\n\n"
            "Catatan: agar daftar admin akurat, aktifkan **Server Members Intent** di Discord Developer Portal."
        )
        await send_followup_long(interaction, text, ephemeral=True)
    except Exception as exc:  # noqa: BLE001
        await interaction.followup.send(f"⚠️ Gagal membaca admin: `{type(exc).__name__}: {exc}`", ephemeral=True)


@bot.tree.command(name="events", description="Lihat acara/event terjadwal di server Discord.")
async def events(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)
    if not await require_public_ai_or_admin(interaction):
        return
    if not interaction.guild:
        await interaction.followup.send("Command ini hanya bisa dipakai di server.", ephemeral=True)
        return
    try:
        snapshot = await build_server_snapshot(interaction.guild)
        if not snapshot.scheduled_events:
            await interaction.followup.send("Belum ada event terjadwal yang bisa kubaca.", ephemeral=True)
            return
        text = "**Event terjadwal:**\n" + "\n".join(f"- {event}" for event in snapshot.scheduled_events)
        await send_followup_long(interaction, text, ephemeral=True)
    except Exception as exc:  # noqa: BLE001
        await interaction.followup.send(f"⚠️ Gagal membaca event: `{type(exc).__name__}: {exc}`", ephemeral=True)


@bot.tree.command(name="ragscan", description="Masukkan isi pesan channel ke database RAG lokal bot.")
@app_commands.describe(channel="Channel teks yang mau dibaca, misalnya #peraturan", limit="Jumlah pesan terakhir yang dibaca")
async def ragscan(interaction: discord.Interaction, channel: discord.TextChannel, limit: int = 100) -> None:
    await interaction.response.defer(ephemeral=True)
    if not await require_bot_admin(interaction, action="scan RAG"):
        return
    if not interaction.guild:
        await interaction.followup.send("Command ini hanya bisa dipakai di server.", ephemeral=True)
        return
    if not can_manage_server(interaction):
        await interaction.followup.send("Command ini hanya untuk Admin/Manage Server, supaya RAG tidak discan sembarangan.", ephemeral=True)
        return
    if channel.guild.id != interaction.guild.id:
        await interaction.followup.send("Channel itu bukan dari server ini.", ephemeral=True)
        return

    limit = max(1, min(int(limit), settings.rag_scan_limit))
    try:
        items = await scan_text_channel(channel, limit=limit)
        added = knowledge_store.add_items(interaction.guild.id, items)
        stats = knowledge_store.stats(interaction.guild.id)
        await interaction.followup.send(
            f"✅ RAG scan selesai untuk #{channel.name}.\n"
            f"Pesan terbaca: **{len(items)}** | Baru masuk database: **{added}**\n"
            f"Total database lokal: **{stats['messages']} pesan** dari **{stats['channels']} channel**.\n"
            "Sekarang tanyakan dengan `/serverask question:` atau `/ragask question:`.",
            ephemeral=True,
        )
    except discord.Forbidden:
        await interaction.followup.send(
            "⚠️ Aku tidak punya izin membaca channel itu. Beri izin **View Channel** dan **Read Message History**.",
            ephemeral=True,
        )
    except Exception as exc:  # noqa: BLE001
        await interaction.followup.send(f"⚠️ Gagal scan RAG: `{type(exc).__name__}: {exc}`", ephemeral=True)


@bot.tree.command(name="ragstatus", description="Cek jumlah data RAG lokal yang sudah disimpan.")
async def ragstatus(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)
    if not await require_bot_admin(interaction, action="melihat status RAG"):
        return
    if not interaction.guild:
        await interaction.followup.send("Command ini hanya bisa dipakai di server.", ephemeral=True)
        return
    stats = knowledge_store.stats(interaction.guild.id)
    await interaction.followup.send(
        f"📚 Database RAG lokal: **{stats['messages']} pesan** dari **{stats['channels']} channel**.\n"
        "Tambah data dengan `/ragscan channel:#channel limit:100`.",
        ephemeral=True,
    )


@bot.tree.command(name="ragclear", description="Hapus database RAG lokal server ini.")
async def ragclear(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)
    if not await require_bot_admin(interaction, action="menghapus database RAG"):
        return
    if not interaction.guild:
        await interaction.followup.send("Command ini hanya bisa dipakai di server.", ephemeral=True)
        return
    if not can_manage_server(interaction):
        await interaction.followup.send("Command ini hanya untuk Admin/Manage Server.", ephemeral=True)
        return
    knowledge_store.clear(interaction.guild.id)
    await interaction.followup.send("✅ Database RAG lokal server ini sudah dihapus.", ephemeral=True)


@bot.tree.command(name="ragask", description="Tanya AI berdasarkan dokumen/channel yang sudah discan RAG.")
@app_commands.describe(question="Pertanyaan tentang isi channel/dokumen server yang sudah discan")
async def ragask(interaction: discord.Interaction, question: str) -> None:
    await interaction.response.defer()
    if not await require_public_ai_or_admin(interaction):
        return
    if not interaction.guild:
        await interaction.followup.send("Command ini hanya bisa dipakai di server.", ephemeral=True)
        return
    wait = cooldown_left(interaction.user.id)
    if wait:
        await interaction.followup.send(f"Tunggu {wait} detik dulu ya biar Gemini tidak kena limit.", ephemeral=True)
        return

    if is_sensitive_server_question(question) and not can_control_bot_member(interaction.user):
        await interaction.followup.send(
            "⚠️ Aku tidak bisa membocorkan token, API key, secret, `.env`, webhook, atau menjalankan operasi RAG sensitif untuk member biasa.",
            ephemeral=True,
        )
        return

    items = knowledge_store.search(interaction.guild.id, question, top_k=settings.rag_top_k)
    items = filter_rag_for_viewer(interaction.guild, interaction.user, items)
    context = format_rag_context(items, max_chars=settings.rag_max_context_chars)
    if not can_control_bot_member(interaction.user):
        context = (
            "AKSES USER: Member biasa. Jawab hanya dari channel yang user ini boleh lihat. "
            "Info owner/admin/moderator/staff/dev boleh dijawab jika ada di konteks. "
            "Jangan bocorkan token/API key/.env/webhook atau isi channel yang user tidak punya akses.\n\n" + context
        )
    try:
        answer = await ai.chat_with_context(
            f"rag:{interaction.guild.id}:{interaction.user.id}",
            question,
            context=context,
            speaker_name=interaction.user.display_name,
        )
        await interaction.followup.send(answer[:1900])
    except Exception as exc:  # noqa: BLE001
        await interaction.followup.send(friendly_ai_error(exc))


@bot.tree.command(name="serverask", description="Tanya AI pakai data server + RAG channel yang sudah discan.")
@app_commands.describe(question="Pertanyaan tentang server, rules, admin, channel, event, atau dokumen RAG")
async def serverask(interaction: discord.Interaction, question: str) -> None:
    await interaction.response.defer()
    if not await require_public_ai_or_admin(interaction):
        return
    if not interaction.guild:
        await interaction.followup.send("Command ini hanya bisa dipakai di server.", ephemeral=True)
        return
    wait = cooldown_left(interaction.user.id)
    if wait:
        await interaction.followup.send(f"Tunggu {wait} detik dulu ya biar Gemini tidak kena limit.", ephemeral=True)
        return

    try:
        answer = await answer_with_server_context(
            conversation_id=f"serverask:{interaction.guild.id}:{interaction.user.id}",
            question=question,
            guild=interaction.guild,
            viewer=interaction.user,
            speaker_name=interaction.user.display_name,
        )
        await interaction.followup.send(answer[:1900])
    except Exception as exc:  # noqa: BLE001
        await interaction.followup.send(friendly_ai_error(exc))



@bot.tree.command(name="dm", description="Kirim DM ke user ID. Bisa sekalian upload gambar.")
@app_commands.describe(
    user_id="ID user target, contoh 123456789012345678 atau mention user",
    isi_pesan="Isi pesan yang akan dikirim ke DM target",
    gambar="Opsional: gambar yang ikut dikirim ke DM",
)
async def dm_command(
    interaction: discord.Interaction,
    user_id: str,
    isi_pesan: str,
    gambar: discord.Attachment | None = None,
) -> None:
    await send_dm_command_impl(interaction, user_id=user_id, isi_pesan=isi_pesan, gambar=gambar)


@bot.tree.command(name="kirimdm", description="Kirim DM ke user ID. Versi cepat dari /kirim dm.")
@app_commands.describe(
    user_id="ID user target, contoh 123456789012345678 atau mention user",
    isi_pesan="Isi pesan yang akan dikirim ke DM target",
    gambar="Opsional: gambar yang ikut dikirim ke DM",
)
async def kirimdm_command(
    interaction: discord.Interaction,
    user_id: str,
    isi_pesan: str,
    gambar: discord.Attachment | None = None,
) -> None:
    await send_dm_command_impl(interaction, user_id=user_id, isi_pesan=isi_pesan, gambar=gambar)


kirim_group = app_commands.Group(name="kirim", description="Command kirim pesan dari GLADIATOR.")


@kirim_group.command(name="dm", description="Kirim DM ke user ID. Bisa sekalian upload gambar.")
@app_commands.describe(
    user_id="ID user target, contoh 123456789012345678 atau mention user",
    isi_pesan="Isi pesan yang akan dikirim ke DM target",
    gambar="Opsional: gambar yang ikut dikirim ke DM",
)
async def kirim_dm_command(
    interaction: discord.Interaction,
    user_id: str,
    isi_pesan: str,
    gambar: discord.Attachment | None = None,
) -> None:
    await send_dm_command_impl(interaction, user_id=user_id, isi_pesan=isi_pesan, gambar=gambar)


bot.tree.add_command(kirim_group)


# =====================================
# Scam Warning / Block Commands
# =====================================

@bot.tree.command(name="mrbeastblock", description="Kirim peringatan DM scam Mr. Beast ke user.")
@app_commands.describe(
    user_id="ID user yang menyebarkan scam Mr. Beast",
    alasan="Alasan peringatan",
)
async def mrbeastblock(
    interaction: discord.Interaction,
    user_id: str,
    alasan: str = "Menyebarkan scam palsu mengatasnamakan Mr. Beast",
) -> None:
    if not await require_bot_admin(interaction, action="memblokir scam Mr. Beast"):
        return

    await interaction.response.defer(ephemeral=True, thinking=True)

    target_id = parse_discord_user_id(user_id)
    if target_id is None:
        await interaction.followup.send("User ID tidak valid.", ephemeral=True)
        return

    target = None
    if interaction.guild:
        target = interaction.guild.get_member(target_id)
    if target is None:
        try:
            target = await bot.fetch_user(target_id)
        except discord.NotFound:
            await interaction.followup.send("User tidak ditemukan.", ephemeral=True)
            return
        except discord.HTTPException as exc:
            if _is_discord_global_429(exc):
                wait = _mark_discord_api_pause(exc)
                await interaction.followup.send(f"Discord API rate-limit. Coba lagi {wait} detik.", ephemeral=True)
                return
            await interaction.followup.send(f"Gagal cari user: HTTP {getattr(exc, 'status', 'unknown')}.", ephemeral=True)
            return

    target_label = str(target)
    guild_name = interaction.guild.name if interaction.guild else "Discord"

    dm_message = (
        "**PERINGATAN RESMI DARI GLADIATOR GUARD**\n\n"
        "Akun kamu terdeteksi menyebarkan **SCAM MR. BEAST** di server **{guild}**.\n\n"
        "Akun kamu mengirim pesan/gambar palsu yang mengatasnamakan Mr. Beast "
        "(giveaway/crypto/casino palsu) untuk menipu anggota server lain.\n\n"
        "AKUN KAMU MUNGKIN SUDAH DIRETAS! "
        "Scam Mr. Beast biasanya menyebar melalui akun yang dicuri.\n\n"
        "1. Ganti password Discord dari perangkat BERSIH\n"
        "2. Scan komputer dengan antivirus\n"
        "3. Cek Authorized Apps Discord, cabut aplikasi mencurigakan\n"
        "4. Log out dari semua perangkat\n"
        "5. Aktifkan 2FA jika belum\n\n"
        "Alasan: {reason}\n\n"
        "Hubungi admin server jika kamu merasa ini kesalahan."
        .format(
            guild=guild_name,
            reason=alasan,
        )
    )

    try:
        await target.send(dm_message, allowed_mentions=discord.AllowedMentions.none())
    except discord.Forbidden:
        await interaction.followup.send(f"DM tidak terkirim ke **{target_label}** - DM tertutup.", ephemeral=True)
        return
    except discord.HTTPException as exc:
        if _is_discord_global_429(exc):
            wait = _mark_discord_api_pause(exc)
            await interaction.followup.send(f"Discord API rate-limit. Coba lagi {wait} detik.", ephemeral=True)
            return
        await interaction.followup.send(f"Gagal kirim DM: HTTP {getattr(exc, 'status', 'unknown')}.", ephemeral=True)
        return
    except Exception as exc:
        await interaction.followup.send(f"Gagal kirim DM: {type(exc).__name__}: {exc}", ephemeral=True)
        return

    await log_direct_dm_action(
        interaction, target_id=target_id, target_label=target_label,
        ok=True, reason=f"MRBEAST WARNING: {alasan[:200]}", has_image=False,
    )
    await interaction.followup.send(
        f"Peringatan scam Mr. Beast terkirim ke **{target_label}** (`{target_id}`).",
        ephemeral=True,
    )


@bot.tree.command(name="scamwarn", description="Kirim peringatan scam umum ke DM user.")
@app_commands.describe(
    user_id="ID user target",
    jenis_scam="Jenis scam (mrbeast, crypto, celebrity, general)",
    alasan="Alasan peringatan",
)
@app_commands.choices(jenis_scam=[
    app_commands.Choice(name="Mr. Beast Scam", value="mrbeast"),
    app_commands.Choice(name="Crypto/Casino Scam", value="crypto"),
    app_commands.Choice(name="Celebrity Scam", value="celebrity"),
    app_commands.Choice(name="General Scam", value="general"),
])
async def scamwarn(
    interaction: discord.Interaction,
    user_id: str,
    jenis_scam: str = "general",
    alasan: str = "Menyebarkan konten scam di server",
) -> None:
    if not await require_bot_admin(interaction, action="mengirim peringatan scam"):
        return

    await interaction.response.defer(ephemeral=True, thinking=True)

    target_id = parse_discord_user_id(user_id)
    if target_id is None:
        await interaction.followup.send("User ID tidak valid.", ephemeral=True)
        return

    alasan = alasan.strip()[:1800]

    target = None
    if interaction.guild:
        target = interaction.guild.get_member(target_id)
    if target is None:
        try:
            target = await bot.fetch_user(target_id)
        except discord.NotFound:
            await interaction.followup.send("User tidak ditemukan.", ephemeral=True)
            return
        except discord.HTTPException as exc:
            if _is_discord_global_429(exc):
                wait = _mark_discord_api_pause(exc)
                await interaction.followup.send(f"Discord API rate-limit. Coba lagi {wait} detik.", ephemeral=True)
                return
            await interaction.followup.send(f"Gagal cari user: HTTP {getattr(exc, 'status', 'unknown')}.", ephemeral=True)
            return

    target_label = str(target)
    guild_name = interaction.guild.name if interaction.guild else "Discord"

    templates = {
        "mrbeast": "PERINGATAN SCAM MR. BEAST - Akun kamu di **{guild}** terdeteksi menyebarkan scam palsu yang mengatasnamakan Mr. Beast.",
        "crypto": "PERINGATAN SCAM CRYPTO/CASINO - Akun kamu di **{guild}** terdeteksi menyebarkan promosi scam crypto, casino, atau withdrawal palsu.",
        "celebrity": "PERINGATAN SCAM SELEBRITI - Akun kamu di **{guild}** terdeteksi menyebarkan scam yang mengatasnamakan figur publik.",
        "general": "PERINGATAN SCAM DARI GLADIATOR GUARD - Akun kamu di **{guild}** terdeteksi menyebarkan konten scam.",
    }

    template = templates.get(jenis_scam, templates["general"])
    dm_message = (
        template.format(guild=guild_name)
        + "\n\nAlasan: " + alasan
        + "\n\nHubungi admin server jika kamu merasa ini kesalahan."
    )

    try:
        await target.send(dm_message, allowed_mentions=discord.AllowedMentions.none())
    except discord.Forbidden:
        await interaction.followup.send(f"DM tidak terkirim ke **{target_label}** - DM tertutup.", ephemeral=True)
        return
    except discord.HTTPException as exc:
        if _is_discord_global_429(exc):
            wait = _mark_discord_api_pause(exc)
            await interaction.followup.send(f"Discord API rate-limit. Coba lagi {wait} detik.", ephemeral=True)
            return
        await interaction.followup.send(f"Gagal kirim DM: HTTP {getattr(exc, 'status', 'unknown')}.", ephemeral=True)
        return
    except Exception as exc:
        await interaction.followup.send(f"Gagal kirim DM: {type(exc).__name__}: {exc}", ephemeral=True)
        return

    await log_direct_dm_action(
        interaction, target_id=target_id, target_label=target_label,
        ok=True, reason=f"SCAMWARN ({jenis_scam}): {alasan[:200]}", has_image=False,
    )
    await interaction.followup.send(
        f"Peringatan scam ({jenis_scam}) terkirim ke **{target_label}** (`{target_id}`).",
        ephemeral=True,
    )


# =====================================
# Send-to: Owner DM + file + AI compose
# =====================================

@bot.tree.command(name="sendto", description="[OWNER] Kirim DM + file ke user, bisa AI bantu compose pesan.")
@app_commands.describe(
    user_id="ID user target atau mention",
    isi_pesan="Isi pesan (opsional jika pakai AI, wajib jika manual)",
    file="File lampiran (gambar, audio, PDF, dokumen) maks 8MB",
    bantuan_ai="True = AI bantu susun pesan dari instruksi",
)
@app_commands.choices(bantuan_ai=[
    app_commands.Choice(name="Ya, AI bantu compose", value="true"),
    app_commands.Choice(name="Tidak, kirim manual", value="false"),
])
async def sendto(
    interaction: discord.Interaction,
    user_id: str,
    isi_pesan: str = "",
    file: discord.Attachment | None = None,
    bantuan_ai: str = "false",
) -> None:
    """Kirim DM dengan file attachment ke user. Bisa AI bantu compose pesan."""
    if interaction.user.id not in settings.bot_owner_ids:
        await interaction.response.send_message(
            "Command /sendto hanya untuk Owner Bot di BOT_OWNER_IDS.\nPakai /dm untuk kirim DM biasa.",
            ephemeral=True,
        )
        return

    await interaction.response.defer(ephemeral=True, thinking=True)

    target_id = parse_discord_user_id(user_id)
    if target_id is None:
        await interaction.followup.send("User ID tidak valid.", ephemeral=True)
        return

    target = None
    if interaction.guild:
        target = interaction.guild.get_member(target_id)
    if target is None:
        try:
            target = await bot.fetch_user(target_id)
        except discord.NotFound:
            await interaction.followup.send("User tidak ditemukan.", ephemeral=True)
            return
        except discord.HTTPException as exc:
            if _is_discord_global_429(exc):
                wait = _mark_discord_api_pause(exc)
                await interaction.followup.send(f"Discord API rate-limit. Coba lagi {wait} detik.", ephemeral=True)
                return
            await interaction.followup.send(f"Gagal cari user: HTTP {getattr(exc, 'status', 'unknown')}.", ephemeral=True)
            return

    target_label = str(target)

    file_to_send: discord.File | None = None
    has_file = file is not None
    max_bytes = int(os.getenv("DM_IMAGE_MAX_BYTES", "8000000"))

    if file is not None:
        if file.size and file.size > max_bytes:
            await interaction.followup.send(f"File terlalu besar. Maksimal {max_bytes // 1_000_000} MB.", ephemeral=True)
            return
        try:
            try:
                file_to_send = await file.to_file(use_cached=True)
            except TypeError:
                file_to_send = await file.to_file()
        except discord.HTTPException as exc:
            if _is_discord_global_429(exc):
                wait = _mark_discord_api_pause(exc)
                await interaction.followup.send(f"Discord API rate-limit. Coba lagi {wait} detik.", ephemeral=True)
                return
            await interaction.followup.send(f"Gagal baca file: HTTP {getattr(exc, 'status', 'unknown')}.", ephemeral=True)
            return
        except Exception as exc:
            await interaction.followup.send(f"Gagal proses file: {type(exc).__name__}: {exc}", ephemeral=True)
            return

    use_ai = bantuan_ai == "true"
    user_prompt = (isi_pesan or "").strip()
    final_message = ""
    ai_generated = False

    if use_ai:
        if not user_prompt:
            await interaction.followup.send(
                "Kamu pilih AI, tapi isi_pesan kosong.\n"
                "Tulis INSTRUKSI di isi_pesan, contoh: 'Beri peringatan spam' atau 'Buat pengumuman resmi'.",
                ephemeral=True,
            )
            return

        guild_name = interaction.guild.name if interaction.guild else "server Discord"
        ai_prompt = (
            f"Tujuan: {user_prompt}\n"
            f"Pengirim: Staff server {guild_name}\n"
            f"Nama staff: {interaction.user.display_name}\n"
            f"Target: User Discord\n\n"
            "Buat pesan DM sesuai tujuan. Bijaksana, tegas jika perlu, sopan, profesional."
        )

        try:
            ai_text = await ai.compose_dm_message(ai_prompt)
        except Exception as exc:
            await interaction.followup.send(f"AI gagal: {friendly_ai_error(exc)}", ephemeral=True)
            return

        if not ai_text.strip():
            await interaction.followup.send("AI tidak menghasilkan teks. Coba lagi.", ephemeral=True)
            return

        final_message = ai_text.strip()
        ai_generated = True

        preview = f"AI telah menyusun pesan:\n\n{final_message}\n\nFile: {'Ada' if has_file else 'Tidak'}\n\nKetik `kirim` untuk kirim, `batal` untuk batalkan. (30 detik)"
        await interaction.followup.send(preview[:1900], ephemeral=True)

        def check(msg):
            return (
                msg.author.id == interaction.user.id
                and msg.channel.id == interaction.channel_id
                and msg.content.strip().lower() in ("kirim", "batal")
            )

        try:
            reply = await bot.wait_for("message", timeout=30.0, check=check)
        except asyncio.TimeoutError:
            await interaction.followup.send("Waktu habis. Ulangi perintah.", ephemeral=True)
            return

        if reply.content.strip().lower() == "batal":
            await interaction.followup.send("Dibatalkan.", ephemeral=True)
            return

        await interaction.followup.send("Mengirim...", ephemeral=True)

    else:
        if not user_prompt:
            await interaction.followup.send(
                "isi_pesan tidak boleh kosong. Tulis pesan, atau aktifkan bantuan_ai.",
                ephemeral=True,
            )
            return
        final_message = user_prompt

    if len(final_message) > 1900:
        final_message = final_message[:1900] + "..."

    guild_name = interaction.guild.name if interaction.guild else "Discord"
    dm_text = f"Pesan dari staff server {guild_name}\n\n{final_message}"

    try:
        await target.send(content=dm_text, file=file_to_send, allowed_mentions=discord.AllowedMentions.none())
    except discord.Forbidden:
        await log_direct_dm_action(interaction, target_id=target_id, target_label=target_label, ok=False,
            reason="DM tertutup", has_image=has_file)
        await interaction.followup.send("DM tidak terkirim: DM target tertutup.", ephemeral=True)
        return
    except discord.HTTPException as exc:
        if _is_discord_global_429(exc):
            wait = _mark_discord_api_pause(exc)
            await interaction.followup.send(f"Discord API rate-limit. Coba lagi {wait} detik.", ephemeral=True)
            return
        await interaction.followup.send(f"Gagal kirim DM: HTTP {getattr(exc, 'status', 'unknown')}.", ephemeral=True)
        return
    except Exception as exc:
        await interaction.followup.send(f"Gagal kirim DM: {type(exc).__name__}: {exc}", ephemeral=True)
        return

    await log_direct_dm_action(
        interaction, target_id=target_id, target_label=target_label,
        ok=True, reason=f"SendTo {'AI' if ai_generated else 'manual'}", has_image=has_file,
    )

    await interaction.followup.send(
        f"DM terkirim ke {target_label} ({target_id}) via {'AI' if ai_generated else 'manual'}."
        + (" File terkirim." if has_file else ""),
        ephemeral=True,
    )


# =========================
# Slash command cleaner

# =========================
# Dipakai ketika command Discord dobel/kebanyakan karena pernah sync global + guild berkali-kali.
# Command ini tetap dibatasi untuk owner/admin/role tinggi.

SLASH_SCOPE_CHOICES = [
    app_commands.Choice(name="Server ini saja", value="guild"),
    app_commands.Choice(name="Global semua server", value="global"),
    app_commands.Choice(name="Server + Global", value="both"),
]

SLASH_KEEP_CHOICES = [
    app_commands.Choice(name="Keep Global, hapus duplikat server", value="global"),
    app_commands.Choice(name="Keep Server, hapus duplikat global", value="guild"),
]


def _format_command_names(commands: list[discord.app_commands.AppCommand], *, limit: int = 35) -> str:
    if not commands:
        return "- kosong"
    lines = []
    for cmd in commands[:limit]:
        lines.append(f"- `/{cmd.name}`  id:`{cmd.id}`")
    if len(commands) > limit:
        lines.append(f"...dan {len(commands) - limit} command lain")
    return "\n".join(lines)


async def _safe_fetch_app_commands(scope: str, guild: discord.Guild | None) -> list[tuple[str, discord.Guild | None, list[discord.app_commands.AppCommand]]]:
    fetched: list[tuple[str, discord.Guild | None, list[discord.app_commands.AppCommand]]] = []
    if scope in {"guild", "server", "both"}:
        if guild is not None:
            guild_commands = await bot.tree.fetch_commands(guild=guild)
            fetched.append(("server", guild, guild_commands))
    if scope in {"global", "both"}:
        global_commands = await bot.tree.fetch_commands(guild=None)
        fetched.append(("global", None, global_commands))
    return fetched


async def _delete_app_command(cmd: discord.app_commands.AppCommand, guild: discord.Guild | None) -> None:
    # discord.py AppCommand biasanya punya .delete(). Fallback ke HTTP internal kalau tidak ada.
    try:
        await cmd.delete()
        return
    except AttributeError:
        pass

    app_id = bot.application_id
    if app_id is None:
        app_info = await bot.application_info()
        app_id = app_info.id

    if guild is None:
        await bot.http.delete_global_command(app_id, cmd.id)
    else:
        await bot.http.delete_guild_command(app_id, guild.id, cmd.id)


slash_group = app_commands.Group(name="slash", description="Kelola dan bersihkan slash command GLADIATOR.")


@slash_group.command(name="list", description="Lihat command yang terdaftar di server/global.")
@app_commands.describe(scope="Pilih lokasi command yang mau dicek")
@app_commands.choices(scope=SLASH_SCOPE_CHOICES)
async def slash_list(interaction: discord.Interaction, scope: str = "both") -> None:
    if not await require_bot_admin(interaction, action="melihat daftar slash command"):
        return
    await interaction.response.defer(ephemeral=True)
    try:
        fetched = await _safe_fetch_app_commands(scope, interaction.guild)
        chunks = []
        for label, _guild, commands in fetched:
            chunks.append(f"**{label.upper()} commands ({len(commands)}):**\n{_format_command_names(commands)}")
        await interaction.followup.send("\n\n".join(chunks)[:1900], ephemeral=True)
    except Exception as exc:  # noqa: BLE001
        if _is_discord_global_429(exc):
            wait = _mark_discord_api_pause(exc)
            await interaction.followup.send(f"⚠️ Discord API lagi rate limit. Coba lagi sekitar {wait//60} menit lagi.", ephemeral=True)
            return
        await interaction.followup.send(f"⚠️ Gagal baca slash command: `{type(exc).__name__}: {exc}`", ephemeral=True)


@slash_group.command(name="delete", description="Hapus slash command tertentu dari server/global.")
@app_commands.describe(
    name="Nama command tanpa garis miring. Contoh: dm, kirimdm, kirim, aihelp",
    scope="Lokasi command yang mau dihapus",
)
@app_commands.choices(scope=SLASH_SCOPE_CHOICES)
async def slash_delete(interaction: discord.Interaction, name: str, scope: str = "guild") -> None:
    if not await require_bot_admin(interaction, action="menghapus slash command"):
        return
    await interaction.response.defer(ephemeral=True)
    target_name = name.strip().lower().lstrip("/")
    if target_name in {"slash"}:
        await interaction.followup.send("⚠️ Aku tidak hapus `/slash` supaya masih ada alat buat bersihin command lain.", ephemeral=True)
        return

    deleted: list[str] = []
    failed: list[str] = []
    try:
        for label, guild_obj, commands in await _safe_fetch_app_commands(scope, interaction.guild):
            for cmd in commands:
                if cmd.name.lower() == target_name:
                    try:
                        await _delete_app_command(cmd, guild_obj)
                        deleted.append(f"/{cmd.name} ({label})")
                    except Exception as exc:  # noqa: BLE001
                        failed.append(f"/{cmd.name} ({label}) -> {type(exc).__name__}: {exc}")
                        if _is_discord_global_429(exc):
                            _mark_discord_api_pause(exc)
                            break
        if deleted:
            msg = "✅ Terhapus:\n" + "\n".join(f"- {x}" for x in deleted)
        else:
            msg = f"ℹ️ Tidak ada command `/{target_name}` di scope `{scope}`."
        if failed:
            msg += "\n\n⚠️ Gagal:\n" + "\n".join(f"- {x}" for x in failed[:5])
        await interaction.followup.send(msg[:1900], ephemeral=True)
    except Exception as exc:  # noqa: BLE001
        if _is_discord_global_429(exc):
            wait = _mark_discord_api_pause(exc)
            await interaction.followup.send(f"⚠️ Discord API lagi rate limit. Coba lagi sekitar {wait//60} menit lagi.", ephemeral=True)
            return
        await interaction.followup.send(f"⚠️ Gagal hapus command: `{type(exc).__name__}: {exc}`", ephemeral=True)


@slash_group.command(name="dedupe", description="Hapus command dobel antara server/global atau dobel dalam satu scope.")
@app_commands.describe(
    scope="Biasanya pilih Server + Global",
    keep="Kalau command sama ada di server dan global, pilih yang mau disimpan",
)
@app_commands.choices(scope=SLASH_SCOPE_CHOICES, keep=SLASH_KEEP_CHOICES)
async def slash_dedupe(interaction: discord.Interaction, scope: str = "both", keep: str = "global") -> None:
    if not await require_bot_admin(interaction, action="membersihkan slash command dobel"):
        return
    await interaction.response.defer(ephemeral=True)
    deleted: list[str] = []
    failed: list[str] = []
    protected = {"slash"}

    try:
        fetched = await _safe_fetch_app_commands(scope, interaction.guild)

        # 1) Hapus duplikat di scope yang sama, simpan command pertama.
        for label, guild_obj, commands in fetched:
            seen: set[str] = set()
            for cmd in commands:
                key = cmd.name.lower()
                if key in protected:
                    seen.add(key)
                    continue
                if key in seen:
                    try:
                        await _delete_app_command(cmd, guild_obj)
                        deleted.append(f"/{cmd.name} duplikat ({label})")
                    except Exception as exc:  # noqa: BLE001
                        failed.append(f"/{cmd.name} ({label}) -> {type(exc).__name__}: {exc}")
                else:
                    seen.add(key)

        # 2) Kalau ada command dengan nama sama di server dan global, hapus salah satu sesuai keep.
        if scope == "both" and interaction.guild is not None:
            guild_commands = next((cmds for label, _g, cmds in fetched if label == "server"), [])
            global_commands = next((cmds for label, _g, cmds in fetched if label == "global"), [])
            guild_by_name = {cmd.name.lower(): cmd for cmd in guild_commands}
            global_by_name = {cmd.name.lower(): cmd for cmd in global_commands}
            overlap = sorted((set(guild_by_name) & set(global_by_name)) - protected)
            for name_key in overlap:
                cmd = guild_by_name[name_key] if keep == "global" else global_by_name[name_key]
                guild_obj = interaction.guild if keep == "global" else None
                label = "server" if keep == "global" else "global"
                try:
                    await _delete_app_command(cmd, guild_obj)
                    deleted.append(f"/{cmd.name} dobel lintas-scope ({label})")
                except Exception as exc:  # noqa: BLE001
                    failed.append(f"/{cmd.name} ({label}) -> {type(exc).__name__}: {exc}")

        msg = "✅ Slash command dibersihkan."
        if deleted:
            msg += "\n\nTerhapus:\n" + "\n".join(f"- {x}" for x in deleted[:25])
            if len(deleted) > 25:
                msg += f"\n...dan {len(deleted)-25} lagi"
        else:
            msg += " Tidak ada duplikat yang ditemukan."
        if failed:
            msg += "\n\n⚠️ Gagal:\n" + "\n".join(f"- {x}" for x in failed[:5])
        await interaction.followup.send(msg[:1900], ephemeral=True)
    except Exception as exc:  # noqa: BLE001
        if _is_discord_global_429(exc):
            wait = _mark_discord_api_pause(exc)
            await interaction.followup.send(f"⚠️ Discord API lagi rate limit. Coba lagi sekitar {wait//60} menit lagi.", ephemeral=True)
            return
        await interaction.followup.send(f"⚠️ Gagal dedupe: `{type(exc).__name__}: {exc}`", ephemeral=True)


@slash_group.command(name="prune", description="Hapus banyak command, simpan hanya nama command yang ditulis.")
@app_commands.describe(
    keep_names="Command yang disimpan, pisahkan koma. Contoh: help,slash,kirimdm,modstatus",
    scope="Lokasi command yang mau dibersihkan",
)
@app_commands.choices(scope=SLASH_SCOPE_CHOICES)
async def slash_prune(interaction: discord.Interaction, keep_names: str = "help,slash", scope: str = "guild") -> None:
    if not await require_bot_admin(interaction, action="prune slash command"):
        return
    await interaction.response.defer(ephemeral=True)
    keep = {x.strip().lower().lstrip("/") for x in keep_names.split(",") if x.strip()}
    keep.add("slash")  # alat bersih-bersih jangan ikut hilang
    deleted: list[str] = []
    failed: list[str] = []
    try:
        for label, guild_obj, commands in await _safe_fetch_app_commands(scope, interaction.guild):
            for cmd in commands:
                if cmd.name.lower() in keep:
                    continue
                try:
                    await _delete_app_command(cmd, guild_obj)
                    deleted.append(f"/{cmd.name} ({label})")
                except Exception as exc:  # noqa: BLE001
                    failed.append(f"/{cmd.name} ({label}) -> {type(exc).__name__}: {exc}")
        msg = f"✅ Prune selesai. Command yang disimpan: `{', '.join(sorted(keep))}`"
        if deleted:
            msg += "\n\nTerhapus:\n" + "\n".join(f"- {x}" for x in deleted[:25])
            if len(deleted) > 25:
                msg += f"\n...dan {len(deleted)-25} lagi"
        if failed:
            msg += "\n\n⚠️ Gagal:\n" + "\n".join(f"- {x}" for x in failed[:5])
        await interaction.followup.send(msg[:1900], ephemeral=True)
    except Exception as exc:  # noqa: BLE001
        if _is_discord_global_429(exc):
            wait = _mark_discord_api_pause(exc)
            await interaction.followup.send(f"⚠️ Discord API lagi rate limit. Coba lagi sekitar {wait//60} menit lagi.", ephemeral=True)
            return
        await interaction.followup.send(f"⚠️ Gagal prune: `{type(exc).__name__}: {exc}`", ephemeral=True)


bot.tree.add_command(slash_group)


@bot.tree.command(name="modstatus", description="Cek status auto-moderation guard bot.")
async def modstatus(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)
    if not await require_bot_admin(interaction, action="melihat status moderation"):
        return
    if not interaction.guild:
        await interaction.followup.send("Command ini hanya bisa dipakai di server.", ephemeral=True)
        return

    log_channel = get_mod_log_channel(interaction.guild)
    allowlist = ", ".join(settings.mod_allowed_link_domains) or "(kosong: link member biasa akan diblok)"
    bypass = ", ".join(settings.mod_bypass_role_names) or "(kosong)"
    text = (
        "🛡️ **Status Auto-Moderation**\n"
        f"Patch: **{BOT_PATCH_VERSION}**\n"
        f"Aktif: **{settings.moderation_enabled}**\n"
        f"AI check: **{settings.moderation_ai_enabled}**\n"
        f"Delete link member biasa: **{settings.moderation_delete_links}**\n"
        f"DM warning: **{settings.moderation_dm_warnings}**\n"
        f"Channel log: {log_channel.mention if log_channel else '**belum ketemu**'}\n"
        f"Allowlist domain: `{allowlist}`\n"
        f"Bypass roles: `{bypass}`\n"
        f"Spam limit: **{settings.mod_spam_message_limit} pesan/{settings.mod_spam_window_seconds} detik**\n"
        f"Duplicate limit: **{settings.mod_duplicate_message_limit}x pesan sama**\n"
        f"Max mention: **{settings.mod_max_mentions}**\n\n"
        "**Anti-Scam & Anti-Promosi 2026**\n"
        f"Mr. Beast scam detect: **{settings.mrbeast_scam_detection}**\n"
        f"Mr. Beast scam ban: **{settings.mrbeast_scam_ban}**\n"
        f"Celebrity scam detect: **{settings.celebrity_scam_detection}**\n"
        f"Robux scam detect: **Aktif**\n"
        f"Server promo block: **Aktif**"
    )
    await interaction.followup.send(text, ephemeral=True)


@bot.tree.command(name="status", description="Cek status bot voice AI.")
async def status(interaction: discord.Interaction) -> None:
    if not await require_bot_admin(interaction, action="melihat status voice bot"):
        return
    if not interaction.guild:
        await interaction.response.send_message("Status hanya tersedia di server.", ephemeral=True)
        return
    session = bot.sessions.get(interaction.guild.id)
    if not session or not session.voice_client or not session.voice_client.is_connected():
        await interaction.response.send_message("🔴 Aku belum connect ke voice.", ephemeral=True)
        return

    vc = session.voice_client
    await interaction.response.send_message(
        f"🟢 Connected: **{vc.channel.name}**\n"
        f"🎧 Listening: **{session.is_listening()}**\n"
        f"🔊 Playing: **{session.is_playing()}**\n"
        f"🗣️ Speech Enabled: **{session.speech_enabled}**",
        ephemeral=True,
    )


def build_help_menu_text(*, is_admin: bool) -> str:
    public_commands = (
        "**🛡️ GLADIATOR — Assistant Guard Server AI**\n"
        "Aku bisa bantu jawab pertanyaan, baca info server, guard/moderasi, dan voice command.\n\n"
        "**Command umum**\n"
        "`/help` — tampilkan menu bantuan ini.\n"
        "`/talk prompt:` — tanya AI lewat teks.\n"
        "`/serverask question:` — tanya info server + RAG.\n"
        "`/ragask question:` — tanya dari data channel yang sudah discan.\n\n"
        "**Voice**\n"
        "`/join` — bot masuk voice.\n"
        "`/say text:` — bot membacakan teks dengan TTS lokal.\n"
        "`/saypremium text:` — bot membacakan teks dengan Gemini TTS premium.\n"
        "`/leave` — bot keluar voice.\n\n"
        "**Server info**\n"
        "`/serverinfo` — lihat info server.\n"
        "`/admins` — lihat owner/admin/role tinggi.\n"
        "`/events` — lihat event/acara server.\n\n"
        "Kamu juga bisa mention aku langsung, contoh: `@GLADIATOR nama server ini apa?`"
    )
    if not is_admin:
        return public_commands + (
            "\n\n**Catatan:** beberapa command kontrol hanya bisa dipakai Owner/Admin/role tinggi."
        )

    admin_commands = (
        "\n\n**Command Owner/Admin/role tinggi**\n"
        "`/ragscan channel limit:` — scan isi channel untuk RAG.\n"
        "`/ragclear` — hapus database RAG lokal.\n"
        "`/ragstatus` — cek jumlah data RAG.\n"
        "`/modstatus` — cek status auto-moderation & anti-scam.\n"
        "`/mrbeastblock user_id:` — kirim peringatan scam Mr. Beast ke DM.\n"
        "`/scamwarn user_id:` — kirim peringatan scam umum ke DM.\n"
        "`/sendto user_id:` — [OWNER] kirim DM + file, bisa AI compose pesan.\n"
        "`/resetvoice` — reset voice kalau stuck.\n"
        "`/stopvoice`, `/stoplisten`, `/stopspeak` — kontrol voice/audio.\n"
        "`/status` — cek status voice bot."
    )
    return public_commands + admin_commands


@bot.tree.command(name="help", description="Lihat menu bantuan GLADIATOR.")
async def help_command(interaction: discord.Interaction) -> None:
    is_admin = can_control_bot_member(interaction.user)
    await interaction.response.send_message(build_help_menu_text(is_admin=is_admin), ephemeral=True)


@bot.tree.command(name="aihelp", description="Lihat daftar command bot voice AI.")
async def aihelp(interaction: discord.Interaction) -> None:
    # Backward-compatible alias. /help adalah command global utama yang muncul di profil bot.
    is_admin = can_control_bot_member(interaction.user)
    await interaction.response.send_message(build_help_menu_text(is_admin=is_admin), ephemeral=True)


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return

    if await handle_auto_moderation(message):
        return

    should_answer = isinstance(message.channel, discord.DMChannel)
    if bot.user and bot.user in message.mentions:
        should_answer = True

    # Discord reply mode: kalau user reply ke pesan bot, bot juga menjawab walaupun tidak mention langsung.
    if not should_answer and message.reference and bot.user:
        resolved = message.reference.resolved
        if isinstance(resolved, discord.Message) and resolved.author.id == bot.user.id:
            should_answer = True

    if should_answer:
        if message.guild and not settings.public_ai_commands and not can_control_bot_member(message.author):
            await message.reply(
                "⚠️ AI public sedang dimatikan. Hanya Owner/Admin/role tinggi yang bisa memakai bot.",
                mention_author=False,
            )
            await bot.process_commands(message)
            return

        wait = cooldown_left(message.author.id)
        if wait:
            await message.reply(f"Tunggu {wait} detik dulu ya biar Gemini tidak kena limit.", mention_author=False)
            await bot.process_commands(message)
            return

        prompt = message.content
        if bot.user:
            prompt = prompt.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "").strip()
        if not prompt:
            prompt = "Halo"

        guild_part = message.guild.id if message.guild else "dm"
        conversation_id = f"mention:{guild_part}:{message.author.id}"
        try:
            async with message.channel.typing():
                if looks_like_identity_question(prompt):
                    answer = identity_answer()
                elif message.guild and looks_like_server_question(prompt):
                    answer = await answer_with_server_context(
                        conversation_id=conversation_id,
                        question=prompt,
                        guild=message.guild,
                        viewer=message.author,
                        speaker_name=getattr(message.author, "display_name", str(message.author)),
                    )
                else:
                    answer = await ai.chat(
                        conversation_id,
                        prompt,
                        speaker_name=getattr(message.author, "display_name", str(message.author)),
                    )
            await message.reply(answer[:1900], mention_author=False)
        except Exception as exc:  # noqa: BLE001
            await message.reply(friendly_ai_error(exc), mention_author=False)

    await bot.process_commands(message)


@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
    original = getattr(error, "original", error)

    if _is_discord_global_429(original) or _discord_api_paused():
        wait = _mark_discord_api_pause(original) if _is_discord_global_429(original) else _discord_pause_remaining()
        logging.warning(
            "Command %s kena Discord global 429. Tidak kirim error response supaya tidak crash-loop. Pause %ss.",
            getattr(getattr(interaction, "command", None), "name", "unknown"),
            wait,
        )
        return

    message = f"⚠️ Error: `{type(error).__name__}: {error}`"
    try:
        if interaction.response.is_done():
            await interaction.followup.send(message[:1900], ephemeral=True)
        else:
            await interaction.response.send_message(message[:1900], ephemeral=True)
    except discord.HTTPException as exc:
        if _is_discord_global_429(exc):
            wait = _mark_discord_api_pause(exc)
            logging.warning("Discord global 429 saat mengirim error command. Di-suppress. Pause %ss.", wait)
            return
        logging.warning("Gagal mengirim error command ke Discord HTTP %s: %s", getattr(exc, "status", "unknown"), exc)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Gagal mengirim error command: %s: %s", type(exc).__name__, exc)


def _looks_like_discord_global_login_rate_limit(exc: BaseException) -> bool:
    return _is_discord_global_429(exc)


def run_bot_safely() -> None:
    # Kalau Discord menolak login karena global rate limit, jangan crash-loop.
    # Crash-loop akan membuat Pterodactyl/Heavencloud restart terus dan limit makin lama.
    wait_seconds = int(os.getenv("DISCORD_LOGIN_RETRY_SECONDS", "900"))
    while True:
        try:
            bot.run(settings.discord_token)
            return
        except Exception as exc:  # noqa: BLE001
            if _looks_like_discord_global_login_rate_limit(exc):
                logging.error(
                    "Discord API global login rate limit / 429. Bot akan diam %s detik, "
                    "bukan restart spam. Jangan tekan Start/Restart berulang.",
                    wait_seconds,
                )
                time.sleep(wait_seconds)
                # Coba login lagi dari proses yang sama. Jangan crash ke panel, karena auto-restart
                # berulang justru memperparah global rate limit Discord.
                continue
            raise


if __name__ == "__main__":
    run_bot_safely()
