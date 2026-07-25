from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque
from urllib.parse import urlparse


URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+|\b(?:discord\.gg|dsc\.gg|bit\.ly|tinyurl\.com|t\.me|wa\.me)/\S+")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")


@dataclass(slots=True)
class ModerationResult:
    should_delete: bool
    category: str
    reason: str
    evidence: str = ""
    # action = "delete" atau "ban". Ban hanya dipakai untuk scam berat/crypto scam.
    action: str = "delete"


class SpamTracker:
    """Lightweight in-memory spam detector per guild + user."""

    def __init__(self) -> None:
        self._events: dict[tuple[int, int], Deque[tuple[float, str]]] = defaultdict(lambda: deque(maxlen=20))

    def check(
        self,
        *,
        guild_id: int,
        user_id: int,
        content: str,
        window_seconds: int,
        message_limit: int,
        duplicate_limit: int,
    ) -> ModerationResult | None:
        now = time.monotonic()
        key = (guild_id, user_id)
        normalized = normalize_text(content)
        events = self._events[key]
        events.append((now, normalized))

        while events and now - events[0][0] > window_seconds:
            events.popleft()

        if len(events) >= message_limit:
            return ModerationResult(
                should_delete=True,
                category="spam",
                reason=f"Mengirim terlalu banyak pesan dalam {window_seconds} detik.",
                evidence=f"{len(events)} pesan/{window_seconds} detik",
            )

        if normalized and sum(1 for _, item in events if item == normalized) >= duplicate_limit:
            return ModerationResult(
                should_delete=True,
                category="spam",
                reason="Mengirim pesan yang sama berulang kali.",
                evidence="duplikat berulang",
            )

        return None


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "[link]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:500]


def extract_urls(text: str) -> list[str]:
    return [match.group(0).rstrip(")].,!?;:'\"") for match in URL_RE.finditer(text or "")]


def domain_from_url(url: str) -> str:
    candidate = url.strip()
    if candidate.lower().startswith("www."):
        candidate = "https://" + candidate
    if not candidate.lower().startswith(("http://", "https://")):
        candidate = "https://" + candidate
    parsed = urlparse(candidate)
    domain = (parsed.netloc or parsed.path.split("/", 1)[0]).lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def domain_is_allowed(domain: str, allowlist: tuple[str, ...]) -> bool:
    if not allowlist:
        return False
    domain = domain.lower().strip()
    for allowed in allowlist:
        allowed = allowed.lower().strip()
        if not allowed:
            continue
        if domain == allowed or domain.endswith("." + allowed):
            return True
    return False


def has_image_attachment(attachments: object) -> bool:
    for attachment in attachments or []:
        content_type = (getattr(attachment, "content_type", "") or "").lower()
        filename = (getattr(attachment, "filename", "") or "").lower()
        if content_type.startswith("image/") or filename.endswith(IMAGE_EXTS):
            return True
    return False


# Keep this list practical and non-exhaustive. Gemini can refine suspicious cases when enabled.
ADULT_HINTS = (
    "18+", "nsfw", "adult", "bokep", "porno", "porn", "hentai", "sange",
    "telanjang", "bugil", "mesum", "xnxx", "xvideos", "onlyfans",
)

PROMO_HINTS = (
    "join server", "join discord", "discord.gg", "dsc.gg", "invite", "promosi",
    "promote", "promo", "jual", "jualan", "diskon", "gratis nitro", "free nitro",
    "giveaway", "subscribe", "follow", "wa.me", "t.me/", "shortlink", "bit.ly",
)

SCAM_HINTS = (
    "airdrop", "claim reward", "claim your reward", "verify wallet", "connect wallet",
    "steam gift", "nitro free", "free nitro", "free robux", "klik link", "ambil hadiah",
    "hadiah gratis", "login hadiah", "bonus gratis", "free money", "withdraw instantly",
    "withdraw bonus", "bonus instantly", "register and withdraw", "withdrawal success",
)

# Pola scam crypto/casino seperti contoh: caption pendek "omg 💰" + gambar bukti withdraw,
# profile palsu public figure, bonus $2500, crypto casino, USDT/TRX, promo code, dll.
CRYPTO_SCAM_HINTS = (
    "crypto casino", "casino crypto", "bitcoin casino", "btc casino", "ponzobet", "spin games",
    "promo code", "special promo", "special code", "bonus $", "$2500", "2,500", "2500 usdt",
    "+2500 usdt", "usdt", "trx", "bitcoin", "btc", "ethereum", "eth", "wallet",
    "block explorer", "network fee", "withdrawal success", "withdraw success", "select crypto to withdraw",
    "claim bonus", "withdraw bonus", "receive your", "registers", "register and", "your balance $0.00",
)

MONEY_EMOJI_HINTS = ("💰", "🤑", "💸", "💵", "💲")
SHORT_SCAM_CAPTIONS = (
    "omg", "omg!", "wow", "wow!", "crazy", "wth", "no way", "lihat ini", "cek ini",
)


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in words)


def looks_like_crypto_scam_caption(content: str, *, has_image: bool) -> bool:
    """Detect very common scam pattern: image attachment + short money caption."""
    text = (content or "").strip().lower()
    compact = re.sub(r"\s+", " ", text)
    if not has_image:
        return False
    if not compact:
        return False

    has_money_emoji = any(emoji in text for emoji in MONEY_EMOJI_HINTS)
    has_money_word = any(token in compact for token in ("money", "uang", "duit", "bonus", "profit", "wd", "withdraw"))
    short_caption = len(compact) <= 80 and any(caption in compact for caption in SHORT_SCAM_CAPTIONS)
    crypto_word = _contains_any(compact, CRYPTO_SCAM_HINTS)

    return (has_money_emoji and short_caption) or (has_image and crypto_word and (has_money_emoji or has_money_word))


def rule_based_check(
    content: str,
    *,
    mention_count: int = 0,
    max_mentions: int = 6,
    allowed_domains: tuple[str, ...] = (),
    delete_links: bool = True,
    has_attachment: bool = False,
    has_image: bool = False,
) -> ModerationResult | None:
    text = (content or "").strip()
    lowered = text.lower()

    if mention_count >= max_mentions:
        return ModerationResult(
            should_delete=True,
            category="spam",
            reason=f"Terlalu banyak mention dalam satu pesan ({mention_count} mention).",
            evidence=f"{mention_count} mention",
        )

    # Scam gambar seperti contoh: upload gambar + caption "omg 💰" / caption crypto bonus.
    if looks_like_crypto_scam_caption(text, has_image=has_image):
        return ModerationResult(
            should_delete=True,
            category="crypto-scam-image",
            reason="Terdeteksi pola scam crypto/giveaway palsu: gambar promosi/withdraw + caption uang singkat.",
            evidence=(text or "image + money caption")[:250],
            action="ban",
        )

    if _contains_any(lowered, CRYPTO_SCAM_HINTS) and ("$" in lowered or "bonus" in lowered or "withdraw" in lowered or "promo" in lowered):
        return ModerationResult(
            should_delete=True,
            category="crypto-scam",
            reason="Pesan terdeteksi sebagai promosi/scam crypto, casino, bonus palsu, atau withdrawal palsu.",
            evidence="indikasi crypto/casino/bonus scam",
            action="ban",
        )

    urls = extract_urls(text)
    if urls and delete_links:
        blocked: list[str] = []
        for url in urls:
            domain = domain_from_url(url)
            if not domain_is_allowed(domain, allowed_domains):
                blocked.append(domain)
        if blocked:
            category = "link"
            reason = "Membagikan link yang tidak diizinkan untuk member biasa."
            action = "delete"
            evidence = ", ".join(sorted(set(blocked)))[:300]
            if _contains_any(lowered, CRYPTO_SCAM_HINTS) or _contains_any(lowered, SCAM_HINTS):
                category = "crypto-scam-link"
                reason = "Membagikan link yang terindikasi scam/crypto/casino/bonus palsu."
                action = "ban"
            return ModerationResult(
                should_delete=True,
                category=category,
                reason=reason,
                evidence=evidence,
                action=action,
            )

    if any(word in lowered for word in ADULT_HINTS):
        return ModerationResult(
            should_delete=True,
            category="adult-content",
            reason="Pesan terdeteksi mengandung konten 18+/NSFW yang tidak cocok untuk server.",
            evidence="kata/indikasi 18+",
        )

    if any(word in lowered for word in SCAM_HINTS):
        return ModerationResult(
            should_delete=True,
            category="scam/promo",
            reason="Pesan terdeteksi seperti promosi/scam atau ajakan klik hadiah/link mencurigakan.",
            evidence="indikasi scam/promo",
        )

    if urls and any(word in lowered for word in PROMO_HINTS):
        return ModerationResult(
            should_delete=True,
            category="promotion",
            reason="Pesan berisi link dan pola promosi/undangan yang tidak diizinkan.",
            evidence="link + promosi",
        )

    # Attachment gambar tanpa teks tidak langsung dihukum rule-based supaya tidak false positive.
    # Kalau MODERATION_IMAGE_AI_ENABLED=true, Gemini Vision akan cek gambar di main.py.
    return None


def should_ai_review(content: str, *, has_attachment: bool = False, has_image: bool = False) -> bool:
    lowered = (content or "").lower()
    if extract_urls(content):
        return True
    if has_image or has_attachment:
        return True
    return any(word in lowered for word in (*ADULT_HINTS, *PROMO_HINTS, *SCAM_HINTS, *CRYPTO_SCAM_HINTS))
