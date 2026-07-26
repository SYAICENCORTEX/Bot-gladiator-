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

# =========================
# ROBUX SCAM DETECTION
# =========================
# Scam Robux sangat marak di server Discord: akun diretas menyebarkan
# "free robux", "robux generator", "robux giveaway", dll untuk menipu
# anggota server, terutama anak-anak dan remaja.
ROBUX_SCAM_HINTS = (
    "free robux", "robux gratis", "robux free", "robux generator",
    "robux hack", "robux cheat", "robux glitch", "robux exploit",
    "robux giveaway", "giveaway robux", "robux claim", "claim robux",
    "robux hadiah", "hadiah robux", "bonus robux", "robux bonus",
    "robux tanpa bayar", "robux tanpa membayar", "robux mudah",
    "dapatkan robux", "cara dapat robux gratis", "robux ilegal",
    "robux promo", "promo robux", "robux code", "robux gift card",
    "robux 100000", "robux 10000", "robux 5000", "robux 1000",
    "free robux 2026", "robux 2026", "roblox robux", "roblox free",
    "generate robux", "robux gen", "robux tool", "robux bot",
    "robux tidak terbatas", "unlimited robux", "infinite robux",
    "robux login", "login dapat robux", "robux verify", "verify robux",
)

# Keyword link/situs yang sering dipakai scam Robux
ROBUX_SCAM_DOMAINS = (
    "robux", "rblx", "roblox", "free-robux", "robux-generator",
    "robuxhack", "robuxcheat", "robuxclub", "robuxking",
)

# =========================
# DISCORD SERVER PROMOTION DETECTION
# =========================
# Promosi server Discord lain yang tidak diizinkan.
# Biasanya: invite link + ajakan join server lain.
SERVER_PROMO_HINTS = (
    "join server", "join discord", "join our server", "join my server",
    "join our discord", "join my discord", "join us at",
    "server discord", "discord server", "server baru", "new server",
    "server keren", "server recommended", "server terbaik",
    "ayo join", "yuk join", "join yuk", "mari join",
    "discord.gg/", "dsc.gg/", "discordservers.com",
    "server partner", "partner server", "kerjasama server",
    "promosi server", "promosi discord", "promote server",
    "gabung server", "gabung discord", "masuk server",
    "server komunitas", "komunitas discord", "komunitas baru",
    "top global", "server indo", "server malaysia",
    "boost server", "leveling server", "level up server",
    "server gaming", "gaming server", "game server",
    "server anime", "anime server", "server rp", "roleplay server",
    "giveaway server", "server giveaway", "nitro giveaway server",
)

# Kombinasi: invite link + ajakan join (paling khas promosi server)
SERVER_PROMO_LINK_PATTERNS = (
    "discord.gg/", "dsc.gg/", "discord.com/invite/",
    "discordapp.com/invite/", "discordservers.com/server/",
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

# =========================
# MR. BEAST SCAM DETECTION
# =========================
# Pola scam yang mengatasnamakan Mr. Beast (Jimmy Donaldson).
# Scam ini sangat marak di Discord 2026: akun diretas menyebarkan gambar palsu
# klaim giveaway $2500, bonus crypto, casino, dll dengan foto Mr. Beast palsu.
# Pelaku biasanya upload 4 gambar berisi:
#   1) Foto Mr. Beast palsu / screenshot endorsemen palsu
#   2) Screenshot fake withdrawal success ($2500-$3200)
#   3) Screenshot fake dashboard/saldo crypto
#   4) Fake QR/promo code untuk deposit

MRBEAST_NAME_VARIANTS = (
    "mrbeast", "mr beast", "mr.beast", "mr_ beast", "mr _beast",
    "jimmy donaldson", "mrbeast", "mrbe ast", "mrbe ast",
    "tuan beast", "tuanbeast", "mrbeastt", "mr bea$t",
)

# Kata kunci yang sering muncul di scam Mr. Beast
MRBEAST_SCAM_KEYWORDS = (
    "mrbeast giveaway", "mr beast giveaway", "mr.beast giveaway",
    "mrbeast crypto", "mr beast crypto", "mr.beast crypto",
    "mrbeast bonus", "mr beast bonus", "mr.beast bonus",
    "mrbeast $2500", "mr beast $2500", "mr.beast $2500",
    "mrbeast 2500", "mr beast 2500", "mr.beast 2500",
    "mrbeast promo", "mr beast promo", "mr.beast promo",
    "mrbeast casino", "mr beast casino", "mr.beast casino",
    "mrbeast withdrawal", "mr beast withdrawal", "mr.beast withdrawal",
    "giveaway mrbeast", "giveaway mr beast",
    "by mrbeast", "by mr beast",
    "mrbeast partnership", "mr beast partnership",
    "mrbeast million", "mr beast million",
    "celebrating 300 million", "celebrating 100 million", "celebrating 200 million",
    "celebrating 400 million", "celebrating 500 million",
    "beast gaming partner", "beast gaming giveaway",
)

# Kombinasi yang sangat mencurigakan: nama Mr. Beast + uang/giveaway
MRBEAST_MONEY_PATTERNS = (
    "$2500", "$2,500", "2500 usdt", "2500 dollar", "2500 usd",
    "$3200", "$3,200", "3200 usdt",
    "$5000", "$5,000", "5000 usdt",
    "$10000", "$10,000", "10000 usdt",
    "free money", "free cash", "free bonus",
    "claim your prize", "claim reward", "you won", "you win",
    "you are selected", "you've been selected", "congratulations you",
    "airdrop", "reward", "prize",
)

# Pola caption sangat pendek (<=40 chars) yang khas scam gambar Mr. Beast
MRBEAST_SHORT_CAPTIONS = (
    "omg", "omg!", "wow", "wow!", "no way", "wtf", "lol",
    "lihat ini", "cekidot", "cek", "mantap", "gas",
    "free money", "free $$$", "💰", "🤑", "💸",
)

# Nama figur publik lain yang juga sering dipalsukan dalam scam serupa
CELEBRITY_SCAM_VARIANTS = (
    "elon musk", "elonmusk", "elon musk",
    "andrew tate", "andrewtate", "cobratate",
    "kai cenat", "kaicenat",
    "donald trump", "donaldtrump", "realDonaldTrump",
    "sophie rain", "sophierain",
    "cristiano ronaldo", "cristianoronaldo",
)

ALL_CELEBRITY_NAMES = MRBEAST_NAME_VARIANTS + CELEBRITY_SCAM_VARIANTS


def _word_boundary_search(text: str, keyword: str) -> bool:
    """Cek keyword dengan word boundary yang lebih longgar untuk nama selebriti."""
    lowered = text.lower()
    if keyword in lowered:
        return True
    # Cek tanpa spasi
    no_space = keyword.replace(" ", "")
    if no_space and no_space in lowered:
        return True
    return False


def has_mrbeast_reference(text: str) -> bool:
    """Cek apakah teks mengandung referensi Mr. Beast."""
    if not text:
        return False
    lowered = text.lower()
    for variant in MRBEAST_NAME_VARIANTS:
        if variant in lowered:
            return True
    return False


def has_celebrity_reference(text: str) -> bool:
    """Cek apakah teks mengandung referensi selebriti (Mr. Beast dkk)."""
    if not text:
        return False
    lowered = text.lower()
    for name in ALL_CELEBRITY_NAMES:
        if name in lowered:
            return True
    return False


def looks_like_mrbeast_scam(content: str, *, has_image: bool = False, image_count: int = 0) -> bool:
    """Deteksi scam yang mengatasnamakan Mr. Beast (paling marak 2026).

    Pola yang dideteksi:
    1. Caption pendek (<=60 char) + nama Mr. Beast + emoji uang + gambar
    2. 4 gambar berurutan (image_count >= 3) + caption pendek
    3. Teks yang menyebut Mr. Beast + jumlah uang ($2500, 2500 usdt, dll)
    4. Gambar + caption yang mengandung kata-kata scam khas Mr. Beast
    """
    text = (content or "").strip().lower()
    compact = re.sub(r"\\s+", " ", text)
    has_mrbeast = has_mrbeast_reference(text)

    if not has_mrbeast:
        return False

    # Deteksi 1: Mr. Beast + jumlah uang ($2500, $3200, 2500 usdt)
    if any(amount in compact for amount in MRBEAST_MONEY_PATTERNS):
        return True

    # Deteksi 2: Mr. Beast + crypto scam keywords
    if any(keyword in compact for keyword in MRBEAST_SCAM_KEYWORDS):
        return True

    # Deteksi 3: Mr. Beast + gambar + caption pendek
    if has_image and len(compact) <= 80:
        has_money_emoji = any(emoji in text for emoji in MONEY_EMOJI_HINTS)
        has_short_caption = any(caption in compact for caption in MRBEAST_SHORT_CAPTIONS)
        has_money_word = any(token in compact for token in ("money", "uang", "bonus", "profit", "wd", "withdraw", "$"))
        if has_money_emoji or has_short_caption or has_money_word:
            return True

    # Deteksi 4: 4 gambar berurutan (khas scam Mr. Beast) + caption pendek apapun
    if image_count >= 3 and len(compact) <= 60:
        return True

    # Deteksi 5: Mr. Beast + scam/promo/crypto keywords
    if any(word in compact for word in (*CRYPTO_SCAM_HINTS, *SCAM_HINTS, *PROMO_HINTS)):
        return True

    return False


def looks_like_celebrity_scam_image(content: str, *, has_image: bool = False, image_count: int = 0) -> bool:
    """Deteksi scam gambar selebriti (Mr. Beast, Elon Musk, Andrew Tate, dll).

    Sama seperti Mr. Beast scam tapi untuk semua figur publik.
    """
    text = (content or "").strip().lower()
    compact = re.sub(r"\\s+", " ", text)
    has_celeb = has_celebrity_reference(text)

    if not has_celeb:
        return False

    # Deteksi selebriti + uang
    if any(amount in compact for amount in MRBEAST_MONEY_PATTERNS):
        return True

    # Deteksi selebriti + crypto scam
    if any(word in compact for word in (*CRYPTO_SCAM_HINTS, *MRBEAST_SCAM_KEYWORDS)):
        return True

    # Deteksi selebriti + gambar + caption pendek + emoji uang
    if has_image and len(compact) <= 80:
        has_money_emoji = any(emoji in text for emoji in MONEY_EMOJI_HINTS)
        has_money_word = any(token in compact for token in ("money", "bonus", "$", "profit", "wd"))
        if has_money_emoji or has_money_word:
            return True

    # 4 gambar + nama selebriti
    if image_count >= 3 and len(compact) <= 60:
        return True

    return False


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
    image_count: int = 0,
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

    # ================================================================
    # PRIORITAS TERTINGGI: Scam Mr. Beast (paling marak 2026)
    # ================================================================
    # Deteksi scam yang mengatasnamakan Mr. Beast.
    # Scam ini biasanya: gambar + caption pendek + nama Mr. Beast + jumlah uang.
    if has_mrbeast_reference(text):
        mrbeast_result = _check_mrbeast_scam(text, has_image=has_image, image_count=image_count)
        if mrbeast_result is not None:
            return mrbeast_result

    # Deteksi scam selebriti lain (Elon Musk, Andrew Tate, dll)
    if has_celebrity_reference(text) and not has_mrbeast_reference(text):
        celeb_result = _check_celebrity_scam(text, has_image=has_image, image_count=image_count)
        if celeb_result is not None:
            return celeb_result

    # ================================================================
    # Scam gambar crypto (caption pendek + emoji uang + gambar)
    # ================================================================
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

    # ================================================================
    # ROBUX SCAM DETECTION (scam free robux / robux generator)
    # ================================================================
    if _contains_any(lowered, ROBUX_SCAM_HINTS):
        return ModerationResult(
            should_delete=True,
            category="robux-scam",
            reason="Pesan terdeteksi sebagai scam Robux: free robux, robux generator, atau robux giveaway palsu.",
            evidence="indikasi scam robux",
            action="delete",
        )

    # Cek juga domain situs scam Robux
    if urls:
        for url in urls:
            domain = domain_from_url(url)
            if any(robux_domain in domain for robux_domain in ROBUX_SCAM_DOMAINS):
                return ModerationResult(
                    should_delete=True,
                    category="robux-scam-link",
                    reason="Link terdeteksi sebagai situs scam Robux (free robux / robux generator palsu).",
                    evidence=f"domain scam robux: {domain}",
                    action="delete",
                )

    # ================================================================
    # DISCORD SERVER PROMOTION DETECTION
    # ================================================================
    # Deteksi promosi server Discord lain (invite + ajakan join)
    has_server_promo_keyword = _contains_any(lowered, SERVER_PROMO_HINTS)
    has_invite_link = any(pattern in lowered for pattern in SERVER_PROMO_LINK_PATTERNS)

    if has_invite_link or (has_server_promo_keyword and urls):
        return ModerationResult(
            should_delete=True,
            category="server-promotion",
            reason="Promosi server Discord lain tidak diizinkan. Hapus link invite dan ajakan join server lain.",
            evidence="indikasi promosi server discord",
        )

    if has_server_promo_keyword and len(text.split()) >= 3:
        return ModerationResult(
            should_delete=True,
            category="server-promotion",
            reason="Promosi server Discord lain tidak diizinkan. Dilarang mengajak member ke server lain.",
            evidence="ajakan promosi server",
        )

    if any(word in lowered for word in ADULT_HINTS):
        return ModerationResult(
            should_delete=True,
            category="adult-content",
            reason="Pesan terdeteksi mengandung konten 18+/NSFW yang tidak cocok untuk server.",
            evidence="kata/indikasi 18+",
        )

    # ================================================================
    # SCAM / PROMO UMUM
    # ================================================================
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


def _check_mrbeast_scam(text: str, *, has_image: bool = False, image_count: int = 0) -> ModerationResult | None:
    """Cek spesifik scam Mr. Beast dan return ModerationResult jika terdeteksi."""
    lowered = text.lower()
    compact = re.sub(r"\\s+", " ", lowered)

    # Pola 1: Mr. Beast + jumlah uang ($2500, $3200, dll) — scam giveaway
    if any(amount in compact for amount in MRBEAST_MONEY_PATTERNS):
        return ModerationResult(
            should_delete=True,
            category="mrbeast-scam",
            reason="⚠️ SCAM MR. BEAST! Pesan palsu mengatasnamakan Mr. Beast dengan iming-iming uang/giveaway.",
            evidence=f"mrbeast + money pattern: {text[:200]}",
            action="ban",
        )

    # Pola 2: Mr. Beast + gambar + caption pendek + emoji uang
    if has_image and len(compact) <= 80:
        return ModerationResult(
            should_delete=True,
            category="mrbeast-scam-image",
            reason="⚠️ SCAM MR. BEAST! Gambar palsu mengatasnamakan Mr. Beast dengan caption singkat mencurigakan.",
            evidence=f"mrbeast image scam: {text[:200]}",
            action="ban",
        )

    # Pola 3: 4 gambar (khas scam Mr. Beast) + nama Mr. Beast
    if image_count >= 3:
        return ModerationResult(
            should_delete=True,
            category="mrbeast-scam-burst",
            reason="⚠️ SCAM MR. BEAST BURST! Deteksi 3+ gambar scam Mr. Beast dikirim beruntun.",
            evidence=f"mrbeast {image_count}-image burst: {text[:200]}",
            action="ban",
        )

    # Pola 4: Mr. Beast + scam keywords
    if any(keyword in compact for keyword in MRBEAST_SCAM_KEYWORDS):
        return ModerationResult(
            should_delete=True,
            category="mrbeast-scam",
            reason="⚠️ SCAM MR. BEAST! Pesan mengandung kata kunci scam yang mengatasnamakan Mr. Beast.",
            evidence=f"mrbeast scam keyword: {text[:200]}",
            action="ban",
        )

    # Pola 5: Mr. Beast + crypto scam hints
    if any(word in compact for word in CRYPTO_SCAM_HINTS):
        return ModerationResult(
            should_delete=True,
            category="mrbeast-crypto-scam",
            reason="⚠️ SCAM MR. BEAST + CRYPTO! Pesan scam Mr. Beast dengan indikasi crypto/casino palsu.",
            evidence=f"mrbeast + crypto scam: {text[:200]}",
            action="ban",
        )

    return None


def _check_celebrity_scam(text: str, *, has_image: bool = False, image_count: int = 0) -> ModerationResult | None:
    """Cek scam figur publik lain (Elon Musk, Andrew Tate, dll)."""
    lowered = text.lower()
    compact = re.sub(r"\\s+", " ", lowered)

    # Cari figur publik mana yang disebut
    celeb_found = []
    for name in CELEBRITY_SCAM_VARIANTS:
        if name in lowered:
            celeb_found.append(name)

    if not celeb_found:
        return None

    celeb_name = celeb_found[0]
    celeb_label = celeb_name.title()

    # Pola 1: Selebriti + uang
    if any(amount in compact for amount in MRBEAST_MONEY_PATTERNS):
        return ModerationResult(
            should_delete=True,
            category="celebrity-scam",
            reason=f"⚠️ SCAM! Pesan palsu mengatasnamakan {celeb_label} dengan iming-iming uang/giveaway.",
            evidence=f"{celeb_name} + money: {text[:200]}",
            action="ban",
        )

    # Pola 2: Selebriti + gambar + caption pendek
    if has_image and len(compact) <= 80:
        return ModerationResult(
            should_delete=True,
            category="celebrity-scam-image",
            reason=f"⚠️ SCAM! Gambar palsu mengatasnamakan {celeb_label}.",
            evidence=f"{celeb_name} image scam: {text[:200]}",
            action="ban",
        )

    # Pola 3: 4 gambar burst + nama selebriti
    if image_count >= 3:
        return ModerationResult(
            should_delete=True,
            category="celebrity-scam-burst",
            reason=f"⚠️ SCAM BURST! {image_count}+ gambar scam mengatasnamakan {celeb_label}.",
            evidence=f"{celeb_name} {image_count}-image burst: {text[:200]}",
            action="ban",
        )

    # Pola 4: Selebriti + crypto
    if any(word in compact for word in CRYPTO_SCAM_HINTS):
        return ModerationResult(
            should_delete=True,
            category="celebrity-crypto-scam",
            reason=f"⚠️ SCAM {celeb_label.upper()} + CRYPTO! Scam mengatasnamakan {celeb_label} dengan crypto palsu.",
            evidence=f"{celeb_name} + crypto: {text[:200]}",
            action="ban",
        )

    return None


def should_ai_review(content: str, *, has_attachment: bool = False, has_image: bool = False) -> bool:
    lowered = (content or "").lower()
    if extract_urls(content):
        return True
    if has_image or has_attachment:
        return True
    if has_mrbeast_reference(content) or has_celebrity_reference(content):
        return True
    if _contains_any(lowered, ROBUX_SCAM_HINTS) or _contains_any(lowered, SERVER_PROMO_HINTS):
        return True
    return any(word in lowered for word in (*ADULT_HINTS, *PROMO_HINTS, *SCAM_HINTS, *CRYPTO_SCAM_HINTS))
