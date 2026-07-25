from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import discord


STOPWORDS = {
    "yang", "dan", "atau", "di", "ke", "dari", "ini", "itu", "untuk", "dengan", "ada",
    "apa", "siapa", "kapan", "bagaimana", "kenapa", "mengapa", "dalam", "jadi", "kalau",
    "the", "and", "or", "to", "of", "in", "is", "are", "a", "an", "for", "on", "with",
}


@dataclass
class KnowledgeItem:
    guild_id: int
    channel_id: int
    channel_name: str
    message_id: int
    author_name: str
    content: str
    created_at: str
    jump_url: str


@dataclass
class ServerSnapshot:
    guild_id: int
    name: str
    owner: str
    member_count: int
    role_count: int
    admin_roles: list[str]
    admin_members_sample: list[str]
    admin_members_count: int
    text_channel_count: int
    voice_channel_count: int
    category_count: int
    forum_channel_count: int
    stage_channel_count: int
    scheduled_events: list[str]


def normalize_text(text: str) -> str:
    text = re.sub(r"<@!?\d+>", "@user", text)
    text = re.sub(r"<@&\d+>", "@role", text)
    text = re.sub(r"<#\d+>", "#channel", text)
    text = re.sub(r"https?://\S+", "[link]", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9_\u00C0-\u024F\u1E00-\u1EFF]+", text.lower())
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS]


def _score(query_tokens: Counter[str], item_tokens: Counter[str]) -> float:
    if not query_tokens or not item_tokens:
        return 0.0
    overlap = set(query_tokens) & set(item_tokens)
    if not overlap:
        return 0.0
    dot = sum(query_tokens[t] * item_tokens[t] for t in overlap)
    q_norm = math.sqrt(sum(v * v for v in query_tokens.values()))
    i_norm = math.sqrt(sum(v * v for v in item_tokens.values()))
    return dot / (q_norm * i_norm) if q_norm and i_norm else 0.0


class KnowledgeStore:
    def __init__(self, base_dir: str = "data/knowledge") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, guild_id: int) -> Path:
        return self.base_dir / f"guild_{guild_id}.json"

    def load(self, guild_id: int) -> list[KnowledgeItem]:
        path = self._path(guild_id)
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return [KnowledgeItem(**item) for item in raw if item.get("content")]
        except Exception:
            return []

    def save(self, guild_id: int, items: list[KnowledgeItem]) -> None:
        # Deduplicate by message_id while keeping latest scan versions.
        unique: dict[int, KnowledgeItem] = {item.message_id: item for item in items if item.content.strip()}
        ordered = sorted(unique.values(), key=lambda x: x.created_at)
        self._path(guild_id).write_text(
            json.dumps([asdict(item) for item in ordered], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_items(self, guild_id: int, new_items: Iterable[KnowledgeItem]) -> int:
        current = self.load(guild_id)
        before_ids = {item.message_id for item in current}
        merged = current + list(new_items)
        self.save(guild_id, merged)
        after_ids = {item.message_id for item in self.load(guild_id)}
        return len(after_ids - before_ids)

    def clear(self, guild_id: int) -> None:
        path = self._path(guild_id)
        if path.exists():
            path.unlink()

    def search(self, guild_id: int, query: str, *, top_k: int = 6) -> list[KnowledgeItem]:
        items = self.load(guild_id)
        if not items:
            return []
        query_counter = Counter(tokenize(query))
        scored: list[tuple[float, KnowledgeItem]] = []
        for item in items:
            haystack = f"{item.channel_name} {item.author_name} {item.content}"
            score = _score(query_counter, Counter(tokenize(haystack)))
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def stats(self, guild_id: int) -> dict[str, int]:
        items = self.load(guild_id)
        channels = {item.channel_id for item in items}
        return {"messages": len(items), "channels": len(channels)}


async def scan_text_channel(channel: discord.TextChannel | discord.Thread | discord.ForumChannel, *, limit: int) -> list[KnowledgeItem]:
    if isinstance(channel, discord.ForumChannel):
        raise TypeError("Forum channel belum bisa discan langsung. Scan thread/forum post satu per satu kalau dibutuhkan.")

    items: list[KnowledgeItem] = []
    async for message in channel.history(limit=limit, oldest_first=False):
        if message.author.bot:
            continue
        content = normalize_text(message.content or "")
        if not content:
            continue
        if len(content) < 3:
            continue
        items.append(
            KnowledgeItem(
                guild_id=message.guild.id if message.guild else 0,
                channel_id=channel.id,
                channel_name=getattr(channel, "name", str(channel.id)),
                message_id=message.id,
                author_name=getattr(message.author, "display_name", str(message.author)),
                content=content[:1800],
                created_at=message.created_at.astimezone(timezone.utc).isoformat(),
                jump_url=message.jump_url,
            )
        )
    return items


async def build_server_snapshot(guild: discord.Guild) -> ServerSnapshot:
    owner_name = "Tidak diketahui"
    owner = guild.owner
    if owner is None and guild.owner_id:
        try:
            owner = await guild.fetch_member(guild.owner_id)
        except Exception:
            owner = None
    if owner:
        owner_name = f"{owner.display_name} (@{owner.name})"

    admin_roles = [role.name for role in guild.roles if role.permissions.administrator and not role.is_default()]

    admin_members: list[str] = []
    # Try API fetch first. Requires Server Members Intent for full accuracy.
    try:
        async for member in guild.fetch_members(limit=None):
            if member.guild_permissions.administrator:
                admin_members.append(f"{member.display_name} (@{member.name})")
    except Exception:
        # Fallback to cache. May be incomplete if members intent is disabled.
        admin_members = [
            f"{member.display_name} (@{member.name})"
            for member in guild.members
            if member.guild_permissions.administrator
        ]

    events: list[str] = []
    try:
        fetched_events = await guild.fetch_scheduled_events()
    except Exception:
        fetched_events = getattr(guild, "scheduled_events", []) or []

    now = datetime.now(timezone.utc)
    for event in fetched_events:
        status = getattr(event, "status", None)
        start_time = getattr(event, "start_time", None)
        if start_time and start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        when = start_time.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if start_time else "waktu belum jelas"
        name = getattr(event, "name", "Untitled event")
        if start_time is None or start_time >= now:
            events.append(f"{name} — {when} — status: {status}")

    return ServerSnapshot(
        guild_id=guild.id,
        name=guild.name,
        owner=owner_name,
        member_count=guild.member_count or len(guild.members),
        role_count=len(guild.roles),
        admin_roles=admin_roles[:30],
        admin_members_sample=admin_members[:25],
        admin_members_count=len(admin_members),
        text_channel_count=len(guild.text_channels),
        voice_channel_count=len(guild.voice_channels),
        category_count=len(guild.categories),
        forum_channel_count=len(getattr(guild, "forums", []) or []),
        stage_channel_count=len(guild.stage_channels),
        scheduled_events=events[:20],
    )


def format_snapshot(snapshot: ServerSnapshot) -> str:
    events = "\n".join(f"- {event}" for event in snapshot.scheduled_events) or "- Tidak ada event terjadwal yang terbaca."
    admin_roles = ", ".join(snapshot.admin_roles) or "Tidak ada role Administrator yang terbaca."
    admins = "\n".join(f"- {name}" for name in snapshot.admin_members_sample) or "- Tidak ada admin yang terbaca."
    return (
        f"Server: {snapshot.name}\n"
        f"Owner: {snapshot.owner}\n"
        f"Member: {snapshot.member_count}\n"
        f"Role: {snapshot.role_count}\n"
        f"Channel teks: {snapshot.text_channel_count}\n"
        f"Channel voice: {snapshot.voice_channel_count}\n"
        f"Kategori: {snapshot.category_count}\n"
        f"Forum: {snapshot.forum_channel_count}\n"
        f"Stage: {snapshot.stage_channel_count}\n"
        f"Role admin: {admin_roles}\n"
        f"Admin terbaca: {snapshot.admin_members_count}\n"
        f"Sampel admin:\n{admins}\n"
        f"Event terjadwal:\n{events}"
    )


def format_rag_context(items: list[KnowledgeItem], *, max_chars: int = 4500) -> str:
    if not items:
        return "Tidak ada dokumen/pesan RAG yang relevan di database lokal."

    parts: list[str] = []
    total = 0
    for index, item in enumerate(items, start=1):
        part = (
            f"[Sumber {index}] #{item.channel_name} | {item.author_name} | {item.created_at}\n"
            f"URL: {item.jump_url}\n"
            f"Isi: {item.content}\n"
        )
        if total + len(part) > max_chars:
            break
        parts.append(part)
        total += len(part)
    return "\n".join(parts)
