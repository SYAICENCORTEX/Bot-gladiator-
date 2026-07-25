# Discord Voice AI Bot Python

Bot Discord Python yang bisa:

- masuk voice channel dengan slash command `/join`
- mendengar suara user di voice channel
- mengubah suara ke teks / speech-to-text
- menjawab dengan AI
- membacakan jawaban ke voice channel dengan suara cowok
- chat via slash command `/talk`
- chat langsung lewat mention bot atau DM

> Penting: bot ini mendengar voice channel. Pakai hanya di server sendiri/teman yang setuju, dan beri tahu orang kalau bot sedang mendengarkan.

---

## 1. Reset token dulu

Kalau token bot pernah dikirim ke chat, token itu harus dianggap bocor.

Buka:

`Discord Developer Portal -> Applications -> pilih bot -> Bot -> Reset Token`

Pakai token baru di file `.env`. Jangan taruh token di kode dan jangan upload `.env` ke GitHub.

---

## 2. Syarat install

Install dulu:

1. Python 3.11 atau 3.12 direkomendasikan
2. FFmpeg
3. Bot Discord dengan permissions:
   - Send Messages
   - Use Slash Commands
   - Connect
   - Speak
   - Use Voice Activity
4. Aktifkan intent di Discord Developer Portal:
   - Message Content Intent
   - Server Members Intent tidak wajib
   - Presence Intent tidak wajib

### Install FFmpeg

Windows:

- Install dari package manager seperti Winget:

```bash
winget install Gyan.FFmpeg
```

atau install manual lalu pastikan `ffmpeg` bisa dipanggil dari terminal.

Linux/Ubuntu:

```bash
sudo apt update
sudo apt install ffmpeg
```

macOS:

```bash
brew install ffmpeg
```

---

## 3. Setup project

Masuk folder project lalu buat virtual environment.

Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Buat file `.env`

Copy `.env.example` menjadi `.env`.

Windows:

```bat
copy .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

Isi bagian ini:

```env
DISCORD_TOKEN=TOKEN_BOT_BARU_KAMU
OPENAI_API_KEY=OPENAI_API_KEY_KAMU
```

Opsional, isi `DISCORD_GUILD_ID` agar slash command cepat muncul di server testing.

```env
DISCORD_GUILD_ID=ID_SERVER_KAMU
```

Untuk mendapatkan server ID: aktifkan Developer Mode di Discord, klik kanan server, lalu Copy Server ID.

---

## 5. Jalankan bot

```bash
python main.py
```

Atau Windows bisa double click / run:

```bat
run.bat
```

---

## 6. Command yang tersedia

- `/join` — bot masuk voice channel kamu dan mulai mendengarkan.
- `/talk prompt:` — tanya lewat teks, bot jawab di chat dan dibacakan kalau bot sedang di voice.
- `/say text:` — bot membacakan teks ke voice channel dengan suara cowok.
- `/listen` — nyalakan mode mendengar lagi.
- `/stopvoice` — stop mendengar dan stop audio yang sedang diputar.
- `/leave` — keluar dari voice channel.
- `/status` — cek bot sedang connect/listening/playing atau tidak.
- `/aihelp` — lihat bantuan command.

Chat biasa:

- Mention bot di channel, contoh: `@Athena halo jelasin Python dong`
- DM bot langsung.

---

## 7. Cara pakai voice ngobrol

1. Kamu masuk voice channel.
2. Jalankan `/join` di text channel.
3. Bot masuk voice dan mulai mendengarkan.
4. Kamu bicara.
5. Bot akan kirim transkrip ke text channel, lalu jawab di text dan voice.

Catatan: ini bukan telepon real-time 0 delay. Biasanya ada jeda karena alurnya: suara -> transkrip -> AI -> suara.

---

## 8. Ganti suara cowok

Di `.env`:

```env
OPENAI_TTS_VOICE=onyx
```

Pilihan yang bisa dicoba:

```env
OPENAI_TTS_VOICE=onyx
OPENAI_TTS_VOICE=echo
OPENAI_TTS_VOICE=ash
```

---

## 9. Troubleshooting

### Slash command tidak muncul

Isi `DISCORD_GUILD_ID` di `.env`, lalu restart bot. Guild command biasanya muncul lebih cepat daripada global command.

### Bot masuk voice tapi tidak bersuara

Pastikan FFmpeg terinstall dan command ini jalan:

```bash
ffmpeg -version
```

Pastikan bot punya permission `Speak` dan `Use Voice Activity`.

### Bot tidak merespons chat mention

Aktifkan `Message Content Intent` di Discord Developer Portal.

### Voice listening error / tiba-tiba tidak mendengar

Library penerima voice Discord untuk Python masih experimental. Coba jalankan `/listen` lagi atau `/leave` lalu `/join`.

### OpenAI error

Cek:

- `OPENAI_API_KEY` benar
- saldo/billing API tersedia
- model di `.env` tersedia untuk akunmu

---

## Struktur file

```text
discord_voice_ai_bot/
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
├── run.bat
├── run.sh
└── bot/
    ├── __init__.py
    ├── ai_client.py
    ├── audio_utils.py
    ├── config.py
    └── voice_session.py
```
