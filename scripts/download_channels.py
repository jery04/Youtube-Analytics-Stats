#!/usr/bin/env python3
"""
Descarga TODOS los videos de canales específicos de YouTube.
=============================================================
Canales objetivo:
  • Curiosidades al descubierto  (@Curiosidadesaldescubierto)
  • Curiosidades IA              (@curiosidadesai-y1t)

Cada canal se guarda como CSV independiente en dataset/channels/.

Variables por video:
  id, title, description, publishedAt, channelId, channelTitle,
  tags, categoryId, duration, durationSeconds, viewCount,
  commentCount, definicion, idioma_audio

Uso:
    python scripts/download_channels.py
    python scripts/download_channels.py --api-key CLAVE
"""

import os
import sys
import re
import csv
import time
import json
import argparse
import requests

# ── Cargar .env ──────────────────────────────────────────────────────────────
_BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_BASEDIR, ".env"))
except ImportError:
    _env = os.path.join(_BASEDIR, ".env")
    if os.path.exists(_env):
        with open(_env, "r", encoding="utf-8") as _f:
            for _ln in _f:
                _ln = _ln.strip()
                if not _ln or _ln.startswith("#") or "=" not in _ln:
                    continue
                _k, _v = _ln.split("=", 1)
                _k, _v = _k.strip(), _v.strip().strip("\"'")
                if _k and _v and _k not in os.environ:
                    os.environ[_k] = _v

# ── Constantes ───────────────────────────────────────────────────────────────
CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
SEARCH_URL   = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_URL   = "https://www.googleapis.com/youtube/v3/videos"

# Canales objetivo: (nombre legible, handle de YouTube)
TARGET_CHANNELS = [
    ("Curiosidades al descubierto", "@Curiosidadesaldescubierto"),
    ("Curiosidades IA",             "@curiosidadesai-y1t"),
    ("¿Sabías Esto..?",             "@SabíasEsto_27"),
]

FIELDNAMES = [
    "id", "title", "description", "publishedAt",
    "channelId", "channelTitle", "tags", "categoryId",
    "duration", "durationSeconds", "viewCount",
    "commentCount", "definicion", "idioma_audio",
]

OUTPUT_DIR = os.path.join(_BASEDIR, "dataset", "channels")


# ── Utilidades ───────────────────────────────────────────────────────────────
def parse_duration(iso: str) -> int:
    """PT1H2M3S → segundos."""
    if not iso:
        return 0
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso)
    return (int(m[1] or 0) * 3600 + int(m[2] or 0) * 60 + int(m[3] or 0)) if m else 0


def safe_filename(name: str) -> str:
    """Convierte un nombre en un nombre de archivo seguro."""
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    return name.strip().rstrip('.')


def extract(item: dict) -> dict:
    """Extrae los campos deseados de un item de videos.list."""
    sn = item.get("snippet", {})
    cd = item.get("contentDetails", {})
    st = item.get("statistics", {})
    dur_iso = cd.get("duration", "")
    return {
        "id":              item.get("id", ""),
        "title":           sn.get("title", ""),
        "description":     sn.get("description", ""),
        "publishedAt":     sn.get("publishedAt", ""),
        "channelId":       sn.get("channelId", ""),
        "channelTitle":    sn.get("channelTitle", ""),
        "tags":            "|".join(sn.get("tags") or []),
        "categoryId":      sn.get("categoryId", ""),
        "duration":        dur_iso,
        "durationSeconds": parse_duration(dur_iso),
        "viewCount":       int(st.get("viewCount") or 0),
        "commentCount":    int(st.get("commentCount") or 0),
        "definicion":      cd.get("definition", ""),
        "idioma_audio":    (sn.get("defaultAudioLanguage")
                            or sn.get("defaultLanguage") or ""),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DESCARGADOR DE CANAL
# ══════════════════════════════════════════════════════════════════════════════
class ChannelDownloader:
    """Descarga TODOS los videos de un canal y los guarda en CSV."""

    def __init__(self, api_key: str, channel_handle: str, channel_label: str,
                 output_dir: str):
        self.api_key = api_key
        self.handle = channel_handle
        self.label = channel_label
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.channel_id = None
        self.channel_title = None
        self.seen = set()
        self.total_added = 0
        self.quota_used = 0

    # ── Resolver handle → channelId ──────────────────────────────────────
    def resolve_channel(self) -> bool:
        """Obtiene el channelId a partir del handle (@usuario)."""
        print(f"\n🔍 Resolviendo canal: {self.handle}")
        r = requests.get(CHANNELS_URL, params={
            "part": "snippet,contentDetails,statistics",
            "forHandle": self.handle.lstrip("@"),
            "key": self.api_key,
        }, timeout=30)
        self.quota_used += 1
        r.raise_for_status()
        items = r.json().get("items", [])
        if not items:
            print(f"  ✗ No se encontró el canal con handle: {self.handle}")
            return False
        ch = items[0]
        self.channel_id = ch["id"]
        self.channel_title = ch["snippet"]["title"]
        total_videos = ch["statistics"].get("videoCount", "?")
        print(f"  ✓ Canal: {self.channel_title}")
        print(f"    ID:     {self.channel_id}")
        print(f"    Videos reportados: {total_videos}")
        return True

    # ── Cargar progreso existente ────────────────────────────────────────
    def _load_existing(self, csv_path: str):
        """Carga IDs ya descargados para retomar sin duplicados."""
        if not os.path.exists(csv_path):
            return
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                vid = row.get("id")
                if vid:
                    self.seen.add(vid)
        if self.seen:
            print(f"  ↻ Retomando: {len(self.seen)} videos ya descargados")

    # ── Enriquecer videos con videos.list ────────────────────────────────
    def _enrich(self, ids: list) -> list:
        """Obtiene detalles completos de hasta 50 videos por llamada."""
        if not ids:
            return []
        results = []
        # videos.list acepta máximo 50 IDs por llamada
        for i in range(0, len(ids), 50):
            batch = ids[i:i+50]
            r = requests.get(VIDEOS_URL, params={
                "part": "snippet,contentDetails,statistics",
                "id": ",".join(batch),
                "maxResults": 50,
                "key": self.api_key,
            }, timeout=30)
            self.quota_used += 1
            r.raise_for_status()
            results.extend(r.json().get("items", []))
            time.sleep(0.1)
        return results

    # ── Búsqueda paginada de todos los videos del canal ──────────────────
    def _search_all_video_ids(self) -> list:
        """Usa search.list con channelId para obtener TODOS los videoIds.

        Pagina con pageToken hasta agotar resultados.
        Coste: 100 unidades de cuota por página de hasta 50 resultados.
        """
        all_ids = []
        page_token = None
        page = 0

        while True:
            params = {
                "part": "id",
                "type": "video",
                "channelId": self.channel_id,
                "maxResults": 50,
                "order": "date",
                "key": self.api_key,
            }
            if page_token:
                params["pageToken"] = page_token

            try:
                r = requests.get(SEARCH_URL, params=params, timeout=30)
            except requests.exceptions.RequestException as e:
                print(f"  ⚠ Error de red en search: {e}")
                time.sleep(2)
                continue

            self.quota_used += 100  # search.list = 100 unidades

            if r.status_code == 403:
                errs = r.json().get("error", {}).get("errors", [])
                if any(e.get("reason") in ("quotaExceeded", "dailyLimitExceeded")
                       for e in errs):
                    print("  ⚠ Cuota de YouTube agotada.")
                    break
                r.raise_for_status()

            r.raise_for_status()
            data = r.json()
            ids = [
                item["id"]["videoId"]
                for item in data.get("items", [])
                if item.get("id", {}).get("videoId")
            ]
            all_ids.extend(ids)
            page += 1
            print(f"    Página {page}: {len(ids)} videos "
                  f"(acumulado: {len(all_ids)}) | cuota: ~{self.quota_used}")

            page_token = data.get("nextPageToken")
            if not page_token:
                break
            time.sleep(0.15)

        return all_ids

    # ── Descarga principal ───────────────────────────────────────────────
    def download(self):
        """Ejecuta la descarga completa del canal."""
        if not self.resolve_channel():
            return

        fname = safe_filename(self.channel_title) + ".csv"
        csv_path = os.path.join(self.output_dir, fname)
        self._load_existing(csv_path)

        sep = "─" * 55
        print(sep)
        print(f"  Descargando: {self.channel_title}")
        print(f"  Archivo:     {csv_path}")
        print(sep)

        # Fase 1: Obtener todos los IDs de videos del canal
        print("\n  📋 Fase 1: Recopilando IDs de videos...")
        all_ids = self._search_all_video_ids()

        # Filtrar IDs ya descargados
        new_ids = [vid for vid in all_ids if vid not in self.seen]
        print(f"\n  Total IDs encontrados: {len(all_ids)}")
        print(f"  Nuevos (por descargar): {len(new_ids)}")
        if not new_ids:
            print("  ✓ No hay videos nuevos que descargar.")
            return

        # Fase 2: Enriquecer con detalles completos
        print(f"\n  📥 Fase 2: Descargando detalles de {len(new_ids)} videos...")
        has_data = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        with open(csv_path, "a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
            if not has_data:
                writer.writeheader()

            # Procesar en lotes de 50
            for i in range(0, len(new_ids), 50):
                batch = new_ids[i:i+50]
                items = self._enrich(batch)
                for it in items:
                    rec = extract(it)
                    vid = rec["id"]
                    if vid in self.seen:
                        continue
                    writer.writerow({k: rec.get(k, "") for k in FIELDNAMES})
                    self.seen.add(vid)
                    self.total_added += 1

                fh.flush()
                done = min(i + 50, len(new_ids))
                print(f"    Progreso: {done}/{len(new_ids)} "
                      f"(+{self.total_added} guardados) | cuota: ~{self.quota_used}")
                time.sleep(0.1)

        print(f"\n  ✅ {self.channel_title}: {self.total_added} videos añadidos "
              f"(total en archivo: {len(self.seen)})")
        print(f"     Cuota usada: ~{self.quota_used} unidades")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Descarga TODOS los videos de canales específicos de YouTube")
    ap.add_argument("--api-key",
                    help="YouTube Data API v3 key (o YOUTUBE_API_KEY en .env)")
    args = ap.parse_args()

    key = args.api_key or os.environ.get("YOUTUBE_API_KEY")
    if not key:
        # Intentar leer del archivo de API keys
        api_file = os.path.join(_BASEDIR, "API YOUTUBE (LofiAPI).txt")
        if os.path.exists(api_file):
            with open(api_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("AIza"):
                        key = line
                        break
    if not key:
        sys.exit("✗ Falta API key. Usa --api-key CLAVE "
                 "o define YOUTUBE_API_KEY en .env")

    print("═" * 55)
    print("  Descarga de canales YouTube – Todos los videos")
    print(f"  Destino: {OUTPUT_DIR}")
    print(f"  Canales: {len(TARGET_CHANNELS)}")
    print("═" * 55)

    total_quota = 0
    for label, handle in TARGET_CHANNELS:
        dl = ChannelDownloader(key, handle, label, OUTPUT_DIR)
        dl.download()
        total_quota += dl.quota_used

    print("\n" + "═" * 55)
    print(f"  🏁 Descarga completa")
    print(f"     Cuota total usada: ~{total_quota} unidades")
    print("═" * 55)


if __name__ == "__main__":
    main()





