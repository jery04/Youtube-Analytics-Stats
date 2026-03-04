#!/usr/bin/env python3
"""
Pipeline de descarga de videos de YouTube – Categoría Entretenimiento (24)
==========================================================================
• Shorts:  duración ≤ 60 s
• Largos:  3 min < duración ≤ 16 min  (es decir, (180, 960] segundos)
• Proporción objetivo: 50 / 50
• Cuota diaria: 10 000 unidades  (search.list=100, videos.list=1 por llamada)
• Soporte incremental: guarda cursor en dataset/state.json

Estrategia de cuota:
  Cada «página» = 1 search (100) + 1 videos (1) = 101 unidades
  10 000 / 101 ≈ 99 páginas × 50 candidatos = ~4 950 videos examinados/día

Uso:
    python scripts/index.py                     # lee YOUTUBE_API_KEY de .env
    python scripts/index.py --api-key CLAVE
    python scripts/index.py --quota 5000        # limitar cuota
"""

import os
import sys
import time
import json
import csv
import re
import argparse
import calendar
import datetime
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
SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

COST_SEARCH = 100   # quota units por search.list
COST_VIDEOS = 1     # quota units por videos.list
COST_PAGE   = COST_SEARCH + COST_VIDEOS  # 101 por lote de hasta 50 videos

# Queries rotativas para descubrir videos variados de entretenimiento.
# search.list con videoCategoryId=24 requiere un parámetro q para devolver
# resultados; rotamos queries para obtener diversidad.
QUERIES = [
    "entertainment", "funny", "comedy", "prank", "challenge",
    "reaction", "viral", "trending", "humor", "skit",
    "fails", "memes", "try not to laugh", "stand up", "parody",
    "entertainment show", "talent show", "funny moments", "best of",
    "top entertainment",
]

# Proporción objetivo shorts:longs (p.ej. 2.0 → 2 shorts por cada largo, ~67% shorts)
TARGET_RATIO = 2.0

FIELDNAMES = [
    "id", "title", "description", "publishedAt",
    "channelId", "channelTitle", "tags", "categoryId",
    "duration", "durationSeconds",
    "viewCount", "commentCount", "definicion", "idioma_audio",
]

# Limitar búsquedas a este rango (inclusive)
# Desde enero 2025 hasta fin de 2026
YEAR_START = "2025-01-01T00:00:00Z"
YEAR_END = "2026-12-31T23:59:59Z"

# ── Utilidades ───────────────────────────────────────────────────────────────
def parse_duration(iso):
    """PT1H2M3S → segundos."""
    if not iso:
        return 0
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso)
    return (int(m[1] or 0) * 3600 + int(m[2] or 0) * 60 + int(m[3] or 0)) if m else 0


def classify(secs):
    """short | long | None (descartado).

    - `short`: duración positiva y <= 60 s
    - `long` : duración estrictamente mayor a 180 s (3 min) y <= 960 s (16 min)
    - `None` : fuera de los rangos objetivo
    """
    try:
        secs = int(secs or 0)
    except (TypeError, ValueError):
        return None
    if secs > 0 and secs <= 60:
        return "short"
    if secs > 180 and secs <= 960:
        return "long"
    return None


def extract(item):
    """Extrae campos deseados de un item de videos.list."""
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
#  PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
class Pipeline:
    """
    Etapas por cada página:
      1. SEARCH  → IDs de videos  (100 quota / ≤50 resultados)
      2. ENRICH  → detalles completos  (1 quota / ≤50 IDs)
      3. FILTER  → descarta fuera de rango + verifica categoría 24
      4. STORE   → escribe CSV balanceando short/long

    Estado persistente en state.json para retomar cada día.
    Usa queries rotativas y videoDuration para balancear 50/50.
    """

    def __init__(self, api_key, dataset_dir, quota=10_000):
        self.api_key   = api_key
        self.dir       = os.path.abspath(dataset_dir)
        self.quota_max = quota
        self.quota_used = 0
        self.verbose = True

        self.csv_path   = os.path.join(self.dir, "videos.csv")
        self.state_path = os.path.join(self.dir, "state.json")
        os.makedirs(self.dir, exist_ok=True)

        # Estado persistente (cursores de fecha, índice de query, etc.)
        self.state = self._load_json(self.state_path) or {}

        # Cargar IDs existentes y contadores
        self.seen   = set()
        self.counts = {"short": 0, "long": 0}
        self._scan_existing()

        # CSV writer (append)
        has_data = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0
        self._fh = open(self.csv_path, "a", encoding="utf-8", newline="")
        self._wr = csv.DictWriter(self._fh, fieldnames=FIELDNAMES)
        if not has_data:
            self._wr.writeheader()

        self.added = 0

    # ── helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _load_json(path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _log(self, *args, **kwargs):
        if getattr(self, "verbose", False):
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            print("[", ts, "]", *args, **kwargs)

    def _save_state(self):
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def _scan_existing(self):
        """Lee CSV existente para reconstruir seen-set y contadores."""
        if not os.path.exists(self.csv_path):
            return
        with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                vid = row.get("id")
                if not vid:
                    continue
                # Siempre marcar ID como visto para evitar duplicados,
                # pero sólo contar los videos dentro del rango 2025-2026
                self.seen.add(vid)
                pa = row.get("publishedAt") or ""
                in_range = False
                if pa and YEAR_START <= pa <= YEAR_END:
                    in_range = True
                try:
                    dur = int(row.get("durationSeconds") or 0)
                except ValueError:
                    dur = 0
                t = classify(dur)
                if t and in_range:
                    self.counts[t] += 1

    def _budget_left(self):
        return self.quota_max - self.quota_used

    def _can_page(self):
        return self._budget_left() >= COST_PAGE

    def _next_query(self):
        """Devuelve la siguiente query de la lista rotativa."""
        idx = self.state.get("query_idx", 0) % len(QUERIES)
        q = QUERIES[idx]
        self.state["query_idx"] = idx + 1
        return q

    def _duration_filter(self):
        """Selecciona filtro videoDuration priorizando shorts (≤1 min).

        YouTube API videoDuration values:
          'short'  → <4 min  (contiene nuestros shorts ≤60s)
          'medium' → 4-20 min (contiene largos 4-16min)

        Estrategia: buscar siempre con 'short' a menos que los shorts
        ya superen TARGET_RATIO veces los largos, en cuyo caso buscamos
        'medium' para reequilibrar.
        """
        s, l = self.counts["short"], self.counts["long"]
        # Si no hay largos, o los shorts aún no alcanzan el ratio objetivo
        # → seguir priorizando shorts
        if l == 0 or (s / l) < TARGET_RATIO:
            return "short"
        # Shorts ya superan TARGET_RATIO × largos → buscar largos
        return "medium"

    # ── API wrappers ─────────────────────────────────────────────────────
    def _api_search(self, query, page_token=None, published_after=None,
                    published_before=None, video_duration=None):
        """search.list → (ids, nextPageToken, quota_exceeded)"""
        params = {
            "part": "id",
            "type": "video",
            "q": query,
            "videoCategoryId": "24",
            "maxResults": 50,
            "order": "date",
            "regionCode": "US",
            "key": self.api_key,
        }
        if page_token:
            params["pageToken"] = page_token
        # Clamp publishedAfter/Before to the allowed YEAR_START..YEAR_END range
        pa = published_after or YEAR_START
        pb = published_before or YEAR_END
        if pa < YEAR_START:
            pa = YEAR_START
        if pa > YEAR_END:
            pa = YEAR_END
        if pb > YEAR_END:
            pb = YEAR_END
        if pb < YEAR_START:
            pb = YEAR_START
        # If range is invalid, nothing to search
        if pa > pb:
            return [], None, False
        params["publishedAfter"] = pa
        params["publishedBefore"] = pb
        if video_duration:
            params["videoDuration"] = video_duration

        # Log outgoing search parameters
        self._log("SEARCH -> q=", query, "pageToken=", page_token,
                  "publishedAfter=", pa, "publishedBefore=", pb,
                  "videoDuration=", video_duration)
        try:
            r = requests.get(SEARCH_URL, params=params, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"  ⚠  Error de red: {e}")
            return [], None, False
        self.quota_used += COST_SEARCH
        self._log("quota used after search:", self.quota_used)

        if r.status_code == 403:
            errs = r.json().get("error", {}).get("errors", [])
            if any(e.get("reason") in ("quotaExceeded", "dailyLimitExceeded")
                   for e in errs):
                return [], None, True
            r.raise_for_status()

        r.raise_for_status()
        data = r.json()
        ids = [
            i["id"]["videoId"]
            for i in data.get("items", [])
            if i.get("id", {}).get("videoId")
        ]
        self._log(f"SEARCH result: {len(ids)} ids, nextPageToken=", data.get("nextPageToken"))
        return ids, data.get("nextPageToken"), False

    def _api_videos(self, ids):
        """videos.list → lista de items enriquecidos."""
        if not ids:
            return []
        self._log("ENRICH -> requesting details for ids:", ids)
        try:
            r = requests.get(VIDEOS_URL, params={
                "part": "snippet,contentDetails,statistics",
                "id": ",".join(ids),
                "maxResults": 50,
                "key": self.api_key,
            }, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"  ⚠  Error de red: {e}")
            return []
        self.quota_used += COST_VIDEOS
        self._log("quota used after videos.list:", self.quota_used)
        r.raise_for_status()
        items = r.json().get("items", [])
        self._log(f"ENRICH result: {len(items)} items")
        return items

    # ── etapas 3+4: FILTER + STORE ───────────────────────────────────────
    def _process(self, items):
        """Filtra, balancea y guarda items en CSV.  Actualiza cursores de fecha."""
        for it in items:
            rec = extract(it)
            vid = rec["id"]
            if not vid:
                self._log("SKIP: missing id")
                continue
            if vid in self.seen:
                self._log("SKIP: already seen", vid)
                continue

            # Verificar categoría 24 (entretenimiento)
            if rec["categoryId"] != "24":
                self._log("SKIP: wrong category", rec.get("categoryId"), vid)
                continue

            # Rastrear rango de fechas (incluso si no guardamos el video)
            pa = rec["publishedAt"]
            if not pa or not (YEAR_START <= pa <= YEAR_END):
                self._log("SKIP: outside year range", pa or "(no date)", vid)
                # still track seen to avoid reprocessing
                self.seen.add(vid)
                continue
            else:
                if not self.state.get("newest") or pa > self.state["newest"]:
                    self.state["newest"] = pa
                if not self.state.get("oldest") or pa < self.state["oldest"]:
                    self.state["oldest"] = pa

            cat = classify(rec["durationSeconds"])
            if cat is None:
                self._log("SKIP: duration out of target range", rec.get("durationSeconds"), vid)
                self.seen.add(vid)
                continue

            # Balance con prioridad a shorts: descartar si se supera el ratio objetivo
            other = "long" if cat == "short" else "short"
            if self.counts[cat] > 0 and self.counts[other] > 0:
                ratio = self.counts[cat] / self.counts[other]
                # Para shorts: tolerar hasta TARGET_RATIO × largos
                # Para largos: tolerar hasta 1/TARGET_RATIO × shorts
                limit = TARGET_RATIO if cat == "short" else (1.0 / TARGET_RATIO)
                if ratio > limit * 1.5:  # margen del 50% antes de descartar
                    self._log("SKIP: imbalance drop", cat, vid)
                    self.seen.add(vid)
                    continue

            # Añadir
            self._wr.writerow({k: rec.get(k, "") for k in FIELDNAMES})
            self._fh.flush()
            self.seen.add(vid)
            self.counts[cat] += 1
            self.added += 1
            self._log("ADD:", vid, rec.get("title")[:60], "cat=", cat,
                      "dur_s=", rec.get("durationSeconds"))

    # ── crawl loop ───────────────────────────────────────────────────────
    def _crawl(self, tag, published_after=None, published_before=None):
        """Recorre search→enrich→filter→store hasta agotar cuota o datos.

        Rota queries cada vez que se agotan los page tokens para explorar
        con una nueva búsqueda.  Usa videoDuration para balancear 50/50.
        """
        token = None
        pages = 0
        last_before = published_before  # para detectar falta de progreso
        query = self._next_query()
        dur_filter = self._duration_filter()
        empty_streak = 0  # queries consecutivas sin resultados

        while self._can_page():
            # 1) SEARCH
            self._log("CRAWL: page", pages+1, "query=", query, "dur_filter=", dur_filter,
                      "published_after=", published_after, "published_before=", published_before)
            ids, nxt, exceeded = self._api_search(
                query, token, published_after, published_before, dur_filter,
            )
            pages += 1
            if exceeded:
                print("  ⚠  Cuota de YouTube agotada. Reintenta mañana.")
                return pages, True
            if not ids:
                self._log("CRAWL: no ids returned for query", query)
                empty_streak += 1
                if empty_streak >= 3:
                    break  # 3 queries sin resultados → parar
                query = self._next_query()
                dur_filter = self._duration_filter()
                token = None
                continue

            empty_streak = 0

            # 2) ENRICH
            items = self._api_videos(ids)

            # 3+4) FILTER + STORE
            self._process(items)
            self._save_state()

            # Paginación
            if nxt:
                self._log("CRAWL: nextPageToken present, continuing pagination")
                token = nxt
            else:
                # Sin más page-tokens → nueva query o retroceder ventana
                if tag == "explore" and self.state.get("oldest"):
                    new_before = self.state["oldest"]
                    if new_before != last_before:
                        last_before = new_before
                        published_before = new_before
                # Siguiente query con posible nuevo filtro de duración
                query = self._next_query()
                dur_filter = self._duration_filter()
                token = None
                continue

            # Progreso cada 10 páginas
            if pages % 10 == 0:
                print(f"    pág {pages} | cuota {self.quota_used}/{self.quota_max} | "
                      f"+{self.added} (short={self.counts['short']}, "
                      f"long={self.counts['long']})")

            time.sleep(0.15)  # cortesía de rate-limit

        return pages, False

    # ── ejecución principal ──────────────────────────────────────────────
    def run(self):
        sep = "═" * 60
        print(sep)
        print("  Pipeline YouTube · Entretenimiento (cat 24)")
        print(f"  CSV:    {self.csv_path}")
        print(f"  Estado: {self.state_path}")
        print(f"  Dataset existente: {len(self.seen)} videos "
              f"(short={self.counts['short']}, long={self.counts['long']})")
        print(f"  Cuota: {self.quota_max} unidades "
              f"(~{self.quota_max // COST_PAGE} páginas de 50 videos)")
        print(sep)

        total_pages = 0
        if self.verbose:
            print("Verbose mode: detailed pipeline logging enabled")

        # Construir lista de (año, mes) desde hoy hacia atrás hasta enero 2025
        today = datetime.date.today()
        STOP_YEAR, STOP_MONTH = 2025, 1

        year_months = []
        y, m = today.year, today.month
        while (y, m) >= (STOP_YEAR, STOP_MONTH):
            year_months.append((y, m))
            m -= 1
            if m == 0:
                m = 12
                y -= 1

        print(f"\n► Recorrido cronológico descendente: "
              f"{year_months[0][0]}-{year_months[0][1]:02d} → "
              f"{year_months[-1][0]}-{year_months[-1][1]:02d}")

        for y, month in year_months:
            if not self._can_page():
                break
            last_day = calendar.monthrange(y, month)[1]
            pa = f"{y}-{month:02d}-01T00:00:00Z"
            pb = f"{y}-{month:02d}-{last_day:02d}T23:59:59Z"
            print(f"  ↳ {y}-{month:02d} ({pa[:10]} → {pb[:10]})")
            p, exc = self._crawl("year", published_after=pa, published_before=pb)
            total_pages += p
            if exc:
                self._finish(total_pages)
                return

        # Si queda cuota, permitir una pasada general (retrocompatible)
        if self._can_page():
            print("\n► Exploración adicional (sin límite anual)")
            p, _ = self._crawl("explore")
            total_pages += p

        self._finish(total_pages)

    def _finish(self, pages=0):
        self._fh.close()
        self._save_state()
        sep = "═" * 60
        print(f"\n{sep}")
        print(f"  Páginas: {pages}  |  Cuota: {self.quota_used}/{self.quota_max}")
        print(f"  Videos añadidos esta sesión: {self.added}")
        print(f"  Total en dataset: {len(self.seen)} "
              f"(short={self.counts['short']}, long={self.counts['long']})")
        if self.state.get("newest") and self.state.get("oldest"):
            print(f"  Rango: {self.state['oldest'][:10]} → "
                  f"{self.state['newest'][:10]}")
        print(sep)


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Pipeline de descarga YouTube Entretenimiento")
    ap.add_argument("--api-key",
                    help="YouTube Data API v3 key (o YOUTUBE_API_KEY en .env)")
    ap.add_argument("--dataset-dir",
                    default=os.path.join(os.path.dirname(__file__),
                                         "..", "dataset"),
                    help="Carpeta del dataset (default: ./dataset)")
    ap.add_argument("--quota", type=int, default=10_000,
                    help="Presupuesto de cuota diario (default: 10 000)")
    ap.add_argument("--verbose", action="store_true",
                    help="Mostrar logging detallado del pipeline")
    args = ap.parse_args()

    key = args.api_key or os.environ.get("YOUTUBE_API_KEY")
    if not key:
        sys.exit("✗ Falta API key. Usa --api-key CLAVE "
                 "o define YOUTUBE_API_KEY en .env")

    p = Pipeline(key, args.dataset_dir, args.quota)
    p.verbose = bool(args.verbose)
    p.run()


if __name__ == "__main__":
    main()






