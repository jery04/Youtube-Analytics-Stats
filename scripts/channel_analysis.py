"""
Análisis completo del canal "¿Sabías Esto..?" 
──────────────────────────────────────────────
Genera imágenes y TXT para:
  1. Distribución por día de la semana (todos los videos, sin filtro de duración)
  2. Videos por horario de 1h agrupados por día de la semana
  3. Palabras en el título (bins personalizados)
  4. Duración agrupada de 20 en 20 s
  5. Caracteres en la descripción (bins personalizados)
"""

import os
import re
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Rutas ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "dataset" / "channels" / "¿Sabías Esto.._.csv"
OUT_DIR = ROOT / "outputs" / "Canal_SabiasEsto"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Utilidades ──────────────────────────────────────────────────────────
def _to_seconds(x):
    """Parsea un valor de duración a segundos (ISO 8601, HH:MM:SS, numérico)."""
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        val = float(x)
        return int(val / 1000) if val > 1e6 else int(val)
    s = str(x).strip()
    m_iso = re.match(r'^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$', s)
    if m_iso:
        h = int(m_iso.group(1) or 0)
        m = int(m_iso.group(2) or 0)
        ss = int(m_iso.group(3) or 0)
        return h * 3600 + m * 60 + ss
    if ':' in s:
        parts = [p.strip() for p in s.split(':') if p.strip() != '']
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(float(parts[2]))
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(float(parts[1]))
        except Exception:
            pass
    m_digits = re.search(r'^(\d+(?:\.\d+)?)$', s)
    if m_digits:
        try:
            return int(float(m_digits.group(1)))
        except Exception:
            return None
    return None

def _load():
    """Carga el CSV y prepara columnas auxiliares."""
    df = pd.read_csv(CSV_PATH, low_memory=False)

    # Fecha
    date_col = None
    for c in ("publishedAt", "publish_time", "fecha_publicacion", "fecha"):
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        for c in df.columns:
            if re.search(r'publish|fecha|date|upload', c, re.I):
                date_col = c
                break
    if date_col:
        try:
            df['__dt'] = pd.to_datetime(df[date_col], format='mixed', errors='coerce')
        except Exception:
            df['__dt'] = pd.to_datetime(df[date_col].astype(str), format='mixed', errors='coerce')
    else:
        df['__dt'] = pd.NaT

    # Duración en segundos
    dur_col = None
    for c in ("durationSeconds", "duration_seconds", "duration", "duracion_iso"):
        if c in df.columns:
            dur_col = c
            break
    if dur_col is None:
        for c in df.columns:
            if re.search(r'duraci|duration|length', c, re.I):
                dur_col = c
                break
    if dur_col:
        if dur_col in ("durationSeconds", "duration_seconds", "duration_sec", "length_seconds"):
            df['__dur_s'] = pd.to_numeric(df[dur_col], errors='coerce')
        else:
            df['__dur_s'] = df[dur_col].apply(_to_seconds)
    else:
        df['__dur_s'] = None

    return df

# ═══════════════════════════════════════════════════════════════════════
# 1. Distribución por día de la semana (todos los videos)
# ═══════════════════════════════════════════════════════════════════════
def analyze_weekday_distribution(df):
    sub = df.dropna(subset=['__dt']).copy()
    if sub.empty:
        print("Sin fechas válidas para weekday_distribution.")
        return

    sub['__wd'] = sub['__dt'].dt.dayofweek  # Lunes=0 .. Domingo=6
    labels = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    counts = sub['__wd'].value_counts().reindex(range(7), fill_value=0)

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=labels, y=counts.values, palette='viridis')
    for i, v in enumerate(counts.values):
        ax.annotate(str(int(v)), (i, v), textcoords="offset points",
                     xytext=(0, 5), ha='center', fontsize=10, fontweight='bold')
    plt.xlabel('Día de la semana')
    plt.ylabel('Cantidad de videos')
    plt.title('Videos publicados por día de la semana — ¿Sabías Esto..?')
    plt.tight_layout()

    out = OUT_DIR / "weekday_distribution"
    plt.savefig(f"{out}.png", dpi=150)
    with open(f"{out}.txt", 'w', encoding='utf-8') as f:
        f.write('Dia,Count\n')
        for i, lbl in enumerate(labels):
            f.write(f"{lbl},{int(counts.get(i, 0))}\n")
    plt.close()
    print(f"[1] Weekday distribution → {out}.png / .txt")

# ═══════════════════════════════════════════════════════════════════════
# 2. Por cada día de la semana: imagen con horarios de 2 h
# ═══════════════════════════════════════════════════════════════════════
def analyze_hourly_per_day(df, interval_hours=2):

    def analyze_title_word_count(df):
        title_col = None
        for c in ("title", "titulo", "nombre", "video_title"):
            if c in df.columns:
                title_col = c
                break
        if title_col is None:
            for c in df.columns:
                if re.search(r'title|titulo|nombre', c, re.I):
                    title_col = c
                    break
        if title_col is None:
            print("No se encontró columna de título.")
            return

        sub = df.dropna(subset=[title_col]).copy()
        sub['__wc'] = sub[title_col].astype(str).str.split().apply(len)

        bins = [(1, 3), (4, 6), (7, 9), (10, 12), (13, 15), (16, 18), (19, 21), (22, 24)]
        labels = [f"{a}-{b}" for a, b in bins]
        counts = []
        videos_per_bin = {}

        for a, b in bins:
            mask = (sub['__wc'] >= a) & (sub['__wc'] <= b)
            counts.append(int(mask.sum()))
            titles = sub.loc[mask, title_col].tolist()
            videos_per_bin[f"{a}-{b}"] = [str(t) for t in titles if pd.notna(t)]

        total = sum(counts)

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=labels, y=counts, palette='magma')
        for i, v in enumerate(counts):
            pct = (v / total * 100) if total > 0 else 0
            ax.annotate(f"{v} ({pct:.1f}%)", (i, v), textcoords="offset points",
                         xytext=(0, 5), ha='center', fontsize=9, fontweight='bold')
        plt.xlabel('Número de palabras en el título')
        plt.ylabel('Cantidad de videos')
        plt.title('Distribución por palabras en el título — ¿Sabías Esto..?')
        plt.tight_layout()

        out = OUT_DIR / "title_word_count"
        plt.savefig(f"{out}.png", dpi=150)

        with open(f"{out}.txt", 'w', encoding='utf-8') as f:
            f.write('Intervalo,Count,Percent\n')
            for lbl, cnt in zip(labels, counts):
                pct = (cnt / total * 100) if total > 0 else 0
                f.write(f"[{lbl}],{cnt},{pct:.2f}\n")
            f.write(f"\n# Detalle de videos por intervalo\n")
            for lbl in labels:
                vids = videos_per_bin[lbl]
                f.write(f"\n[{lbl}] ({len(vids)} videos)\n")
                for t in vids:
                    f.write(f"  - {t}\n")

        plt.close()
        print(f"[3] Title word count → {out}.png / .txt")
    """
    Genera imágenes y TXT de videos publicados por intervalos horarios ajustables (por día de la semana).
    interval_hours: tamaño del intervalo en horas (por defecto 2).
    """
    sub = df.dropna(subset=['__dt']).copy()
    if sub.empty:
        print("Sin fechas válidas para hourly_per_day.")
        return

    sub['__wd'] = sub['__dt'].dt.dayofweek
    sub['__hour_frac'] = (
        sub['__dt'].dt.hour.fillna(0).astype(float)
        + sub['__dt'].dt.minute.fillna(0).astype(float) / 60.0
        + sub['__dt'].dt.second.fillna(0).astype(float) / 3600.0
    )

    dias = {
        0: 'Lunes', 1: 'Martes', 2: 'Miércoles',
        3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo',
    }
    dias_file = {
        0: 'lunes', 1: 'martes', 2: 'miercoles',
        3: 'jueves', 4: 'viernes', 5: 'sabado', 6: 'domingo',
    }

    # Construir bins y etiquetas
    bin_size = float(interval_hours)
    n_steps = int(round(24.0 / bin_size))
    bins = [round(i * bin_size, 6) for i in range(n_steps + 1)]
    hour_labels = [f"{int(bins[i]):02d}:00-{int(bins[i+1]):02d}:00" if bin_size >= 1 else f"{bins[i]:.2f}-{bins[i+1]:.2f}h" for i in range(len(bins) - 1)]

    day_dir = OUT_DIR / "horarios_por_dia"
    day_dir.mkdir(parents=True, exist_ok=True)

    # Histograma global (todos los días juntos)
    cats_all = pd.cut(sub['__hour_frac'], bins=bins, labels=hour_labels, include_lowest=True, right=True)
    counts_all = cats_all.value_counts().reindex(hour_labels, fill_value=0)
    videos_per_bin_all = {lbl: [] for lbl in hour_labels}
    for lbl in hour_labels:
        vids = sub.loc[cats_all == lbl, 'title'].tolist()
        videos_per_bin_all[lbl] = [str(t) for t in vids if pd.notna(t)]

    # Imagen global
    fig, ax = plt.subplots(figsize=(max(14, len(hour_labels) * 0.6), 7))
    bars = ax.bar(range(len(hour_labels)), counts_all.values, color=sns.color_palette('viridis', len(hour_labels)),
                   edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(hour_labels)))
    ax.set_xticklabels(hour_labels, rotation=60, ha='right', fontsize=8)
    ax.set_xlabel('Horario')
    ax.set_ylabel('Videos publicados')
    ax.set_title(f'Videos por horario ({interval_hours} h) — TODOS LOS DÍAS — ¿Sabías Esto..?')
    for i, v in enumerate(counts_all.values):
        if v > 0:
            ax.annotate(str(int(v)), (i, v), textcoords="offset points",
                         xytext=(0, 4), ha='center', fontsize=8, fontweight='bold')
    plt.tight_layout()
    out_png_all = OUT_DIR / f"hour_2h_histogram_todos.png"
    plt.savefig(out_png_all, dpi=150)
    plt.close()

    # TXT global
    out_txt_all = OUT_DIR / f"hour_2h_histogram_todos.txt"
    with open(out_txt_all, 'w', encoding='utf-8') as f:
        f.write(f"# TODOS LOS DÍAS\n")
        f.write(f"# Total videos: {len(sub)}\n")
        f.write(f"# Intervalo horario: {interval_hours} horas\n\n")
        for lbl in hour_labels:
            vids = videos_per_bin_all[lbl]
            f.write(f"[{lbl}] ({len(vids)} videos)\n")
            for t in vids:
                f.write(f"  - {t}\n")
            f.write("\n")
    print(f"[2G] Histograma global → {out_png_all}")

    # Por día de la semana (como antes)
    for day_num in range(7):
        day_name = dias[day_num]
        day_file = dias_file[day_num]
        df_day = sub[sub['__wd'] == day_num]

        # Agrupar por bins
        cats = pd.cut(df_day['__hour_frac'], bins=bins, labels=hour_labels, include_lowest=True, right=True)
        counts = cats.value_counts().reindex(hour_labels, fill_value=0)

        # Recopilar títulos de videos por intervalo
        videos_per_bin = {lbl: [] for lbl in hour_labels}
        for lbl in hour_labels:
            vids = df_day.loc[cats == lbl, 'title'].tolist()
            videos_per_bin[lbl] = [str(t) for t in vids if pd.notna(t)]

        # Imagen
        fig, ax = plt.subplots(figsize=(max(14, len(hour_labels) * 0.6), 7))
        bars = ax.bar(range(len(hour_labels)), counts.values, color=sns.color_palette('viridis', len(hour_labels)),
                       edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(len(hour_labels)))
        ax.set_xticklabels(hour_labels, rotation=60, ha='right', fontsize=8)
        ax.set_xlabel('Horario')
        ax.set_ylabel('Videos publicados')
        ax.set_title(f'Videos por horario ({interval_hours} h) — {day_name} — ¿Sabías Esto..?')

        # Anotar cantidad encima
        for i, v in enumerate(counts.values):
            if v > 0:
                ax.annotate(str(int(v)), (i, v), textcoords="offset points",
                             xytext=(0, 4), ha='center', fontsize=8, fontweight='bold')

        plt.tight_layout()
        out_png = day_dir / f"hourly_{day_file}.png"
        plt.savefig(out_png, dpi=150)
        plt.close()

        # TXT con detalle de videos
        out_txt = day_dir / f"hourly_{day_file}.txt"
        with open(out_txt, 'w', encoding='utf-8') as f:
            f.write(f"# Día: {day_name}\n")
            f.write(f"# Total videos este día: {len(df_day)}\n")
            f.write(f"# Intervalo horario: {interval_hours} horas\n\n")
            for lbl in hour_labels:
                vids = videos_per_bin[lbl]
                f.write(f"[{lbl}] ({len(vids)} videos)\n")
                for t in vids:
                    f.write(f"  - {t}\n")
                f.write("\n")

        print(f"[2] {day_name} → {out_png}")

# ═══════════════════════════════════════════════════════════════════════
# 4. Duración agrupada de 20 en 20 s
# ═══════════════════════════════════════════════════════════════════════
def analyze_duration_20s(df):
    sub = df.dropna(subset=['__dur_s']).copy()
    if sub.empty:
        print("Sin duraciones válidas para duration_20s.")
        return

    max_dur = int(sub['__dur_s'].max())
    step = 20
    # Crear bins de 20 en 20 hasta cubrir el máximo
    upper = ((max_dur // step) + 1) * step
    bin_edges = list(range(0, upper + step, step))
    bins = [(bin_edges[i], bin_edges[i+1]) for i in range(len(bin_edges) - 1)]
    labels = [f"{a}-{b}s" for a, b in bins]

    counts = []
    videos_per_bin = {}
    for a, b in bins:
        if a == 0:
            mask = (sub['__dur_s'] >= 0) & (sub['__dur_s'] < b)
        else:
            mask = (sub['__dur_s'] >= a) & (sub['__dur_s'] < b)
        cnt = int(mask.sum())
        counts.append(cnt)
        titles = sub.loc[mask, 'title'].tolist() if 'title' in sub.columns else []
        videos_per_bin[f"{a}-{b}s"] = [str(t) for t in titles if pd.notna(t)]

    # Eliminar bins vacíos al final para gráfico más limpio
    while counts and counts[-1] == 0:
        counts.pop()
        labels.pop()
        bins.pop()

    total = sum(counts)

    plt.figure(figsize=(max(12, len(labels) * 0.8), 6))
    ax = sns.barplot(x=labels, y=counts, palette='crest')
    for i, v in enumerate(counts):
        pct = (v / total * 100) if total > 0 else 0
        ax.annotate(f"{v} ({pct:.1f}%)", (i, v), textcoords="offset points",
                     xytext=(0, 5), ha='center', fontsize=8, fontweight='bold')
    plt.xlabel('Duración (segundos)')
    plt.ylabel('Cantidad de videos')
    plt.title('Distribución por duración (intervalos de 20 s) — ¿Sabías Esto..?')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    out = OUT_DIR / "duration_20s"
    plt.savefig(f"{out}.png", dpi=150)

    with open(f"{out}.txt", 'w', encoding='utf-8') as f:
        f.write('Intervalo,Count,Percent\n')
        for lbl, cnt in zip(labels, counts):
            pct = (cnt / total * 100) if total > 0 else 0
            f.write(f"{lbl},{cnt},{pct:.2f}\n")
        f.write(f"\n# Detalle de videos por intervalo\n")
        for lbl in labels:
            key = lbl
            vids = videos_per_bin.get(key, [])
            f.write(f"\n[{lbl}] ({len(vids)} videos)\n")
            for t in vids:
                f.write(f"  - {t}\n")

    plt.close()
    print(f"[4] Duration 20s → {out}.png / .txt")

# ═══════════════════════════════════════════════════════════════════════
# 5. Caracteres en la descripción — bins personalizados
# ═══════════════════════════════════════════════════════════════════════
def analyze_description_length(df):
    desc_col = None
    for c in ("description", "descripcion", "desc"):
        if c in df.columns:
            desc_col = c
            break
    if desc_col is None:
        for c in df.columns:
            if re.search(r'descri', c, re.I):
                desc_col = c
                break
    if desc_col is None:
        print("No se encontró columna de descripción.")
        return

    sub = df.copy()
    # Tratar NaN como cadena vacía (descripción vacía = 0 caracteres)
    sub[desc_col] = sub[desc_col].fillna('')
    sub['__dlen'] = sub[desc_col].astype(str).str.len()

    bins = [
        (0, 100), (100, 250), (250, 500), (500, 750),
        (750, 1000), (1000, 1500), (1500, 2000),
        (2000, 3000), (3000, 4000), (4000, None),
    ]

    labels = []
    counts = []
    videos_per_bin = {}

    for a, b in bins:
        if b is None:
            lbl = f"{a}+"
            mask = sub['__dlen'] >= a
        elif a == 0:
            lbl = f"{a}-{b}"
            mask = (sub['__dlen'] >= a) & (sub['__dlen'] < b)
        else:
            lbl = f"{a}-{b}"
            mask = (sub['__dlen'] >= a) & (sub['__dlen'] < b)

        labels.append(lbl)
        cnt = int(mask.sum())
        counts.append(cnt)
        titles = sub.loc[mask, 'title'].tolist() if 'title' in sub.columns else []
        videos_per_bin[lbl] = [str(t) for t in titles if pd.notna(t)]

    total = sum(counts)

    plt.figure(figsize=(14, 6))
    ax = sns.barplot(x=labels, y=counts, palette='flare')
    for i, v in enumerate(counts):
        pct = (v / total * 100) if total > 0 else 0
        ax.annotate(f"{v} ({pct:.1f}%)", (i, v), textcoords="offset points",
                     xytext=(0, 5), ha='center', fontsize=9, fontweight='bold')
    plt.xlabel('Caracteres en la descripción')
    plt.ylabel('Cantidad de videos')
    plt.title('Distribución por longitud de descripción — ¿Sabías Esto..?')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    out = OUT_DIR / "description_length"
    plt.savefig(f"{out}.png", dpi=150)

    with open(f"{out}.txt", 'w', encoding='utf-8') as f:
        f.write('Intervalo,Count,Percent\n')
        for lbl, cnt in zip(labels, counts):
            pct = (cnt / total * 100) if total > 0 else 0
            f.write(f"[{lbl}],{cnt},{pct:.2f}\n")
        f.write(f"\n# Detalle de videos por intervalo\n")
        for lbl in labels:
            vids = videos_per_bin[lbl]
            f.write(f"\n[{lbl}] ({len(vids)} videos)\n")
            for t in vids:
                f.write(f"  - {t}\n")

    plt.close()
    print(f"[5] Description length → {out}.png / .txt")

# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Cargando {CSV_PATH} ...")
    df = _load()
    print(f"  {len(df)} videos cargados.\n")


    analyze_weekday_distribution(df)
    # Puedes cambiar interval_hours aquí si lo deseas
    analyze_hourly_per_day(df, interval_hours=2)
    analyze_title_word_count(df)
    analyze_duration_20s(df)
    analyze_description_length(df)

    print(f"\n✓ Todo guardado en {OUT_DIR}")
