"""
Lofi YouTube Channel Viewer
─────────────────────────────
Gráfico interactivo de línea/poligonal para explorar el rendimiento
de videos por canal.  Ejecutar:  python scripts/program.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
import glob
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg   # noqa: E402
from matplotlib.figure import Figure                               # noqa: E402
import matplotlib.dates as mdates                                  # noqa: E402
import matplotlib.ticker as mticker                                # noqa: E402
import pandas as pd                                                # noqa: E402
import numpy as np                                                 # noqa: E402

# ── Constantes ──────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FOLDER = os.path.join(BASE_DIR, "dataset", "channels")

DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
DIAS_CORTO = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

ESCALAS = {
    "Semanal":    7,
    "Bisemanal":  14,
    "Mensual":    30,
    "Trimestral": 90,
    "Todo":       None,
}

# Colores
CLR_LINE      = "#6C5CE7"
CLR_MARKER    = "#A29BFE"
CLR_HIGHLIGHT = "#E17055"
CLR_BADGE_BG  = "#FFEAA7"
CLR_BADGE_BD  = "#F39C12"
CLR_BADGE_TXT = "#C0392B"


# ── Aplicación ──────────────────────────────────────────────────────────
class LofiViewer:
    """Ventana principal del visor."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Lofi YouTube — Visor de Canal")
        self.root.geometry("1300x740")
        self.root.minsize(960, 540)

        # Estado
        self.df = None
        self.period_start = None
        self.period_days = 7
        self.points = []          # metadata de puntos dibujados
        self._day_videos = []     # videos del día seleccionado
        self._highlight = None    # artista de resaltado
        self._selected_idx = None # índice del punto seleccionado

        self._build_ui()
        self._load_csv_list()

        # Flechas del teclado
        self.root.bind("<Left>",  self._on_arrow_key)
        self.root.bind("<Right>", self._on_arrow_key)

    # ────────────────────────────────────────────────────────────────
    #  Descubrimiento automático de CSVs
    # ────────────────────────────────────────────────────────────────
    def _load_csv_list(self):
        pattern = os.path.join(CSV_FOLDER, "*.csv")
        self.csv_files = sorted(glob.glob(pattern))
        names = [os.path.splitext(os.path.basename(f))[0] for f in self.csv_files]
        self.ds_combo["values"] = names
        if names:
            self.ds_combo.current(0)
            self._on_dataset()

    # ────────────────────────────────────────────────────────────────
    #  Construcción de la interfaz
    # ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Barra superior ──
        top = ttk.Frame(self.root, padding=(8, 6))
        top.pack(fill=tk.X)

        ttk.Label(top, text="Canal:", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.ds_var = tk.StringVar()
        self.ds_combo = ttk.Combobox(
            top, textvariable=self.ds_var,
            state="readonly", width=30,
        )
        self.ds_combo.pack(side=tk.LEFT, padx=(2, 14))
        self.ds_combo.bind("<<ComboboxSelected>>", self._on_dataset)

        self.cum_var = tk.BooleanVar()
        ttk.Checkbutton(
            top, text="Vistas acumuladas",
            variable=self.cum_var, command=self._refresh,
        ).pack(side=tk.LEFT, padx=(0, 14))

        # Filtros de duración
        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        self.show_lt1 = tk.BooleanVar(value=True)
        self.show_ge1 = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            top, text="≤ 1 min",
            variable=self.show_lt1, command=self._refresh,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(
            top, text="> 1 min",
            variable=self.show_ge1, command=self._refresh,
        ).pack(side=tk.LEFT, padx=4)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(top, text="Escala:", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.sc_var = tk.StringVar(value="Semanal")
        sc = ttk.Combobox(
            top, textvariable=self.sc_var,
            values=list(ESCALAS.keys()), state="readonly", width=11,
        )
        sc.pack(side=tk.LEFT, padx=(2, 14))
        sc.bind("<<ComboboxSelected>>", self._on_scale)

        self.btn_prev = ttk.Button(top, text="◀", width=3, command=self._prev)
        self.btn_prev.pack(side=tk.LEFT)
        self.lbl_period = ttk.Label(
            top, text="", width=30, anchor=tk.CENTER,
            font=("Segoe UI", 9, "bold"),
        )
        self.lbl_period.pack(side=tk.LEFT, padx=4)
        self.btn_next = ttk.Button(top, text="▶", width=3, command=self._next)
        self.btn_next.pack(side=tk.LEFT)

        # ── Área principal (gráfico + detalles) ──
        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        # Gráfico
        chart_fr = ttk.Frame(pane)
        self.fig = Figure(figsize=(9, 5), dpi=100, facecolor="#fafafa")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_fr)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        pane.add(chart_fr, weight=3)

        # Panel de detalles
        det_fr = ttk.LabelFrame(pane, text=" Detalles del Video ", padding=8)

        self.txt = tk.Text(
            det_fr, wrap=tk.WORD, width=34, height=20,
            font=("Consolas", 10), state=tk.DISABLED,
            bg="#f9f9f9", relief=tk.GROOVE, bd=1,
        )
        self.txt.pack(fill=tk.BOTH, expand=True)

        # Selector de videos (oculto por defecto)
        self.sel_fr = ttk.Frame(det_fr)
        self.sel_lbl = ttk.Label(
            self.sel_fr, text="", font=("Segoe UI", 9, "italic"),
        )
        self.sel_lbl.pack(anchor=tk.W)
        self.sel_list = tk.Listbox(
            self.sel_fr, height=5, font=("Consolas", 9),
            activestyle="dotbox", selectbackground=CLR_LINE,
            selectforeground="white",
        )
        self.sel_list.pack(fill=tk.X)
        self.sel_list.bind("<<ListboxSelect>>", self._on_video_select)

        pane.add(det_fr, weight=1)
        self._clear_details()

    # ────────────────────────────────────────────────────────────────
    #  Carga de datos
    # ────────────────────────────────────────────────────────────────
    def _on_dataset(self, _=None):
        idx = self.ds_combo.current()
        if idx < 0:
            return
        path = self.csv_files[idx]
        try:
            df = pd.read_csv(path)
            df["fecha"] = pd.to_datetime(df["publishedAt"], utc=True)
            df["fecha"] = df["fecha"].dt.tz_localize(None)       # → tz-naive
            df = df.sort_values("fecha").reset_index(drop=True)
            df["date_only"] = df["fecha"].dt.normalize()         # medianoche
            self.df = df
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar:\n{e}")
            return

        # Comenzar en la última semana con datos
        latest = self.df["fecha"].max()
        monday = latest - timedelta(days=latest.weekday())
        self.period_start = monday.normalize()
        self._refresh()

    # ────────────────────────────────────────────────────────────────
    #  Escala y navegación
    # ────────────────────────────────────────────────────────────────
    def _on_scale(self, _=None):
        self.period_days = ESCALAS[self.sc_var.get()]
        if self.df is not None:
            if self.period_days is None:
                self.period_start = self.df["fecha"].min().normalize()
            else:
                latest = self.df["fecha"].max()
                monday = latest - timedelta(days=latest.weekday())
                self.period_start = monday.normalize()
            self._refresh()

    def _prev(self):
        if self.period_start is None or self.period_days is None:
            return
        self.period_start -= timedelta(days=self.period_days)
        self._refresh()

    def _next(self):
        if self.period_start is None or self.period_days is None:
            return
        self.period_start += timedelta(days=self.period_days)
        self._refresh()

    # ────────────────────────────────────────────────────────────────
    #  Filtrado y renderizado del gráfico
    # ────────────────────────────────────────────────────────────────
    def _period_data(self):
        """Devuelve DataFrame filtrado al período actual."""
        if self.df is None:
            return pd.DataFrame()
        df = self.df.copy()

        # Filtro de duración
        mask = pd.Series(False, index=df.index)
        if self.show_lt1.get():
            mask |= df["durationSeconds"] <= 60
        if self.show_ge1.get():
            mask |= df["durationSeconds"] > 60
        df = df[mask]

        if self.period_days is None:
            return df
        s = pd.Timestamp(self.period_start)
        e = s + timedelta(days=self.period_days)
        return df.loc[(df["fecha"] >= s) & (df["fecha"] < e)].copy()

    def _refresh(self):
        """Re-dibuja el gráfico completo."""
        if self.df is None:
            return

        self.ax.clear()
        self._highlight = None
        self._selected_idx = None
        data = self._period_data()
        cumulative = self.cum_var.get()
        scale = self.sc_var.get()

        # Actualizar label del período y botones
        nav_ok = self.period_days is not None
        self.btn_prev.config(state=tk.NORMAL if nav_ok else tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if nav_ok else tk.DISABLED)

        if nav_ok:
            end = self.period_start + timedelta(days=self.period_days - 1)
            self.lbl_period.config(
                text=f"{self.period_start.strftime('%d/%m/%Y')} — "
                     f"{end.strftime('%d/%m/%Y')}"
            )
        else:
            self.lbl_period.config(text="Todo el historial")

        # Sin datos → mensaje
        if data.empty:
            self.ax.text(
                0.5, 0.5, "Sin videos en este período",
                transform=self.ax.transAxes, ha="center", va="center",
                fontsize=14, color="#999",
            )
            self._format_empty_xaxis(scale)
            self.canvas.draw()
            self.points = []
            return

        # Calcular Y
        data = data.sort_values("fecha")
        views = data["viewCount"].astype(float)
        if cumulative:
            data["y"] = views.cumsum()
            ylabel = "Vistas acumuladas"
        else:
            data["y"] = views.values
            ylabel = "Vistas"

        xs = data["fecha"].values
        ys = data["y"].values

        # Línea conectora
        self.ax.plot(
            xs, ys, "-",
            color="#888", linewidth=0.9, alpha=0.4,
            zorder=2,
        )

        # Marcadores coloreados por duración
        dur = data["durationSeconds"].values
        colors = np.where(dur <= 60, "#e74c3c", "#2980b9")
        self.ax.scatter(
            xs, ys,
            c=colors, s=50, alpha=0.85,
            edgecolors="black", linewidths=0.4,
            zorder=3,
        )

        # Anotar días con múltiples videos
        for _, grp in data.groupby("date_only"):
            n = len(grp)
            if n > 1:
                top_y = grp["y"].max()
                mid_x = grp["fecha"].iloc[n // 2]
                self.ax.annotate(
                    f"{n} videos",
                    xy=(mid_x, top_y),
                    xytext=(0, 16), textcoords="offset points",
                    ha="center", fontsize=8, fontweight="bold",
                    color=CLR_BADGE_TXT,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        fc=CLR_BADGE_BG, ec=CLR_BADGE_BD, alpha=0.92,
                    ),
                )

        # Almacenar puntos para detección de clics
        x_nums = mdates.date2num(xs)
        self.points = []
        for i, (_, row) in enumerate(data.iterrows()):
            self.points.append({"xn": x_nums[i], "y": ys[i], "row": row})

        # Formatear ejes
        self._format_xaxis(scale)

        self.ax.set_ylabel(ylabel, fontsize=11)
        n_videos = len(data)
        self.ax.set_title(
            f"{self.ds_var.get()}  —  {ylabel}  ({n_videos} video{'s' if n_videos != 1 else ''})",
            fontsize=12, fontweight="bold",
        )
        self.ax.grid(True, alpha=0.25, linestyle="--")

        # Leyenda de colores
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
                   markersize=8, label="≤ 1 min"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2980b9",
                   markersize=8, label="> 1 min"),
        ]
        self.ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

        self.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", "."))
        )

        self.fig.tight_layout()
        self.canvas.draw()

        self.sel_fr.pack_forget()
        self._clear_details()

    def _format_xaxis(self, scale):
        """Configura las etiquetas del eje X según la escala."""
        if scale in ("Semanal", "Bisemanal"):
            days = self.period_days
            ticks = [self.period_start + timedelta(days=i) for i in range(days)]
            self.ax.set_xticks(ticks)
            labels = [
                f"{DIAS_CORTO[d.weekday()]}\n{d.strftime('%d/%m')}" for d in ticks
            ]
            self.ax.set_xticklabels(labels, fontsize=7 if days > 7 else 9)
            pad = timedelta(hours=10)
            self.ax.set_xlim(ticks[0] - pad, ticks[-1] + pad)
        elif scale == "Mensual":
            self.ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
            self.ax.xaxis.set_minor_locator(mdates.DayLocator())
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d/%m"))
            for lbl in self.ax.get_xticklabels():
                lbl.set_rotation(40)
                lbl.set_ha("right")
                lbl.set_fontsize(8)
        else:  # Trimestral o Todo
            self.ax.xaxis.set_major_locator(mdates.MonthLocator())
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            for lbl in self.ax.get_xticklabels():
                lbl.set_rotation(35)
                lbl.set_ha("right")
                lbl.set_fontsize(8)

    def _format_empty_xaxis(self, scale):
        """Eje X para períodos vacíos (semanal/bisemanal muestra días)."""
        if scale in ("Semanal", "Bisemanal") and self.period_start:
            days = self.period_days or 7
            ticks = [self.period_start + timedelta(days=i) for i in range(days)]
            self.ax.set_xticks(ticks)
            labels = [
                f"{DIAS_CORTO[d.weekday()]}\n{d.strftime('%d/%m')}" for d in ticks
            ]
            self.ax.set_xticklabels(labels, fontsize=7 if days > 7 else 9)
            pad = timedelta(hours=10)
            self.ax.set_xlim(ticks[0] - pad, ticks[-1] + pad)
        self.fig.tight_layout()

    # ────────────────────────────────────────────────────────────────
    #  Interacción por clic
    # ────────────────────────────────────────────────────────────────
    def _on_click(self, event):
        if event.inaxes != self.ax or not self.points:
            return

        cx, cy = event.xdata, event.ydata

        # Distancia normalizada para encontrar el punto más cercano
        all_x = [p["xn"] for p in self.points]
        all_y = [p["y"] for p in self.points]
        rx = (max(all_x) - min(all_x)) or 1
        ry = (max(all_y) - min(all_y)) or 1

        best_i = min(
            range(len(self.points)),
            key=lambda i: (
                ((self.points[i]["xn"] - cx) / rx) ** 2
                + ((self.points[i]["y"] - cy) / ry) ** 2
            ),
        )

        self._select_point(best_i)

    def _select_point(self, idx):
        """Selecciona el punto idx, resalta y actualiza el panel."""
        if not self.points:
            return
        idx = max(0, min(idx, len(self.points) - 1))
        self._selected_idx = idx

        clicked_date = self.points[idx]["row"]["date_only"]
        same = [p for p in self.points if p["row"]["date_only"] == clicked_date]

        self._add_highlight(
            self.points[idx]["row"]["fecha"],
            self.points[idx]["y"],
        )

        if len(same) > 1:
            self.sel_fr.pack(fill=tk.X, pady=(6, 0))
            self.sel_lbl.config(
                text=f"{len(same)} videos el "
                     f"{clicked_date.strftime('%d/%m/%Y')}:",
            )
            self.sel_list.delete(0, tk.END)
            self._day_videos = same
            for i, v in enumerate(same):
                titulo = str(v["row"].get("title", ""))[:48]
                self.sel_list.insert(tk.END, f" {i + 1}. {titulo}")
            self._show_details(same[0]["row"])
        else:
            self.sel_fr.pack_forget()
            self._show_details(same[0]["row"])

    def _on_arrow_key(self, event):
        """Navega entre puntos con las flechas ← →."""
        if not self.points:
            return
        if self._selected_idx is None:
            self._select_point(0)
            return
        delta = -1 if event.keysym == "Left" else 1
        self._select_point(self._selected_idx + delta)

    def _add_highlight(self, x, y):
        """Dibuja un círculo de resaltado sobre el punto clicado."""
        if self._highlight is not None:
            try:
                self._highlight.remove()
            except ValueError:
                pass
        self._highlight = self.ax.plot(
            x, y, "o",
            markersize=13, markerfacecolor="none",
            markeredgecolor=CLR_HIGHLIGHT, markeredgewidth=2.5,
            zorder=5,
        )[0]
        self.canvas.draw_idle()

    def _on_video_select(self, _=None):
        sel = self.sel_list.curselection()
        if sel and self._day_videos:
            v = self._day_videos[sel[0]]
            self._show_details(v["row"])
            self._add_highlight(v["row"]["fecha"], v["y"])

    # ────────────────────────────────────────────────────────────────
    #  Panel de detalles
    # ────────────────────────────────────────────────────────────────
    def _show_details(self, row):
        f = row["fecha"]
        titulo = str(row.get("title", "") or "")
        desc = str(row.get("description", "") or "")

        dur_s = int(row["durationSeconds"])
        dur_str = f"{dur_s // 60}m {dur_s % 60}s" if dur_s >= 60 else f"{dur_s}s"

        vistas = int(row["viewCount"]) if pd.notna(row["viewCount"]) else 0

        tags_raw = str(row["tags"]) if pd.notna(row["tags"]) else ""
        n_tags = len([t for t in tags_raw.split("|") if t.strip()]) if tags_raw and tags_raw != "nan" else 0

        palabras = len(titulo.split())
        chars_desc = len(desc) if desc != "nan" else 0

        lines = [
            f"  {titulo[:65]}",
            "  " + "─" * 34,
            f"  Duración:    {dur_str}",
            f"  Fecha:       {f.strftime('%Y-%m-%d')}",
            f"  Hora:        {f.strftime('%H:%M')} ({f.strftime('%I:%M %p')})",
            f"  Día:         {DIAS[f.weekday()]}",
            "  " + "─" * 34,
            f"  Vistas:      {vistas:,}",
            "  " + "─" * 34,
            f"  Palabras tít:{palabras}",
            f"  Chars desc:  {chars_desc:,}",
            f"  Nº tags:     {n_tags}",
        ]
        text = "\n".join(lines)

        self.txt.config(state=tk.NORMAL)
        self.txt.delete("1.0", tk.END)
        self.txt.insert("1.0", text)
        self.txt.config(state=tk.DISABLED)

    def _clear_details(self):
        self.txt.config(state=tk.NORMAL)
        self.txt.delete("1.0", tk.END)
        self.txt.insert(
            "1.0",
            "\n  Haz clic en un punto del\n"
            "  gráfico para ver los\n"
            "  detalles del video.\n",
        )
        self.txt.config(state=tk.DISABLED)


# ── Punto de entrada ────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = LofiViewer(root)
    root.mainloop()
