import os
from typing import Optional
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Función para cargar el CSV de videos
def load_csv(path: str) -> pd.DataFrame:
	"""Carga el CSV de videos y devuelve un DataFrame.

	Espera que exista la columna numérica `durationSeconds`.
	"""
	return pd.read_csv(path, low_memory=False)

# Gráfico de pastel con porcentaje de videos por calidad (definición)
def plot_definicion_pie(df=None, csv_path=None, root=None, output_dir=None, file_suffix=None, save=True, show=False):
	"""
	Genera un gráfico de pastel con el porcentaje de videos por la columna de calidad
	(variable `definicion`). Si no se recibe `df`, intenta cargar `csv_path` o
	`data/lofi_videos_latest.csv` relativo al `root` del proyecto.

	Retorna DataFrame con columnas: `definicion`, `count`, `percent`.
	"""
	if root is None:
		root = Path(__file__).resolve().parents[1]
	outputs_root = root / "outputs"
	outputs_root.mkdir(parents=True, exist_ok=True)
	# Usar outputs/EDA por defecto; si se pasa output_dir, usarlo tal cual
	if output_dir is None:
		script_dir = outputs_root / "EDA"
	else:
		script_dir = Path(output_dir)
	# Asegurar que exista la carpeta destino
	script_dir.mkdir(parents=True, exist_ok=True)

	# Cargar dataframe si es necesario
	if df is None:
		if csv_path is None:
			csv_path = root / 'data' / 'lofi_videos_latest.csv'
		try:
			df = pd.read_csv(csv_path, low_memory=False)
		except Exception as e:
			print(f"No se pudo cargar CSV desde {csv_path}: {e}")
			return None

	df = df.copy()

	# Detectar columna de definición/quality
	candidate_cols = [
		'definicion', 'definicion_video', 'definition', 'quality', 'video_quality', 'def', 'def_quality'
	]
	found = None
	for c in candidate_cols:
		if c in df.columns:
			found = c
			break
	if not found:
		for c in df.columns:
			if re.search(r'definici|defin|quality|qualit', c, re.I):
				found = c
				break
	if not found:
		print("No se encontró columna de 'definicion' o equivalente en el dataset.")
		return None

	# Detectar columna de duración (para poder filtrar)
	duration_cols = [
		"duration",
		"duracion",
		"duracion_iso",
		"duracion_segundos",
		"duracion_legible",
		"video_duration",
		"length",
		"duration_sec",
		"length_seconds",
		"duration_seconds",
		"duration_ms",
	]
	dur_col = None
	for c in duration_cols:
		if c in df.columns:
			dur_col = c
			break
	if not dur_col:
		for c in df.columns:
			if re.search(r'duraci|duration|length|time_length', c, re.I):
				dur_col = c
				break
	if not dur_col:
		print("No se encontró una columna de duración; no se puede filtrar por 1min/1-16min.")
		return None

	# Función local para parsear a segundos (versión reducida)
	def to_seconds(x):
		if pd.isna(x):
			return None
		if isinstance(x, (int, float)) and not isinstance(x, bool):
			val = float(x)
			if val > 1e6:
				return int(val / 1000)
			return int(val)
		s = str(x).strip()
		m_iso = re.match(r'^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$', s)
		if m_iso:
			h = int(m_iso.group(1) or 0)
			m = int(m_iso.group(2) or 0)
			ss = int(m_iso.group(3) or 0)
			return h*3600 + m*60 + ss
		if ':' in s:
			parts = [p.strip() for p in s.split(':') if p.strip()!='']
			try:
				if len(parts) == 3:
					return int(parts[0])*3600 + int(parts[1])*60 + int(float(parts[2]))
				if len(parts) == 2:
					return int(parts[0])*60 + int(float(parts[1]))
			except Exception:
				pass
		m_hms = re.search(r'(\d+)\s*h', s, re.I)
		m_min = re.search(r'(\d+)\s*m', s, re.I)
		m_sec = re.search(r'(\d+)\s*s', s, re.I)
		if m_hms or m_min or m_sec:
			return (int(m_hms.group(1)) if m_hms else 0)*3600 + (int(m_min.group(1)) if m_min else 0)*60 + (int(m_sec.group(1)) if m_sec else 0)
		m_digits = re.search(r'^(\d+(?:\.\d+)?)$', s)
		if m_digits:
			try:
				return int(float(m_digits.group(1)))
			except Exception:
				return None
		return None

	seconds_cols = {"duracion_segundos", "duration_seconds", "duration_sec", "length_seconds"}
	if dur_col in seconds_cols:
		df['__duration_s'] = pd.to_numeric(df[dur_col], errors='coerce')
		if df['__duration_s'].notna().any():
			maxv = df['__duration_s'].max()
			if pd.notna(maxv) and maxv > 1e6:
				df['__duration_s'] = (df['__duration_s'] / 1000.0).astype(float)
	else:
		df['__duration_s'] = df[dur_col].apply(to_seconds)

	df = df.dropna(subset=['__duration_s'])
	if df.empty:
		print("No hay duraciones válidas para filtrar en plot_definicion_pie.")
		return None

	# Buckets que nos interesan: 0-1min y 1-16min
	b1 = 60
	b2 = 16 * 60
	buckets = [
		("1min", lambda s: s <= b1),
		("1_16min", lambda s: (s > b1) & (s <= b2)),
	]

	results = {}
	suffix_base = f"_{re.sub(r'\\W+', '_', str(file_suffix)).strip('_')}" if file_suffix else ""

	for name, cond in buckets:
		sub = df[cond(df['__duration_s'])] if callable(cond) else df[cond]

		# Si no hay filas para el bucket, generamos un gráfico indicativo y
		# un archivo TXT vacío (mantener consistencia: siempre generar 2 imágenes)
		if sub.empty:
			print(f"No hay videos para el bucket {name}; se generará imagen indicativa.")
			results_df = pd.DataFrame({'definicion': [], 'count': [], 'percent': []})
			labels = ['No data']
			sizes = [1]
			colors = ['#cccccc']
			# Preparar sufijo y nombres de salida
			suffix = f"_{name}{suffix_base}"
			out_file = script_dir / f"definicion_pie{suffix}.png"
			txt_file = script_dir / f"definicion_counts{suffix}.txt"

			plt.figure(figsize=(8,8))
			wedges, texts, autotexts = plt.pie(
				sizes,
				labels=None,
				autopct=lambda p: '',
				colors=colors,
				startangle=90,
				wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'}
			)
			for t in autotexts:
				t.set_fontsize(11)
				t.set_weight('bold')
				t.set_color('black')
			plt.legend(wedges, labels, title="Definición", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=11, title_fontsize=12)
			plt.title(f'Distribución de videos por calidad (definición) - {name}')
			plt.tight_layout()
			if save:
				plt.savefig(out_file, dpi=150)
				print(f"Gráfico indicativo de definición guardado en {out_file}")
			with open(txt_file, 'w', encoding='utf-8') as f:
				f.write('Definicion,Count,Percent\\n')
			print(f"(Vacío) Conteos de definición guardados en {txt_file}")
			if show:
				plt.show()
			plt.close()
			results[name] = results_df
			continue

		# Normalizar columna de definición dentro del subset
		sub = sub.dropna(subset=[found])
		sub[found] = sub[found].astype(str).str.strip()
		sub = sub[sub[found] != '']
		if sub.empty:
			# Si tras normalizar no hay valores de definición, tratamos como "sin datos"
			print(f"No hay valores de definición en el bucket {name}; se generará imagen indicativa.")
			results_df = pd.DataFrame({'definicion': [], 'count': [], 'percent': []})
			labels = ['No data']
			sizes = [1]
			colors = ['#cccccc']
			suffix = f"_{name}{suffix_base}"
			out_file = script_dir / f"definicion_pie{suffix}.png"
			txt_file = script_dir / f"definicion_counts{suffix}.txt"
			plt.figure(figsize=(8,8))
			wedges, texts, autotexts = plt.pie(sizes, labels=None, autopct=lambda p: '', colors=colors, startangle=90, wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'})
			for t in autotexts:
				t.set_fontsize(11)
				t.set_weight('bold')
				t.set_color('black')
			plt.legend(wedges, labels, title="Definición", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=11, title_fontsize=12)
			plt.title(f'Distribución de videos por calidad (definición) - {name}')
			plt.tight_layout()
			if save:
				plt.savefig(out_file, dpi=150)
				print(f"Gráfico indicativo de definición guardado en {out_file}")
			with open(txt_file, 'w', encoding='utf-8') as f:
				f.write('Definicion,Count,Percent\\n')
			print(f"(Vacío) Conteos de definición guardados en {txt_file}")
			if show:
				plt.show()
			plt.close()
			results[name] = results_df
			continue

		counts = sub[found].value_counts()
		total = int(counts.sum())
		if total == 0:
			print(f"Bucket {name} sin filas válidas; se generará imagen indicativa.")
			results_df = pd.DataFrame({'definicion': [], 'count': [], 'percent': []})
			labels = ['No data']
			sizes = [1]
			colors = ['#cccccc']
			suffix = f"_{name}{suffix_base}"
			out_file = script_dir / f"definicion_pie{suffix}.png"
			txt_file = script_dir / f"definicion_counts{suffix}.txt"
			plt.figure(figsize=(8,8))
			wedges, texts, autotexts = plt.pie(sizes, labels=None, autopct=lambda p: '', colors=colors, startangle=90, wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'})
			for t in autotexts:
				t.set_fontsize(11)
				t.set_weight('bold')
				t.set_color('black')
			plt.legend(wedges, labels, title="Definición", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=11, title_fontsize=12)
			plt.title(f'Distribución de videos por calidad (definición) - {name}')
			plt.tight_layout()
			if save:
				plt.savefig(out_file, dpi=150)
				print(f"Gráfico indicativo de definición guardado en {out_file}")
			with open(txt_file, 'w', encoding='utf-8') as f:
				f.write('Definicion,Count,Percent\\n')
			print(f"(Vacío) Conteos de definición guardados en {txt_file}")
			if show:
				plt.show()
			plt.close()
			results[name] = results_df
			continue

		perc = (counts / total * 100).round(2)

		# Preparar archivos
		suffix = f"_{name}{suffix_base}"
		results_df = pd.DataFrame({'definicion': counts.index, 'count': counts.values, 'percent': perc.values})

		# Gráfico
		plt.figure(figsize=(8,8))
		labels = [str(l) for l in counts.index]
		sizes = counts.values.tolist()
		colors = plt.cm.tab20([i / max(1, len(labels)-1) for i in range(len(labels))])

		wedges, texts, autotexts = plt.pie(
			sizes,
			labels=None,
			autopct=lambda p: f"{p:.1f}%" if p > 1 else '',
			colors=colors,
			startangle=90,
			pctdistance=0.75,
			wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'}
		)
		for t in autotexts:
			t.set_fontsize(11)
			t.set_weight('bold')
			t.set_color('white')

		plt.legend(wedges, labels, title="Definición", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=11, title_fontsize=12)
		plt.title(f'Distribución de videos por calidad (definición) - {name}')
		plt.tight_layout()

		out_file = script_dir / f"definicion_pie{suffix}.png"
		txt_file = script_dir / f"definicion_counts{suffix}.txt"
		if save:
			plt.savefig(out_file, dpi=150)
			print(f"Gráfico de definición guardado en {out_file}")

		with open(txt_file, 'w', encoding='utf-8') as f:
			f.write('Definicion,Count,Percent\\n')
			for _, row in results_df.iterrows():
				f.write(f"{row['definicion']},{int(row['count'])},{row['percent']:.2f}\\n")
		print(f"Conteos de definición guardados en {txt_file}")

		if show:
			plt.show()
		plt.close()

		results[name] = results_df

	return results

# Distribución de videos por rangos fijos de duración (gráfico de pastel)
def plot_duration_pie_buckets(df, root=None, output_dir=None, file_suffix=None, labels=None):
	"""
	Genera un gráfico de pastel que muestra el % de videos del total en tres rangos:
	- (0, 1min]
	- (1min, 16min]
	- (16min, infinito)
	Guarda PNG y TXT en outputs/<script>/.
	"""
	if root is None:
		root = Path(__file__).resolve().parents[1]
	outputs_root = root / "outputs"
	outputs_root.mkdir(parents=True, exist_ok=True)
	# Usar carpeta EDA por defecto en lugar del nombre del módulo
	script_dir = outputs_root / "EDA"
	script_dir.mkdir(parents=True, exist_ok=True)
	if output_dir is not None:
		script_dir = Path(output_dir)
		script_dir.mkdir(parents=True, exist_ok=True)

	df = df.copy()

	# Buscar columna de duración (mismo heurístico que otras funciones)
	duration_cols = [
		"duration",
		"duracion",
		"duracion_iso",
		"duracion_segundos",
		"duracion_legible",
		"video_duration",
		"length",
		"duration_sec",
		"length_seconds",
		"duration_seconds",
		"duration_ms",
	]
	found = None
	for c in duration_cols:
		if c in df.columns:
			found = c
			break
	if not found:
		for c in df.columns:
			if re.search(r'duraci|duration|length|time_length', c, re.I):
				found = c
				break
	if not found:
		print("No se encontró una columna de duración para el gráfico de pastel.")
		return None

	# Función local para parsear a segundos (versión reducida)
	def to_seconds(x):
		if pd.isna(x):
			return None
		if isinstance(x, (int, float)) and not isinstance(x, bool):
			val = float(x)
			if val > 1e6:
				return int(val / 1000)
			return int(val)
		s = str(x).strip()
		m_iso = re.match(r'^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$', s)
		if m_iso:
			h = int(m_iso.group(1) or 0)
			m = int(m_iso.group(2) or 0)
			ss = int(m_iso.group(3) or 0)
			return h*3600 + m*60 + ss
		if ':' in s:
			parts = [p.strip() for p in s.split(':') if p.strip()!='']
			try:
				if len(parts) == 3:
					return int(parts[0])*3600 + int(parts[1])*60 + int(float(parts[2]))
				if len(parts) == 2:
					return int(parts[0])*60 + int(float(parts[1]))
			except Exception:
				pass
		m_hms = re.search(r'(\d+)\s*h', s, re.I)
		m_min = re.search(r'(\d+)\s*m', s, re.I)
		m_sec = re.search(r'(\d+)\s*s', s, re.I)
		if m_hms or m_min or m_sec:
			return (int(m_hms.group(1)) if m_hms else 0)*3600 + (int(m_min.group(1)) if m_min else 0)*60 + (int(m_sec.group(1)) if m_sec else 0)
		m_digits = re.search(r'^(\d+(?:\.\d+)?)$', s)
		if m_digits:
			try:
				return int(float(m_digits.group(1)))
			except Exception:
				return None
		return None

	seconds_cols = {"duracion_segundos", "duration_seconds", "duration_sec", "length_seconds"}
	if found in seconds_cols:
		df['__duration_s'] = pd.to_numeric(df[found], errors='coerce')
		if df['__duration_s'].notna().any():
			maxv = df['__duration_s'].max()
			if pd.notna(maxv) and maxv > 1e6:
				df['__duration_s'] = (df['__duration_s'] / 1000.0).astype(float)
	else:
		df['__duration_s'] = df[found].apply(to_seconds)

	df = df.dropna(subset=['__duration_s'])
	if df.empty:
		print("No hay duraciones válidas para el gráfico de pastel.")
		return None

	# Definir buckets en segundos
	b1 = 60
	b2 = 16 * 60
	# Contar respetando los intervalos: (0,1], (1,16], (16, inf)
	count1 = int(((df['__duration_s'] > 0) & (df['__duration_s'] <= b1)).sum())
	count2 = int(((df['__duration_s'] > b1) & (df['__duration_s'] <= b2)).sum())
	count3 = int((df['__duration_s'] > b2).sum())
	total = count1 + count2 + count3
	if total == 0:
		print("No hay videos para contar en los buckets.")
		return None

	if labels is None:
		labels = ['(0-1min]', '(1min-16min]', '16min+']

	sizes = [count1, count2, count3]
	perc = [s / total * 100 for s in sizes]

	# Plot pie
	plt.figure(figsize=(8,8))
	colors = plt.cm.Set2([0,1,2])
	explode = (0.03, 0.03, 0.03)
	plt.pie(sizes, labels=None, autopct=lambda p: f'{p:.1f}%' if p>0 else '', colors=colors, startangle=90, explode=explode, wedgeprops={'edgecolor':'white'})
	plt.legend(labels=[f"{lab} ({cnt}, {p:.1f}%)" for lab,cnt,p in zip(labels, sizes, perc)], loc='center left', bbox_to_anchor=(1, 0.5))
	plt.title('Distribución de videos por duración', fontsize=14)
	plt.tight_layout()

	suffix = f"_{re.sub(r'\\W+', '_', str(file_suffix)).strip('_')}" if file_suffix else ""
	out_file = script_dir / f"duration_pie_buckets{suffix}.png"
	plt.savefig(out_file, dpi=150)
	print(f"Gráfico de pastel guardado en {out_file}")

	# Guardar TXT
	txt_file = script_dir / f"duration_pie_buckets{suffix}.txt"
	with open(txt_file, 'w', encoding='utf-8') as f:
		f.write('Bucket,Count,Percent\n')
		for lab, cnt, p in zip(labels, sizes, perc):
			f.write(f"{lab},{int(cnt)},{p:.2f}\n")
	print(f"Conteos guardados en {txt_file}")
	plt.close()

	# Devolver Series con conteos y porcentajes
	res = pd.DataFrame({'label': labels, 'count': sizes, 'percent': [round(p,2) for p in perc]})
	return res

# Analizar el porcentaje de filas por año en un dataset
def analyze_yearly_percentage(df, root=None, output_dir=None, file_suffix=None):
	"""
 	Calcula el porcentaje de filas del dataset por año y guarda PNG y TXT.
	Guarda archivos en outputs/<script>/ y evita sobreescrituras usando `file_suffix`.
	"""
	if root is None:
		root = Path(__file__).resolve().parents[1]
	outputs_root = root / "outputs"
	outputs_root.mkdir(parents=True, exist_ok=True)
	# Guardar todo en outputs/EDA por defecto (misma carpeta para EDA)
	if output_dir is None:
		script_dir = outputs_root 
	else:
		script_dir = Path(output_dir)
	# Asegurar que exista la carpeta destino
	script_dir.mkdir(parents=True, exist_ok=True)

	df = df.copy()

	# Buscar columna de fecha/hora común (incluye nombres en español)
	fallback_cols = [
		"publish_time",
		"publish_date",
		"publishTimestamp",
		"publishedAt",
		"upload_time",
		"uploaded_at",
		"fecha_publicacion",
		"fecha_publicación",
		"fecha",
		"published_at",
	]
	found = None
	for c in fallback_cols:
		if c in df.columns:
			found = c
			break
	if not found:
		# Buscar por patrones más generales (incluye español)
		for c in df.columns:
			if re.search(r'publish|upload|date|time|fecha|publica|publicación|publicacion', c, re.I):
				found = c
				break
	if not found:
		print("No se encontró una columna de fecha/hora para el análisis anual.")
		return

	# Parsear fechas
	try:
		df[found] = pd.to_datetime(df[found], errors='coerce')
	except Exception:
		df[found] = pd.to_datetime(df[found].astype(str), errors='coerce')

	df = df.dropna(subset=[found])
	if df.empty:
		print("No hay datos válidos con fechas para el análisis anual.")
		return

	# Extraer año
	df['__year'] = df[found].dt.year
	counts = df['__year'].value_counts().sort_index()
	total = int(counts.sum())
	if total == 0:
		print("No hay filas con año válido para el análisis anual.")
		return

	perc = (counts / total * 100).round(2)

	# Preparar sufijo seguro
	suffix = f"_{re.sub(r'\\W+', '_', str(file_suffix)).strip('_')}" if file_suffix else ""

	# Gráfico
	plt.figure(figsize=(10,6))
	sns.barplot(x=[str(y) for y in counts.index], y=perc.values, palette='viridis')
	plt.xlabel('Año')
	plt.ylabel('% del total')
	plt.title('Porcentaje de filas del dataset por año')

	# Ajustar límite superior para dejar espacio a las anotaciones
	ymax = float(perc.values.max()) if len(perc.values) > 0 else 0.0
	plt.ylim(0, ymax * 1.2 + 2)

	# Anotar cada barra con porcentaje y conteo real
	for i, year in enumerate(counts.index):
		v = float(perc.loc[year])
		cnt = int(counts.loc[year])
		offset = ymax * 0.02 if ymax > 0 else 0.5
		plt.text(i, v + offset, f"{v:.2f}% ({cnt})", ha='center')
	plt.tight_layout()

	out_file = script_dir / f"yearly_percentage{suffix}.png"
	plt.savefig(out_file)
	print(f"Gráfico anual guardado en {out_file}")

	# Guardar txt
	txt_file = script_dir / f"yearly_percentage{suffix}.txt"
	with open(txt_file, 'w', encoding='utf-8') as f:
		f.write('Year,Count,Percent\n')
		for year, cnt in counts.items():
			p = perc.loc[year]
			f.write(f"{int(year)},{int(cnt)},{p:.2f}\n")
	print(f"Conteos anuales guardados en {txt_file}")
	plt.close()

# Conteo de videos por día de la semana (Lun-Dom)
def analyze_weekday_distribution(df, root=None, output_dir=None, file_suffix=None, canal_filter=None, duracion_filter=None):
	"""
	Cuenta la cantidad de videos publicados por día de la semana y guarda
	un histograma PNG y un TXT con los conteos en `outputs/<script>/`.
	Las etiquetas están en español: [lunes,...,domingo].

	Args:
		df: DataFrame con los datos.
		root: Directorio raíz del proyecto.
		output_dir: Directorio de salida personalizado (opcional).
		file_suffix: Sufijo para el nombre de archivo (opcional).
		canal_filter: Nombre del canal para filtrar (columna "canal").
			Si es None no se filtra por canal.
		duracion_filter: Tupla (min_seg, max_seg) para filtrar por duración
			(columna "duracion_iso"). Cualquiera de los dos valores puede ser
			None para indicar sin límite.
			Ejemplos:
				(None, 60)    → videos de menos de 1 minuto
				(3600, None)  → videos de más de 1 hora
				(60, 3600)    → videos entre 1 min y 1 hora
			Si es None no se filtra por duración.

	Returns:
		Series con conteos por día de la semana (índices en español).
	"""
	if root is None:
		root = Path(__file__).resolve().parents[1]
	outputs_root = root / "outputs"
	outputs_root.mkdir(parents=True, exist_ok=True)
	# Usar carpeta EDA por defecto; evitar crear outputs/analysis
	script_dir = outputs_root / "EDA"
	script_dir.mkdir(parents=True, exist_ok=True)
	if output_dir is not None:
		script_dir = Path(output_dir)
		script_dir.mkdir(parents=True, exist_ok=True)

	df = df.copy()

	# ------------------------------------------------------------------
	# Función interna para parsear duraciones a segundos
	# ------------------------------------------------------------------
	def _to_seconds(x):
		if pd.isna(x):
			return None
		if isinstance(x, (int, float)) and not isinstance(x, bool):
			try:
				val = float(x)
				return int(val / 1000) if val > 1e6 else int(val)
			except Exception:
				return None
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
		m_hms = re.search(r'(\d+)\s*h', s, re.I)
		m_min = re.search(r'(\d+)\s*m', s, re.I)
		m_sec = re.search(r'(\d+)\s*s', s, re.I)
		if m_hms or m_min or m_sec:
			return (int(m_hms.group(1)) if m_hms else 0) * 3600 \
				 + (int(m_min.group(1)) if m_min else 0) * 60 \
				 + (int(m_sec.group(1)) if m_sec else 0)
		m_digits = re.search(r'^(\d+(?:\.\d+)?)$', s)
		if m_digits:
			try:
				return int(float(m_digits.group(1)))
			except Exception:
				return None
		return None

	# ------------------------------------------------------------------
	# Detectar columna de duración y parsear a segundos (necesario para buckets)
	# ------------------------------------------------------------------
	duration_cols = [
		"duration",
		"duracion",
		"duracion_iso",
		"duracion_segundos",
		"duracion_legible",
		"video_duration",
		"length",
		"duration_sec",
		"length_seconds",
		"duration_seconds",
		"duration_ms",
	]
	dur_col = None
	for c in duration_cols:
		if c in df.columns:
			dur_col = c
			break
	if not dur_col:
		for c in df.columns:
			if re.search(r'duraci|duration|length|time_length', c, re.I):
				dur_col = c
				break

	if dur_col is not None:
		seconds_cols = {"duracion_segundos", "duration_seconds", "duration_sec", "length_seconds"}
		if dur_col in seconds_cols:
			df['__duration_s'] = pd.to_numeric(df[dur_col], errors='coerce')
			if df['__duration_s'].notna().any():
				maxv = df['__duration_s'].max()
				if pd.notna(maxv) and maxv > 1e6:
					df['__duration_s'] = (df['__duration_s'] / 1000.0).astype(float)
		else:
			df['__duration_s'] = df[dur_col].apply(_to_seconds)

	# ------------------------------------------------------------------
	# Filtro por canal
	# ------------------------------------------------------------------
	filter_desc_parts = []  # para título y nombre de archivo
	if canal_filter is not None:
		canal_col = "canal"
		if canal_col not in df.columns:
			# Intentar búsqueda heurística
			for c in df.columns:
				if re.search(r'canal|channel|uploader|creator', c, re.I):
					canal_col = c
					break
		if canal_col in df.columns:
			df = df[df[canal_col].astype(str).str.strip().str.lower() == str(canal_filter).strip().lower()]
			filter_desc_parts.append(f"canal={canal_filter}")
			if df.empty:
				print(f"No hay videos para el canal '{canal_filter}'.")
				return None
		else:
			print("No se encontró la columna de canal para aplicar el filtro.")
			return None

	# ------------------------------------------------------------------
	# Filtro por duración
	# ------------------------------------------------------------------
	if duracion_filter is not None:
		min_seg, max_seg = duracion_filter
		dur_col = "duracion_iso"
		if dur_col not in df.columns:
			for c in df.columns:
				if re.search(r'duraci|duration|length|time_length', c, re.I):
					dur_col = c
					break
		if dur_col in df.columns:
			# Parsear duración a segundos
			seconds_cols = {"duracion_segundos", "duration_seconds", "duration_sec", "length_seconds"}
			if dur_col in seconds_cols:
				df['__dur_filter_s'] = pd.to_numeric(df[dur_col], errors='coerce')
				if df['__dur_filter_s'].notna().any():
					maxv = df['__dur_filter_s'].max()
					if pd.notna(maxv) and maxv > 1e6:
						df['__dur_filter_s'] = (df['__dur_filter_s'] / 1000.0).astype(float)
			else:
				df['__dur_filter_s'] = df[dur_col].apply(_to_seconds)

			df = df.dropna(subset=['__dur_filter_s'])
			if df.empty:
				print("No hay duraciones válidas tras parsear para el filtro de duración.")
				return None

			if min_seg is not None:
				df = df[df['__dur_filter_s'] >= float(min_seg)]
			if max_seg is not None:
				df = df[df['__dur_filter_s'] <= float(max_seg)]

			# Construir descripción legible del filtro
			def _fmt_dur(seg):
				if seg >= 3600:
					h = seg / 3600
					return f"{h:g}h"
				if seg >= 60:
					m = seg / 60
					return f"{m:g}min"
				return f"{seg:g}s"

			if min_seg is not None and max_seg is not None:
				filter_desc_parts.append(f"dur={_fmt_dur(min_seg)}-{_fmt_dur(max_seg)}")
			elif min_seg is not None:
				filter_desc_parts.append(f"dur>={_fmt_dur(min_seg)}")
			elif max_seg is not None:
				filter_desc_parts.append(f"dur<={_fmt_dur(max_seg)}")

			if df.empty:
				rng = f"[{min_seg}, {max_seg}]"
				print(f"No hay videos en el rango de duración {rng} segundos.")
				return None
		else:
			print("No se encontró la columna de duración para aplicar el filtro.")
			return None

	# ------------------------------------------------------------------
	# Buscar columna de fecha/hora
	# ------------------------------------------------------------------
	fallback_cols = [
		"publish_time",
		"publish_date",
		"publishTimestamp",
		"publishedAt",
		"upload_time",
		"uploaded_at",
		"fecha_publicacion",
		"fecha_publicación",
		"fecha",
		"published_at",
	]
	found = None
	for c in fallback_cols:
		if c in df.columns:
			found = c
			break
	if not found:
		for c in df.columns:
			if re.search(r'publish|upload|date|time|fecha|publica|publicación|publicacion', c, re.I):
				found = c
				break
	if not found:
		print("No se encontró una columna de fecha/hora para el análisis por día de la semana.")
		return None

	# Parsear fechas
	try:
		df[found] = pd.to_datetime(df[found], errors='coerce')
	except Exception:
		df[found] = pd.to_datetime(df[found].astype(str), errors='coerce')

	df = df.dropna(subset=[found])
	if df.empty:
		print("No hay datos válidos con fechas para el análisis por día de la semana.")
		return None

	# dayofweek: Monday=0 .. Sunday=6
	df['__weekday'] = df[found].dt.dayofweek

	# Asegurar el orden Lun(0) .. Dom(6)
	counts = df['__weekday'].value_counts().reindex(range(7), fill_value=0)

	labels = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']

	# Construir título y sufijo con info de filtros
	filter_label = " | ".join(filter_desc_parts) if filter_desc_parts else ""
	title = 'Cantidad de videos publicados por día de la semana'
	if filter_label:
		title += f'\n({filter_label})'

	# Gráfico
	plt.figure(figsize=(10, 5))
	sns.barplot(x=labels, y=counts.values, palette='viridis')
	plt.xlabel('Día de la semana')
	plt.ylabel('Cantidad de videos')
	plt.title(title)
	plt.tight_layout()

	suffix = f"_{re.sub(r'\\W+', '_', str(file_suffix)).strip('_')}" if file_suffix else ""
	out_file = script_dir / f"weekday_histogram{suffix}.png"
	plt.savefig(out_file, dpi=150)
	print(f"Histograma por día de la semana guardado en {out_file}")

	# Guardar txt
	txt_file = script_dir / f"weekday_histogram{suffix}.txt"
	with open(txt_file, 'w', encoding='utf-8') as f:
		if filter_label:
			f.write(f"# Filtros: {filter_label}\n")
		f.write('Day,Count\n')
		for i, lbl in enumerate(labels):
			f.write(f"{lbl},{int(counts.get(i,0))}\n")
	print(f"Conteos por día de la semana guardados en {txt_file}")

	plt.close()

	# Devolver una Series con índices en español
	ser = pd.Series(data=counts.values, index=labels)
	return ser

# Distribución de videos por intervalos de cantidad de palabras en el título
def analyze_title_word_count(df, root=None, output_dir=None, file_suffix=None, canal_filter=None, duracion_filter=None):
	"""
	Cuenta cuántos videos caen en cada intervalo de cantidad de palabras en el título:
	[1-3], [4-6], [7-9], [10-12], [13-15], [16+] y guarda un histograma PNG
	y un TXT con los conteos en `outputs/<script>/`.

	Args:
		df: DataFrame con los datos.
		root: Directorio raíz del proyecto.
		output_dir: Directorio de salida personalizado (opcional).
		file_suffix: Sufijo para el nombre de archivo (opcional).
		canal_filter: Nombre del canal para filtrar (columna "canal").
			Si es None no se filtra por canal.
		duracion_filter: Tupla (min_seg, max_seg) para filtrar por duración
			(columna "duracion_iso"). Cualquiera de los dos valores puede ser
			None para indicar sin límite.
			Ejemplos:
				(None, 60)    → videos de menos de 1 minuto
				(3600, None)  → videos de más de 1 hora
				(60, 3600)    → videos entre 1 min y 1 hora
			Si es None no se filtra por duración.

	Returns:
		Series con conteos por intervalo de palabras en el título.
	"""
	if root is None:
		root = Path(__file__).resolve().parents[1]
	outputs_root = root / "outputs"
	outputs_root.mkdir(parents=True, exist_ok=True)
	# Usar carpeta "Title & Description" por defecto
	if output_dir is None:
		script_dir = outputs_root / "Title & Description"
	else:
		script_dir = Path(output_dir)
	script_dir.mkdir(parents=True, exist_ok=True)

	df = df.copy()

	# ------------------------------------------------------------------
	# Función interna para parsear duraciones a segundos
	# ------------------------------------------------------------------
	def _to_seconds(x):
		if pd.isna(x):
			return None
		if isinstance(x, (int, float)) and not isinstance(x, bool):
			try:
				val = float(x)
				return int(val / 1000) if val > 1e6 else int(val)
			except Exception:
				return None
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
		m_hms = re.search(r'(\d+)\s*h', s, re.I)
		m_min = re.search(r'(\d+)\s*m', s, re.I)
		m_sec = re.search(r'(\d+)\s*s', s, re.I)
		if m_hms or m_min or m_sec:
			return (int(m_hms.group(1)) if m_hms else 0) * 3600 \
				 + (int(m_min.group(1)) if m_min else 0) * 60 \
				 + (int(m_sec.group(1)) if m_sec else 0)
		m_digits = re.search(r'^(\d+(?:\.\d+)?)$', s)
		if m_digits:
			try:
				return int(float(m_digits.group(1)))
			except Exception:
				return None
		return None

	# ------------------------------------------------------------------
	# Filtro por canal
	# ------------------------------------------------------------------
	filter_desc_parts = []
	if canal_filter is not None:
		canal_col = "canal"
		if canal_col not in df.columns:
			for c in df.columns:
				if re.search(r'canal|channel|uploader|creator', c, re.I):
					canal_col = c
					break
		if canal_col in df.columns:
			df = df[df[canal_col].astype(str).str.strip().str.lower() == str(canal_filter).strip().lower()]
			filter_desc_parts.append(f"canal={canal_filter}")
			if df.empty:
				print(f"No hay videos para el canal '{canal_filter}'.")
				return None
		else:
			print("No se encontró la columna de canal para aplicar el filtro.")
			return None

	# ------------------------------------------------------------------
	# Detectar columna de duración y parsear a segundos (necesario para buckets)
	# ------------------------------------------------------------------
	duration_cols = [
		"duration",
		"duracion",
		"duracion_iso",
		"duracion_segundos",
		"duracion_legible",
		"video_duration",
		"length",
		"duration_sec",
		"length_seconds",
		"duration_seconds",
		"duration_ms",
	]
	dur_col = None
	for c in duration_cols:
		if c in df.columns:
			dur_col = c
			break
	if not dur_col:
		for c in df.columns:
			if re.search(r'duraci|duration|length|time_length', c, re.I):
				dur_col = c
				break

	if dur_col is None:
		print("No se encontró una columna de duración para el análisis de títulos.")
		return None

	seconds_cols = {"duracion_segundos", "duration_seconds", "duration_sec", "length_seconds"}
	if dur_col in seconds_cols:
		df['__duration_s'] = pd.to_numeric(df[dur_col], errors='coerce')
		if df['__duration_s'].notna().any():
			maxv = df['__duration_s'].max()
			if pd.notna(maxv) and maxv > 1e6:
				df['__duration_s'] = (df['__duration_s'] / 1000.0).astype(float)
	else:
		df['__duration_s'] = df[dur_col].apply(_to_seconds)

	df = df.dropna(subset=['__duration_s'])
	if df.empty:
		print("No hay duraciones válidas para el análisis de títulos.")
		return None

	# ------------------------------------------------------------------
	# Buscar columna de título
	# ------------------------------------------------------------------
	title_col = None
	for c in ["title", "titulo", "nombre", "name", "video_title"]:
		if c in df.columns:
			title_col = c
			break
	if title_col is None:
		for c in df.columns:
			if re.search(r'title|titulo|nombre', c, re.I):
				title_col = c
				break
	if title_col is None:
		print("No se encontró una columna de título para el análisis.")
		return None

	# Buckets objetivo
	b1 = 60
	b2 = 16 * 60
	if duracion_filter is not None:
		# Si se proporciona un filtro explícito, usar solo ese
		buckets = [("custom", lambda s: (s >= (duracion_filter[0] or -float('inf'))) & (s <= (duracion_filter[1] or float('inf'))))]
	else:
		buckets = [
			("le1min", lambda s: s <= b1),
			("1_16min", lambda s: (s > b1) & (s <= b2)),
		]

	interval_labels = ['1-3', '4-6', '7-9', '10-12', '13-15', '16+']

	def _classify(n):
		if n <= 3:
			return '1-3'
		elif n <= 6:
			return '4-6'
		elif n <= 9:
			return '7-9'
		elif n <= 12:
			return '10-12'
		elif n <= 15:
			return '13-15'
		else:
			return '16+'

	results = {}
	suffix_base = f"_{re.sub(r'\\W+', '_', str(file_suffix)).strip('_')}" if file_suffix else ""

	for name, cond in buckets:
		# Aplicar máscara sobre la columna de segundos
		try:
			mask = cond(df['__duration_s']) if callable(cond) else cond
		except Exception:
			mask = df['__duration_s'].apply(lambda x: bool(cond))
		sub = df[mask]
		if sub.empty:
			print(f"No hay videos para el bucket {name}; se guardará un gráfico vacío con ceros.")
			# Crear counts vacíos para mantener consistencia en archivos generados
			counts = pd.Series([0]*len(interval_labels), index=interval_labels)
		else:
			# Contar palabras en el título (split por espacios, ignorando vacíos)
			sub['__word_count'] = sub[title_col].apply(
				lambda x: len(str(x).split()) if pd.notna(x) and str(x).strip() != '' else 0
			)
			sub['__word_interval'] = sub['__word_count'].apply(_classify)

			cats = pd.Categorical(sub['__word_interval'], categories=interval_labels, ordered=True)
			counts = cats.value_counts().reindex(interval_labels, fill_value=0)

		# Contar palabras en el título (split por espacios, ignorando vacíos)
		sub['__word_count'] = sub[title_col].apply(
			lambda x: len(str(x).split()) if pd.notna(x) and str(x).strip() != '' else 0
		)
		sub['__word_interval'] = sub['__word_count'].apply(_classify)

		cats = pd.Categorical(sub['__word_interval'], categories=interval_labels, ordered=True)
		counts = cats.value_counts().reindex(interval_labels, fill_value=0)

		# Construir título y sufijo
		filter_label = " | ".join(filter_desc_parts) if filter_desc_parts else ""
		plot_title = f'Videos por cantidad de palabras en el título - {name}'
		if filter_label:
			plot_title += f'\n({filter_label})'

		suffix = f"_{name}{suffix_base}"

		# Gráfico
		plt.figure(figsize=(10, 6))
		bars = sns.barplot(x=list(counts.index), y=counts.values, palette='viridis')
		for i, v in enumerate(counts.values):
			bars.text(i, v + max(counts.values) * 0.01, str(int(v)), ha='center', va='bottom', fontsize=10)
		plt.xlabel('Intervalo de palabras en el título')
		plt.ylabel('Cantidad de videos')
		plt.title(plot_title)
		plt.tight_layout()

		out_png = script_dir / f"title_word_count_histogram{suffix}.png"
		plt.savefig(out_png, dpi=150)
		print(f"Histograma de palabras en título guardado en {out_png}")

		# TXT
		out_txt = script_dir / f"title_word_count_histogram{suffix}.txt"
		with open(out_txt, 'w', encoding='utf-8') as f:
			if filter_label:
				f.write(f"# Filtros: {filter_label}\n")
			f.write('Interval,Count\n')
			for lbl, cnt in counts.items():
				f.write(f"{lbl},{int(cnt)}\n")
		print(f"Conteos de palabras en título guardados en {out_txt}")

		plt.close()
		results[name] = counts

	return results

# Distribución de videos por intervalos de longitud de descripción (caracteres)
def analyze_description_length(df, root=None, output_dir=None, file_suffix=None, canal_filter=None, duracion_filter=None):
	"""
	Cuenta cuántos videos caen en cada intervalo de longitud (caracteres) de la descripción:
	[0-100], [100-250], [250-500], [500-750], [750-1000], [1000-1500],
	[1500-2000], [2000-3000], [3000-4000], [4000-5000], [5000+]
	y guarda un histograma PNG y un TXT con los conteos en `outputs/<script>/`.

	Args:
		df: DataFrame con los datos.
		root: Directorio raíz del proyecto.
		output_dir: Directorio de salida personalizado (opcional).
		file_suffix: Sufijo para el nombre de archivo (opcional).
		canal_filter: Nombre del canal para filtrar (columna "canal").
			Si es None no se filtra por canal.
		duracion_filter: Tupla (min_seg, max_seg) para filtrar por duración
			(columna "duracion_iso"). Cualquiera de los dos valores puede ser
			None para indicar sin límite.
			Ejemplos:
				(None, 60)    → videos de menos de 1 minuto
				(3600, None)  → videos de más de 1 hora
				(60, 3600)    → videos entre 1 min y 1 hora
			Si es None no se filtra por duración.

	Returns:
		Series con conteos por intervalo de longitud de descripción.
	"""
	if root is None:
		root = Path(__file__).resolve().parents[1]
	outputs_root = root / "outputs"
	outputs_root.mkdir(parents=True, exist_ok=True)
	# Guardar visualizaciones y txts dentro de outputs/Tags por defecto
	script_dir = outputs_root / "Tags"
	script_dir.mkdir(parents=True, exist_ok=True)
	if output_dir is not None:
		script_dir = Path(output_dir)
		script_dir.mkdir(parents=True, exist_ok=True)

	df = df.copy()

	# ------------------------------------------------------------------
	# Detectar columna de duración y parsear a segundos (necesario para buckets)
	# ------------------------------------------------------------------
	duration_cols = [
		"duration",
		"duracion",
		"duracion_iso",
		"duracion_segundos",
		"duracion_legible",
		"video_duration",
		"length",
		"duration_sec",
		"length_seconds",
		"duration_seconds",
		"duration_ms",
	]
	dur_col = None
	for c in duration_cols:
		if c in df.columns:
			dur_col = c
			break
	if not dur_col:
		for c in df.columns:
			if re.search(r'duraci|duration|length|time_length', c, re.I):
				dur_col = c
				break

	# Función local para parsear a segundos
	def _to_seconds(x):
		if pd.isna(x):
			return None
		if isinstance(x, (int, float)) and not isinstance(x, bool):
			val = float(x)
			if val > 1e6:
				return int(val / 1000)
			return int(val)
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
		m_hms = re.search(r'(\d+)\s*h', s, re.I)
		m_min = re.search(r'(\d+)\s*m', s, re.I)
		m_sec = re.search(r'(\d+)\s*s', s, re.I)
		if m_hms or m_min or m_sec:
			return (int(m_hms.group(1)) if m_hms else 0) * 3600 \
				 + (int(m_min.group(1)) if m_min else 0) * 60 \
				 + (int(m_sec.group(1)) if m_sec else 0)
		m_digits = re.search(r'^(\d+(?:\.\d+)?)$', s)
		if m_digits:
			try:
				return int(float(m_digits.group(1)))
			except Exception:
				return None
		return None

	if dur_col is None:
		print("No se encontró una columna de duración para el análisis de descripción.")
		return None

	seconds_cols = {"duracion_segundos", "duration_seconds", "duration_sec", "length_seconds"}
	if dur_col in seconds_cols:
		df['__duration_s'] = pd.to_numeric(df[dur_col], errors='coerce')
		if df['__duration_s'].notna().any():
			maxv = df['__duration_s'].max()
			if pd.notna(maxv) and maxv > 1e6:
				df['__duration_s'] = (df['__duration_s'] / 1000.0).astype(float)
	else:
		df['__duration_s'] = df[dur_col].apply(_to_seconds)

	df = df.dropna(subset=['__duration_s'])
	if df.empty:
		print("No hay duraciones válidas para el análisis de descripción.")
		return None

	# ------------------------------------------------------------------
	# Filtro por canal (opcional)
	# ------------------------------------------------------------------
	filter_desc_parts = []
	if canal_filter is not None:
		canal_col = "canal"
		if canal_col not in df.columns:
			for c in df.columns:
				if re.search(r'canal|channel|uploader|creator', c, re.I):
					canal_col = c
					break
		if canal_col in df.columns:
			df = df[df[canal_col].astype(str).str.strip().str.lower() == str(canal_filter).strip().lower()]
			filter_desc_parts.append(f"canal={canal_filter}")
			if df.empty:
				print(f"No hay videos para el canal '{canal_filter}'.")
				return None
		else:
			print("No se encontró la columna de canal para aplicar el filtro.")
			return None

	# Si se pasó un filtro explícito de duración, se aplicará y/o se usará como bucket "custom"
	if duracion_filter is not None:
		min_seg, max_seg = duracion_filter
		if min_seg is not None:
			df = df[df['__duration_s'] >= float(min_seg)]
		if max_seg is not None:
			df = df[df['__duration_s'] <= float(max_seg)]
		if df.empty:
			print("No hay videos tras aplicar el filtro de duración proporcionado.")
			return None
		filter_desc_parts.append(f"dur={min_seg or ''}-{max_seg or ''}")

	# ------------------------------------------------------------------
	# Buscar columna de descripción
	# ------------------------------------------------------------------
	desc_col = None
	for c in ["description", "descripcion", "descripción", "desc", "about"]:
		if c in df.columns:
			desc_col = c
			break
	if desc_col is None:
		for c in df.columns:
			if re.search(r'descri|about', c, re.I):
				desc_col = c
				break
	if desc_col is None:
		print("No se encontró una columna de descripción para el análisis.")
		return None

	# Calcular longitud en caracteres de la descripción
	df['__desc_len'] = df[desc_col].apply(
		lambda x: len(str(x)) if pd.notna(x) and str(x).strip() not in ('', 'nan') else 0
	)

	# Intervalos y etiquetas globales
	bins = [0, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, float('inf')]
	bin_labels = [
		'0-100', '100-250', '250-500', '500-750', '750-1000',
		'1000-1500', '1500-2000', '2000-3000', '3000-4000', '4000-5000', '5000+'
	]

	df['__desc_interval'] = pd.cut(
		df['__desc_len'],
		bins=bins,
		labels=bin_labels,
		right=False,
		include_lowest=True,
	)

	# Buckets por defecto si no hay duracion_filter: <=1min y (1min,16min]
	b1 = 60
	b2 = 16 * 60
	if duracion_filter is None:
		buckets = [
			("le1min", lambda s: s <= b1),
			("1_16min", lambda s: (s > b1) & (s <= b2)),
		]
	else:
		buckets = [("custom", lambda s: (s >= (duracion_filter[0] or -float('inf'))) & (s <= (duracion_filter[1] or float('inf'))))]

	results = {}
	suffix_base = f"_{re.sub(r'\\W+', '_', str(file_suffix)).strip('_')}" if file_suffix else ""

	for name, cond in buckets:
		try:
			mask = cond(df['__duration_s']) if callable(cond) else cond
		except Exception:
			mask = df['__duration_s'].apply(lambda x: bool(cond))
		sub = df[mask]

		if sub.empty:
			counts = pd.Series([0] * len(bin_labels), index=bin_labels)
			print(f"No hay videos para el bucket {name}; se guardarán ceros.")
		else:
			counts = sub['__desc_interval'].value_counts().reindex(bin_labels, fill_value=0)

		# Construir título y sufijo
		filter_label = " | ".join(filter_desc_parts) if filter_desc_parts else ""
		plot_title = f'Videos por longitud de descripción (caracteres) - {name}'
		if filter_label:
			plot_title += f'\n({filter_label})'

		suffix = f"_{name}{suffix_base}"

		# Gráfico
		plt.figure(figsize=(12, 6))
		bars = sns.barplot(x=list(counts.index), y=counts.values, palette='viridis')
		for i, v in enumerate(counts.values):
			bars.text(i, v + max(counts.values) * 0.01 if max(counts.values) > 0 else 0.1, str(int(v)), ha='center', va='bottom', fontsize=9)
		plt.xlabel('Intervalo de longitud de descripción (caracteres)')
		plt.ylabel('Cantidad de videos')
		plt.title(plot_title)
		plt.xticks(rotation=30, ha='right')
		plt.tight_layout()

		out_png = script_dir / f"description_length_histogram{suffix}.png"
		plt.savefig(out_png, dpi=150)
		print(f"Histograma de longitud de descripción guardado en {out_png}")

		# TXT
		out_txt = script_dir / f"description_length_histogram{suffix}.txt"
		with open(out_txt, 'w', encoding='utf-8') as f:
			if filter_label:
				f.write(f"# Filtros: {filter_label}\n")
			f.write('Interval,Count\n')
			for lbl, cnt in counts.items():
				f.write(f"{lbl},{int(cnt)}\n")
		print(f"Conteos de longitud de descripción guardados en {out_txt}")

		plt.close()
		results[name] = counts

	return results

# Distribución de videos por intervalos de cantidad de tags
def analyze_tags_intervals(df, root=None, output_dir=None, file_suffix=None, canal_filter=None, duracion_filter=None):
	"""
	Cuenta cuántos videos caen en cada intervalo de cantidad de tags:
	[0-5], [6-10], [11-15], [16-20], [21+] y guarda un histograma PNG
	y un TXT con los conteos en `outputs/<script>/`.

	Args:
		df: DataFrame con los datos.
		root: Directorio raíz del proyecto.
		output_dir: Directorio de salida personalizado (opcional).
		file_suffix: Sufijo para el nombre de archivo (opcional).
		canal_filter: Nombre del canal para filtrar (columna "canal").
			Si es None no se filtra por canal.
		duracion_filter: Tupla (min_seg, max_seg) para filtrar por duración
			(columna "duracion_iso"). Cualquiera de los dos valores puede ser
			None para indicar sin límite.
			Ejemplos:
				(None, 60)    → videos de menos de 1 minuto
				(3600, None)  → videos de más de 1 hora
				(60, 3600)    → videos entre 1 min y 1 hora
			Si es None no se filtra por duración.

	Returns:
		Series con conteos por intervalo de tags.
	"""
	if root is None:
		root = Path(__file__).resolve().parents[1]
	outputs_root = root / "outputs"
	outputs_root.mkdir(parents=True, exist_ok=True)
	# Guardar visualizaciones de tags dentro de outputs/Tags por defecto
	script_dir = outputs_root / "Tags"
	script_dir.mkdir(parents=True, exist_ok=True)
	if output_dir is not None:
		script_dir = Path(output_dir)
		script_dir.mkdir(parents=True, exist_ok=True)

	df = df.copy()

	# ------------------------------------------------------------------
	# Función interna para parsear duraciones a segundos
	# ------------------------------------------------------------------
	def _to_seconds(x):
		if pd.isna(x):
			return None
		if isinstance(x, (int, float)) and not isinstance(x, bool):
			try:
				val = float(x)
				return int(val / 1000) if val > 1e6 else int(val)
			except Exception:
				return None
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
		m_hms = re.search(r'(\d+)\s*h', s, re.I)
		m_min = re.search(r'(\d+)\s*m', s, re.I)
		m_sec = re.search(r'(\d+)\s*s', s, re.I)
		if m_hms or m_min or m_sec:
			return (int(m_hms.group(1)) if m_hms else 0) * 3600 \
				 + (int(m_min.group(1)) if m_min else 0) * 60 \
				 + (int(m_sec.group(1)) if m_sec else 0)
		m_digits = re.search(r'^(\d+(?:\.\d+)?)$', s)
		if m_digits:
			try:
				return int(float(m_digits.group(1)))
			except Exception:
				return None
		return None

	# ------------------------------------------------------------------
	# Detectar columna de duración y parsear a segundos (necesario para buckets)
	# ------------------------------------------------------------------
	duration_cols = [
		"duration",
		"duracion",
		"duracion_iso",
		"duracion_segundos",
		"duracion_legible",
		"video_duration",
		"length",
		"duration_sec",
		"length_seconds",
		"duration_seconds",
		"duration_ms",
	]
	dur_col = None
	for c in duration_cols:
		if c in df.columns:
			dur_col = c
			break
	if not dur_col:
		for c in df.columns:
			if re.search(r'duraci|duration|length|time_length', c, re.I):
				dur_col = c
				break
	if dur_col is None:
		print("No se encontró una columna de duración para el análisis de tags.")
		return None

	seconds_cols = {"duracion_segundos", "duration_seconds", "duration_sec", "length_seconds"}
	if dur_col in seconds_cols:
		df['__duration_s'] = pd.to_numeric(df[dur_col], errors='coerce')
		if df['__duration_s'].notna().any():
			maxv = df['__duration_s'].max()
			if pd.notna(maxv) and maxv > 1e6:
				df['__duration_s'] = (df['__duration_s'] / 1000.0).astype(float)
	else:
		df['__duration_s'] = df[dur_col].apply(_to_seconds)

	df = df.dropna(subset=['__duration_s'])
	if df.empty:
		print("No hay duraciones válidas para el análisis de tags.")
		return None

	# ------------------------------------------------------------------
	# Filtro por canal
	# ------------------------------------------------------------------
	filter_desc_parts = []
	if canal_filter is not None:
		canal_col = "canal"
		if canal_col not in df.columns:
			for c in df.columns:
				if re.search(r'canal|channel|uploader|creator', c, re.I):
					canal_col = c
					break
		if canal_col in df.columns:
			df = df[df[canal_col].astype(str).str.strip().str.lower() == str(canal_filter).strip().lower()]
			filter_desc_parts.append(f"canal={canal_filter}")
			if df.empty:
				print(f"No hay videos para el canal '{canal_filter}'.")
				return None
		else:
			print("No se encontró la columna de canal para aplicar el filtro.")
			return None

	# ------------------------------------------------------------------
	# Filtro por duración
	# ------------------------------------------------------------------
	if duracion_filter is not None:
		min_seg, max_seg = duracion_filter
		dur_col = "duracion_iso"
		if dur_col not in df.columns:
			for c in df.columns:
				if re.search(r'duraci|duration|length|time_length', c, re.I):
					dur_col = c
					break
		if dur_col in df.columns:
			seconds_cols = {"duracion_segundos", "duration_seconds", "duration_sec", "length_seconds"}
			if dur_col in seconds_cols:
				df['__dur_filter_s'] = pd.to_numeric(df[dur_col], errors='coerce')
				if df['__dur_filter_s'].notna().any():
					maxv = df['__dur_filter_s'].max()
					if pd.notna(maxv) and maxv > 1e6:
						df['__dur_filter_s'] = (df['__dur_filter_s'] / 1000.0).astype(float)
			else:
				df['__dur_filter_s'] = df[dur_col].apply(_to_seconds)

			df = df.dropna(subset=['__dur_filter_s'])
			if df.empty:
				print("No hay duraciones válidas tras parsear para el filtro de duración.")
				return None

			if min_seg is not None:
				df = df[df['__dur_filter_s'] >= float(min_seg)]
			if max_seg is not None:
				df = df[df['__dur_filter_s'] <= float(max_seg)]

			def _fmt_dur(seg):
				if seg >= 3600:
					h = seg / 3600
					return f"{h:g}h"
				if seg >= 60:
					m = seg / 60
					return f"{m:g}min"
				return f"{seg:g}s"

			if min_seg is not None and max_seg is not None:
				filter_desc_parts.append(f"dur={_fmt_dur(min_seg)}-{_fmt_dur(max_seg)}")
			elif min_seg is not None:
				filter_desc_parts.append(f"dur>={_fmt_dur(min_seg)}")
			elif max_seg is not None:
				filter_desc_parts.append(f"dur<={_fmt_dur(max_seg)}")

			if df.empty:
				rng = f"[{min_seg}, {max_seg}]"
				print(f"No hay videos en el rango de duración {rng} segundos.")
				return None
		else:
			print("No se encontró la columna de duración para aplicar el filtro.")
			return None

	# ------------------------------------------------------------------
	# Buscar columna de num_tags
	# ------------------------------------------------------------------
	tags_col = None
	for c in ["num_tags", "tag_count", "tags_count", "cantidad_tags"]:
		if c in df.columns:
			tags_col = c
			break
	if tags_col is None:
		# Buscar columna "tags" y contar elementos
		for c in df.columns:
			if re.search(r'^tags$', c, re.I):
				df['__num_tags'] = df[c].apply(
					lambda x: 0 if pd.isna(x) or str(x).strip() in ('', '[]', 'nan')
					else len(str(x).split(','))
				)
				tags_col = '__num_tags'
				break
	if tags_col is None:
		print("No se encontró una columna de tags o num_tags para el análisis.")
		return None

	df['__ntags'] = pd.to_numeric(df[tags_col], errors='coerce').fillna(0).astype(int)

	# ------------------------------------------------------------------
	# Clasificar en intervalos
	# ------------------------------------------------------------------
	interval_labels = ['0-5', '6-10', '11-15', '16-20', '21-30', '31-40', '41-60']

	def _classify(n):
		if n <= 5:
			return '0-5'
		elif n <= 10:
			return '6-10'
		elif n <= 15:
			return '11-15'
		elif n <= 20:
			return '16-20'
		elif n <= 30:
			return '21-30'
		elif n <= 40:
			return '31-40'
		else:
			return '41-60'

	df['__tag_interval'] = df['__ntags'].apply(_classify)

	cats = pd.Categorical(df['__tag_interval'], categories=interval_labels, ordered=True)

	# Construir título y sufijo base
	filter_label = " | ".join(filter_desc_parts) if filter_desc_parts else ""
	suffix_base = f"_{re.sub(r'\\W+', '_', str(file_suffix)).strip('_')}" if file_suffix else ""

	# Buckets por duración: <=1min y (1min,16min]
	b1 = 60
	b2 = 16 * 60
	buckets = [
		("le1min", lambda s: s <= b1),
		("1_16min", lambda s: (s > b1) & (s <= b2)),
	]

	results = {}
	for name, cond in buckets:
		try:
			mask = cond(df['__duration_s']) if callable(cond) else cond
		except Exception:
			mask = df['__duration_s'].apply(lambda x: bool(cond))
		sub = df[mask]
		if sub.empty:
			counts_bucket = pd.Series([0] * len(interval_labels), index=interval_labels)
			print(f"No hay videos para el bucket {name}; se guardarán ceros.")
		else:
			cats_sub = pd.Categorical(sub['__tag_interval'], categories=interval_labels, ordered=True)
			counts_bucket = cats_sub.value_counts().reindex(interval_labels, fill_value=0)

		plot_title = f'Videos por cantidad de tags - {name}'
		if filter_label:
			plot_title += f'\n({filter_label})'

		suffix = f"_{name}{suffix_base}"

		# Gráfico para el bucket
		plt.figure(figsize=(10, 6))
		bars = sns.barplot(x=list(counts_bucket.index), y=counts_bucket.values, palette='viridis')
		for i, v in enumerate(counts_bucket.values):
			bars.text(i, v + (max(counts_bucket.values) * 0.01 if max(counts_bucket.values) > 0 else 0.1), str(int(v)), ha='center', va='bottom', fontsize=10)
		plt.xlabel('Intervalo de tags')
		plt.ylabel('Cantidad de videos')
		plt.title(plot_title)
		plt.tight_layout()

		out_png = script_dir / f"tags_interval_histogram_{name}{suffix}.png"
		plt.savefig(out_png, dpi=150)
		print(f"Histograma de tags guardado en {out_png}")

		out_txt = script_dir / f"tags_interval_histogram_{name}{suffix}.txt"
		with open(out_txt, 'w', encoding='utf-8') as f:
			if filter_label:
				f.write(f"# Filtros: {filter_label}\n")
			f.write('Interval,Count\n')
			for lbl, cnt in counts_bucket.items():
				f.write(f"{lbl},{int(cnt)}\n")
		print(f"Conteos de tags guardados en {out_txt}")

		plt.close()
		results[name] = counts_bucket

	return results

# Tags más frecuentes con porcentaje de aparición
def analyze_most_frequent_tags(df, root=None, output_dir=None, file_suffix=None, top_n=40, canal_filter=None, duracion_filter=None):
	"""
	Calcula los tags individuales más frecuentes y su porcentaje de aparición
	(% de videos que contienen ese tag) y guarda un histograma horizontal PNG
	y un TXT con los resultados en `outputs/<script>/`.

	Args:
		df: DataFrame con los datos.
		root: Directorio raíz del proyecto.
		output_dir: Directorio de salida personalizado (opcional).
		file_suffix: Sufijo para el nombre de archivo (opcional).
		top_n: Número de tags top a mostrar (default: 40).
		canal_filter: Nombre del canal para filtrar (columna "canal").
			Si es None no se filtra por canal.
		duracion_filter: Tupla (min_seg, max_seg) para filtrar por duración
			(columna "duracion_iso"). Cualquiera de los dos valores puede ser
			None para indicar sin límite.
			Ejemplos:
				(None, 60)    → videos de menos de 1 minuto
				(3600, None)  → videos de más de 1 hora
				(60, 3600)    → videos entre 1 min y 1 hora
			Si es None no se filtra por duración.

	Returns:
		DataFrame con columnas [tag, count, percent] ordenado por count desc.
	"""
	if root is None:
		root = Path(__file__).resolve().parents[1]
	outputs_root = root / "outputs"
	outputs_root.mkdir(parents=True, exist_ok=True)
	script_dir = outputs_root / "Tags"
	script_dir.mkdir(parents=True, exist_ok=True)
	if output_dir is not None:
		script_dir = Path(output_dir)
		script_dir.mkdir(parents=True, exist_ok=True)

	df = df.copy()

	# ------------------------------------------------------------------
	# Función interna para parsear duraciones a segundos
	# ------------------------------------------------------------------
	def _to_seconds(x):
		if pd.isna(x):
			return None
		if isinstance(x, (int, float)) and not isinstance(x, bool):
			try:
				val = float(x)
				return int(val / 1000) if val > 1e6 else int(val)
			except Exception:
				return None
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
		m_hms = re.search(r'(\d+)\s*h', s, re.I)
		m_min = re.search(r'(\d+)\s*m', s, re.I)
		m_sec = re.search(r'(\d+)\s*s', s, re.I)
		if m_hms or m_min or m_sec:
			return (int(m_hms.group(1)) if m_hms else 0) * 3600 \
				 + (int(m_min.group(1)) if m_min else 0) * 60 \
				 + (int(m_sec.group(1)) if m_sec else 0)
		m_digits = re.search(r'^(\d+(?:\.\d+)?)$', s)
		if m_digits:
			try:
				return int(float(m_digits.group(1)))
			except Exception:
				return None
		return None

	# ------------------------------------------------------------------
	# Filtro por canal
	# ------------------------------------------------------------------
	filter_desc_parts = []
	if canal_filter is not None:
		canal_col = "canal"
		if canal_col not in df.columns:
			for c in df.columns:
				if re.search(r'canal|channel|uploader|creator', c, re.I):
					canal_col = c
					break
		if canal_col in df.columns:
			df = df[df[canal_col].astype(str).str.strip().str.lower() == str(canal_filter).strip().lower()]
			filter_desc_parts.append(f"canal={canal_filter}")
			if df.empty:
				print(f"No hay videos para el canal '{canal_filter}'.")
				return None
		else:
			print("No se encontró la columna de canal para aplicar el filtro.")
			return None

	# ------------------------------------------------------------------
	# Filtro por duración
	# ------------------------------------------------------------------
	if duracion_filter is not None:
		min_seg, max_seg = duracion_filter
		dur_col = "duracion_iso"
		if dur_col not in df.columns:
			for c in df.columns:
				if re.search(r'duraci|duration|length|time_length', c, re.I):
					dur_col = c
					break
		if dur_col in df.columns:
			seconds_cols = {"duracion_segundos", "duration_seconds", "duration_sec", "length_seconds"}
			if dur_col in seconds_cols:
				df['__dur_filter_s'] = pd.to_numeric(df[dur_col], errors='coerce')
				if df['__dur_filter_s'].notna().any():
					maxv = df['__dur_filter_s'].max()
					if pd.notna(maxv) and maxv > 1e6:
						df['__dur_filter_s'] = (df['__dur_filter_s'] / 1000.0).astype(float)
			else:
				df['__dur_filter_s'] = df[dur_col].apply(_to_seconds)

			df = df.dropna(subset=['__dur_filter_s'])
			if df.empty:
				print("No hay duraciones válidas tras parsear para el filtro de duración.")
				return None

			if min_seg is not None:
				df = df[df['__dur_filter_s'] >= float(min_seg)]
			if max_seg is not None:
				df = df[df['__dur_filter_s'] <= float(max_seg)]

			def _fmt_dur(seg):
				if seg >= 3600:
					h = seg / 3600
					return f"{h:g}h"
				if seg >= 60:
					m = seg / 60
					return f"{m:g}min"
				return f"{seg:g}s"

			if min_seg is not None and max_seg is not None:
				filter_desc_parts.append(f"dur={_fmt_dur(min_seg)}-{_fmt_dur(max_seg)}")
			elif min_seg is not None:
				filter_desc_parts.append(f"dur>={_fmt_dur(min_seg)}")
			elif max_seg is not None:
				filter_desc_parts.append(f"dur<={_fmt_dur(max_seg)}")

			if df.empty:
				rng = f"[{min_seg}, {max_seg}]"
				print(f"No hay videos en el rango de duración {rng} segundos.")
				return None
		else:
			print("No se encontró la columna de duración para aplicar el filtro.")
			return None

	# ------------------------------------------------------------------
	# Buscar columna de tags
	# ------------------------------------------------------------------
	tags_col = None
	for c in df.columns:
		if re.search(r'^tags$', c, re.I):
			tags_col = c
			break
	if tags_col is None:
		print("No se encontró la columna 'tags' para el análisis de tags frecuentes.")
		return None

	# ------------------------------------------------------------------
	# Generar resultados por buckets de duración (<=1min y (1min,16min])
	# Si no se detecta columna de duración, generar un único reporte global
	# ------------------------------------------------------------------
	# Intentar detectar columna de duración (si no existe ya como '__duration_s')
	if '__duration_s' not in df.columns:
		duration_cols = [
			"duration",
			"duracion",
			"duracion_iso",
			"duracion_segundos",
			"duracion_legible",
			"video_duration",
			"length",
			"duration_sec",
			"length_seconds",
			"duration_seconds",
			"duration_ms",
		]
		dur_col = None
		for c in duration_cols:
			if c in df.columns:
				dur_col = c
				break
		if not dur_col:
			for c in df.columns:
				if re.search(r'duraci|duration|length|time_length', c, re.I):
					dur_col = c
					break
		if dur_col is not None:
			seconds_cols = {"duracion_segundos", "duration_seconds", "duration_sec", "length_seconds"}
			if dur_col in seconds_cols:
				df['__duration_s'] = pd.to_numeric(df[dur_col], errors='coerce')
				if df['__duration_s'].notna().any():
					maxv = df['__duration_s'].max()
					if pd.notna(maxv) and maxv > 1e6:
						df['__duration_s'] = (df['__duration_s'] / 1000.0).astype(float)
			else:
				df['__duration_s'] = df[dur_col].apply(_to_seconds)

	# Definir buckets
	b1 = 60
	b2 = 16 * 60
	if duracion_filter is None and '__duration_s' in df.columns:
		buckets = [
			("le1min", lambda s: s <= b1),
			("1_16min", lambda s: (s > b1) & (s <= b2)),
		]
	elif duracion_filter is not None:
		buckets = [("custom", lambda s: (s >= (duracion_filter[0] or -float('inf'))) & (s <= (duracion_filter[1] or float('inf'))))]
	else:
		buckets = [("all", None)]

	results = {}
	suffix_base = f"_{re.sub(r'\\W+', '_', str(file_suffix)).strip('_')}" if file_suffix else ""

	for name, cond in buckets:
		# Construir sub-dataframe según el bucket
		if cond is None:
			sub = df.copy()
		else:
			try:
				mask = cond(df['__duration_s']) if callable(cond) else cond
			except Exception:
				mask = df['__duration_s'].apply(lambda x: bool(cond))
			sub = df[mask]

		total_videos = len(sub)
		if total_videos == 0:
			print(f"No hay videos para el bucket {name}; se omitirá.")
			results[name] = pd.DataFrame(columns=['tag','count','videos_con_tag','percent'])
			continue

		# Parsear tags en el subset
		all_tags = []
		for raw in sub[tags_col].dropna():
			raw_str = str(raw).strip()
			if raw_str in ('', '[]', 'nan'):
				continue
			for tag in re.split(r'[,\|]', raw_str):
				cleaned = tag.strip().lower()
				if cleaned:
					all_tags.append(cleaned)

		if not all_tags:
			print(f"No se encontraron tags válidos en el bucket {name}.")
			results[name] = pd.DataFrame(columns=['tag','count','videos_con_tag','percent'])
			continue

		tag_counts = pd.Series(all_tags).value_counts()

		video_tag_sets = sub[tags_col].dropna().apply(
			lambda x: set(
				t.strip().lower() for t in re.split(r'[,\|]', str(x))
				if t.strip().lower() and str(x).strip() not in ('', '[]', 'nan')
			)
		)
		_tag_video_count = {}
		for tag_set in video_tag_sets:
			for tag in tag_set:
				_tag_video_count[tag] = _tag_video_count.get(tag, 0) + 1

		res_df = pd.DataFrame({
			'tag': tag_counts.index,
			'count': tag_counts.values,
		})
		res_df['videos_con_tag'] = res_df['tag'].map(_tag_video_count).fillna(0).astype(int)
		res_df['percent'] = (res_df['videos_con_tag'] / total_videos * 100).round(2)
		res_df = res_df.sort_values('count', ascending=False).reset_index(drop=True)

		# Título y sufijo
		filter_label = " | ".join(filter_desc_parts) if filter_desc_parts else ""
		title = f'Top {top_n} tags más frecuentes (% de videos) - {name}'
		if filter_label:
			title += f'\n({filter_label})'

		suffix = f"_{name}{suffix_base}"

		# Gráfico horizontal (ajustar dimensiones/márgenes según longitud de etiquetas)
		top = res_df.head(top_n)
		max_label_len = int(top['tag'].astype(str).map(len).max()) if not top.empty else 10
		fig_height = max(6, len(top) * 0.35)
		fig_width = max(12, min(40, 12 + max_label_len * 0.12))
		# margen izquierdo en función de la longitud máxima de etiqueta
		if max_label_len <= 30:
			left = 0.22
		elif max_label_len <= 60:
			left = 0.32
		else:
			left = 0.42

		fig = plt.figure(figsize=(fig_width, fig_height))
		ax = fig.subplots()
		bars = sns.barplot(
			x=top['percent'].values,
			y=top['tag'].values,
			palette='viridis',
			orient='h',
			ax=ax,
		)
		# Ajustar límites y anotar
		xmax = max(top['percent'].max() if not top.empty else 1.0, 1.0) * 1.12
		ax.set_xlim(0, xmax)
		fontsize = 8 if len(top) <= 20 else 7 if len(top) <= 40 else 6
		for i, (pct, cnt) in enumerate(zip(top['percent'].values, top['count'].values)):
			ax.text(pct + xmax * 0.01, i, f"{pct:.1f}% ({cnt})", va='center', fontsize=fontsize)
		ax.set_xlabel('% de videos que contienen el tag')
		ax.set_ylabel('Tag')
		ax.set_title(title)
		plt.subplots_adjust(left=left, right=0.98, top=0.95)

		out_png = script_dir / f"most_frequent_tags_{name}{suffix}.png"
		plt.savefig(out_png, dpi=150)
		print(f"Gráfico de tags frecuentes guardado en {out_png}")

		# TXT
		out_txt = script_dir / f"most_frequent_tags_{name}{suffix}.txt"
		with open(out_txt, 'w', encoding='utf-8') as f:
			f.write(f"# Total videos: {total_videos}\n")
			f.write(f"# Tags únicos encontrados: {len(tag_counts)}\n")
			if filter_label:
				f.write(f"# Filtros: {filter_label}\n")
			f.write('Tag,Apariciones,VideosConTag,Percent\n')
			for _, row in res_df.iterrows():
				f.write(f"{row['tag']},{int(row['count'])},{int(row['videos_con_tag'])},{row['percent']:.2f}\n")
		print(f"Tags frecuentes guardados en {out_txt}")

		plt.close()
		results[name] = res_df

	# Devolver DataFrame único si solo hay un bucket, sino diccionario por bucket
	if len(results) == 1:
		return list(results.values())[0]
	return results

# Conteo de publicaciones por intervalos de 2 horas
def analyze_publications_2h_intervals(df, root=None, output_dir=None, file_suffix=None, tiempo=2, canal_filter=None, duracion_filter=None, por_dia=False):
	"""
	Cuenta cuántos videos se publicaron en intervalos de `tiempo` horas
	(donde `tiempo` puede ser 0.5, 1 o 2) y guarda un histograma PNG y un TXT
	en `outputs/<script>/`.

	Args:
		df: DataFrame con los datos
		root, output_dir, file_suffix: como en otras funciones
		tiempo: tamaño del intervalo en horas. Valores permitidos: 0.5, 1, 2
		canal_filter: Nombre del canal para filtrar (columna "canal").
			Si es None no se filtra por canal.
		duracion_filter: Tupla (min_seg, max_seg) para filtrar por duración
			(columna "duracion_iso"). Cualquiera de los dos valores puede ser
			None para indicar sin límite.
			Ejemplos:
				(None, 60)    → videos de menos de 1 minuto
				(3600, None)  → videos de más de 1 hora
				(60, 3600)    → videos entre 1 min y 1 hora
			Si es None no se filtra por duración.
		por_dia: Si es True, crea una subcarpeta y genera un análisis
			separado para cada día de la semana (lunes a domingo),
			respetando los filtros de canal y duración.
	"""

	# Validar parametro tiempo
	if float(tiempo) not in {0.5, 1.0, 2.0}:
		print("Parámetro 'tiempo' inválido. Use 0.5, 1 o 2 (horas).")
		return None

	bin_size = float(tiempo)

	if root is None:
		root = Path(__file__).resolve().parents[1]
	outputs_root = root / "outputs"
	outputs_root.mkdir(parents=True, exist_ok=True)
	script_dir = outputs_root / "Análisis del día"
	script_dir.mkdir(parents=True, exist_ok=True)
	if output_dir is not None:
		script_dir = Path(output_dir)
		script_dir.mkdir(parents=True, exist_ok=True)

	df = df.copy()

	# ------------------------------------------------------------------
	# Función interna para parsear duraciones a segundos
	# ------------------------------------------------------------------
	def _to_seconds(x):
		if pd.isna(x):
			return None
		if isinstance(x, (int, float)) and not isinstance(x, bool):
			try:
				val = float(x)
				return int(val / 1000) if val > 1e6 else int(val)
			except Exception:
				return None
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
		m_hms = re.search(r'(\d+)\s*h', s, re.I)
		m_min = re.search(r'(\d+)\s*m', s, re.I)
		m_sec = re.search(r'(\d+)\s*s', s, re.I)
		if m_hms or m_min or m_sec:
			return (int(m_hms.group(1)) if m_hms else 0) * 3600 \
				 + (int(m_min.group(1)) if m_min else 0) * 60 \
				 + (int(m_sec.group(1)) if m_sec else 0)
		m_digits = re.search(r'^(\d+(?:\.\d+)?)$', s)
		if m_digits:
			try:
				return int(float(m_digits.group(1)))
			except Exception:
				return None
		return None

	# ------------------------------------------------------------------
	# Filtro por canal
	# ------------------------------------------------------------------
	filter_desc_parts = []  # para título y nombre de archivo
	if canal_filter is not None:
		canal_col = "canal"
		if canal_col not in df.columns:
			# Intentar búsqueda heurística
			for c in df.columns:
				if re.search(r'canal|channel|uploader|creator', c, re.I):
					canal_col = c
					break
		if canal_col in df.columns:
			df = df[df[canal_col].astype(str).str.strip().str.lower() == str(canal_filter).strip().lower()]
			filter_desc_parts.append(f"canal={canal_filter}")
			if df.empty:
				print(f"No hay videos para el canal '{canal_filter}'.")
				return None
		else:
			print("No se encontró la columna de canal para aplicar el filtro.")
			return None

	# ------------------------------------------------------------------
	# Filtro por duración
	# ------------------------------------------------------------------
	if duracion_filter is not None:
		min_seg, max_seg = duracion_filter
		dur_col = "duracion_iso"
		if dur_col not in df.columns:
			for c in df.columns:
				if re.search(r'duraci|duration|length|time_length', c, re.I):
					dur_col = c
					break
		if dur_col in df.columns:
			# Parsear duración a segundos
			seconds_cols = {"duracion_segundos", "duration_seconds", "duration_sec", "length_seconds"}
			if dur_col in seconds_cols:
				df['__dur_filter_s'] = pd.to_numeric(df[dur_col], errors='coerce')
				if df['__dur_filter_s'].notna().any():
					maxv = df['__dur_filter_s'].max()
					if pd.notna(maxv) and maxv > 1e6:
						df['__dur_filter_s'] = (df['__dur_filter_s'] / 1000.0).astype(float)
			else:
				df['__dur_filter_s'] = df[dur_col].apply(_to_seconds)

			df = df.dropna(subset=['__dur_filter_s'])
			if df.empty:
				print("No hay duraciones válidas tras parsear para el filtro de duración.")
				return None

			if min_seg is not None:
				df = df[df['__dur_filter_s'] >= float(min_seg)]
			if max_seg is not None:
				df = df[df['__dur_filter_s'] <= float(max_seg)]

			# Construir descripción legible del filtro
			def _fmt_dur(seg):
				if seg >= 3600:
					h = seg / 3600
					return f"{h:g}h"
				if seg >= 60:
					m = seg / 60
					return f"{m:g}min"
				return f"{seg:g}s"

			if min_seg is not None and max_seg is not None:
				filter_desc_parts.append(f"dur={_fmt_dur(min_seg)}-{_fmt_dur(max_seg)}")
			elif min_seg is not None:
				filter_desc_parts.append(f"dur>={_fmt_dur(min_seg)}")
			elif max_seg is not None:
				filter_desc_parts.append(f"dur<={_fmt_dur(max_seg)}")

			if df.empty:
				rng = f"[{min_seg}, {max_seg}]"
				print(f"No hay videos en el rango de duración {rng} segundos.")
				return None
		else:
			print("No se encontró la columna de duración para aplicar el filtro.")
			return None

	# ------------------------------------------------------------------
	# Buscar columna de fecha/hora (mismo heurístico que otras funciones)
	fallback_cols = [
		"publish_time",
		"publish_date",
		"publishTimestamp",
		"publishedAt",
		"upload_time",
		"uploaded_at",
		"fecha_publicacion",
		"fecha_publicación",
		"fecha",
		"published_at",
	]
	found = None
	for c in fallback_cols:
		if c in df.columns:
			found = c
			break
	if not found:
		for c in df.columns:
			if re.search(r'publish|upload|date|time|fecha|publica|publicación|publicacion', c, re.I):
				found = c
				break
	if not found:
		print("No se encontró una columna de fecha/hora para el análisis de intervalos de 2h.")
		return None

	# Parsear fechas
	try:
		df[found] = pd.to_datetime(df[found], errors='coerce')
	except Exception:
		df[found] = pd.to_datetime(df[found].astype(str), errors='coerce')

	df = df.dropna(subset=[found])
	if df.empty:
		print("No hay datos válidos con fechas para el análisis de intervalos de 2h.")
		return None

	# Extraer hora fraccionaria (horas con minutos/segundos) para soportar intervalos < 1h
	df['__hour_frac'] = (
		df[found].dt.hour.fillna(0).astype(float)
		+ df[found].dt.minute.fillna(0).astype(float) / 60.0
		+ df[found].dt.second.fillna(0).astype(float) / 3600.0
	)

	# Construir bins desde 0 hasta 24 con paso bin_size
	n_steps = int(round(24.0 / bin_size))
	bins = [round(i * bin_size, 6) for i in range(n_steps + 1)]

	# Etiquetas legibles (minutos si bin_size < 1, horas si >=1)
	def _label(a, b):
		if bin_size < 1.0:
			a_min = int(a * 60)
			b_min = int(b * 60)
			return f"{a_min}-{b_min}min"
		else:
			a_h = int(a)
			b_h = int(b)
			return f"{a_h}-{b_h}h"

	labels = [_label(bins[i], bins[i+1]) for i in range(len(bins) - 1)]

	# ------------------------------------------------------------------
	# Modo por_dia: analizar cada día de la semana por separado
	# ------------------------------------------------------------------
	if por_dia:
		suffix = f"_{re.sub(r'\\W+', '_', str(file_suffix)).strip('_')}" if file_suffix else ""
		# If an explicit output_dir was provided, use it directly. Otherwise use outputs/Análisis del día
		if output_dir is not None:
			por_dia_dir = Path(output_dir)
			por_dia_dir.mkdir(parents=True, exist_ok=True)
		else:
			por_dia_dir = outputs_root / "Análisis del día"
			por_dia_dir.mkdir(parents=True, exist_ok=True)

		# Preparar subcarpetas para <=1min y (1min,16min] dentro del directorio por_dia
		long_dir = por_dia_dir / "1_16min"
		short_dir = por_dia_dir / "menos_1min"
		long_dir.mkdir(parents=True, exist_ok=True)
		short_dir.mkdir(parents=True, exist_ok=True)

		dias_semana = {
			0: 'lunes',
			1: 'martes',
			2: 'miercoles',
			3: 'jueves',
			4: 'viernes',
			5: 'sabado',
			6: 'domingo',
		}
		dias_labels = {
			0: 'lunes',
			1: 'martes',
			2: 'miércoles',
			3: 'jueves',
			4: 'viernes',
			5: 'sábado',
			6: 'domingo',
		}

		df['__weekday'] = df[found].dt.dayofweek  # Monday=0 .. Sunday=6
		filter_label = " | ".join(filter_desc_parts) if filter_desc_parts else ""
		all_results = {}



		# Detectar columna de duración disponible (si existe)
		dur_col = None
		for c in df.columns:
			if re.search(r'duraci|duration|length|time_length', c, re.I):
				dur_col = c
				break
		# Calcular segundos en '__dur_s' si encontramos columna de duración
		if dur_col:
			seconds_cols = {"duracion_segundos", "duration_seconds", "duration_sec", "length_seconds"}
			if dur_col in seconds_cols:
				df['__dur_s'] = pd.to_numeric(df[dur_col], errors='coerce')
				if df['__dur_s'].notna().any():
					maxv = df['__dur_s'].max()
					if pd.notna(maxv) and maxv > 1e6:
						df['__dur_s'] = (df['__dur_s'] / 1000.0).astype(float)
			else:
				df['__dur_s'] = df[dur_col].apply(_to_seconds)

		for day_num, day_name_file in dias_semana.items():
			day_label = dias_labels[day_num]
			df_day = df[df['__weekday'] == day_num]

			if df_day.empty:
				print(f"  Sin datos para {day_label}, se omite.")
				continue

			cats_day = pd.cut(df_day['__hour_frac'], bins=bins, labels=labels, include_lowest=True, right=False)
			counts_day = cats_day.value_counts().reindex(labels, fill_value=0)

			# Título
			title_day = f'Publicaciones por intervalos ({day_label})'
			if filter_label:
				title_day += f'\n({filter_label})'

			# Gráfico
			plt.figure(figsize=(12, 5))
			sns.barplot(x=list(counts_day.index), y=counts_day.values, palette='viridis')
			plt.xticks(rotation=45, ha='right')
			plt.xlabel('Intervalo horario')
			plt.ylabel('Cantidad de videos publicados')
			plt.title(title_day)
			plt.tight_layout()

			out_png = por_dia_dir / f"hour_2h_histogram_{day_name_file}.png"
			plt.savefig(out_png, dpi=150)
			print(f"  Histograma {day_label} guardado en {out_png}")

			# TXT
			out_txt = por_dia_dir / f"hour_2h_histogram_{day_name_file}.txt"
			with open(out_txt, 'w', encoding='utf-8') as f:
				if filter_label:
					f.write(f"# Filtros: {filter_label}\n")
				f.write(f"# Día: {day_label}\n")
				f.write('Interval,Count\n')
				for lbl, cnt in counts_day.items():
					f.write(f"{lbl},{int(cnt)}\n")
			print(f"  Conteos {day_label} guardados en {out_txt}")

			plt.close()
			all_results[day_label] = counts_day

			# --- Guardar análisis para videos entre 1min y 16min en long_dir ---
			try:
				df_day_mid = df_day.copy()
				if '__dur_s' in df_day_mid.columns:
					df_day_mid = df_day_mid[(df_day_mid['__dur_s'] > 60) & (df_day_mid['__dur_s'] <= 16 * 60)]
				else:
					df_day_mid = df_day_mid.iloc[0:0]
			except Exception:
				df_day_mid = df_day.iloc[0:0]

			cats_day_mid = pd.cut(df_day_mid['__hour_frac'] if not df_day_mid.empty else pd.Series([], dtype=float), bins=bins, labels=labels, include_lowest=True, right=False)
			counts_day_mid = cats_day_mid.value_counts().reindex(labels, fill_value=0)

			plt.figure(figsize=(12,5))
			sns.barplot(x=list(counts_day_mid.index), y=counts_day_mid.values, palette='rocket')
			plt.xticks(rotation=45, ha='right')
			plt.xlabel('Intervalo horario')
			plt.ylabel('Cantidad de videos publicados')
			plt.title(f'Publicaciones por intervalos (1-16min) - {day_label}')
			plt.tight_layout()

			out_png_mid = long_dir / f"hour_2h_histogram_{day_name_file}.png"
			plt.savefig(out_png_mid, dpi=150)
			print(f"  Histograma 1-16min {day_label} guardado en {out_png_mid}")

			out_txt_mid = long_dir / f"hour_2h_histogram_{day_name_file}.txt"
			with open(out_txt_mid, 'w', encoding='utf-8') as f:
				if filter_label:
					f.write(f"# Filtros: {filter_label}\n")
				f.write(f"# Día: {day_label}\n")
				f.write('Interval,Count\n')
				for lbl, cnt in counts_day_mid.items():
					f.write(f"{lbl},{int(cnt)}\n")
			print(f"  Conteos 1-16min {day_label} guardados en {out_txt_mid}")
			plt.close()

			# --- Guardar análisis para videos <=1min en short_dir ---
			try:
				df_day_short = df_day.copy()
				if '__dur_s' in df_day_short.columns:
					df_day_short = df_day_short[df_day_short['__dur_s'] <= 60]
				else:
					df_day_short = df_day_short.iloc[0:0]
			except Exception:
				df_day_short = df_day.iloc[0:0]

			cats_day_short = pd.cut(df_day_short['__hour_frac'] if not df_day_short.empty else pd.Series([], dtype=float), bins=bins, labels=labels, include_lowest=True, right=False)
			counts_day_short = cats_day_short.value_counts().reindex(labels, fill_value=0)

			plt.figure(figsize=(12,5))
			sns.barplot(x=list(counts_day_short.index), y=counts_day_short.values, palette='mako')
			plt.xticks(rotation=45, ha='right')
			plt.xlabel('Intervalo horario')
			plt.ylabel('Cantidad de videos publicados')
			plt.title(f'Publicaciones por intervalos (<=1min) - {day_label}')
			plt.tight_layout()

			out_png_short = short_dir / f"hour_2h_histogram_{day_name_file}.png"
			plt.savefig(out_png_short, dpi=150)
			print(f"  Histograma <=1min {day_label} guardado en {out_png_short}")

			out_txt_short = short_dir / f"hour_2h_histogram_{day_name_file}.txt"
			with open(out_txt_short, 'w', encoding='utf-8') as f:
				if filter_label:
					f.write(f"# Filtros: {filter_label}\n")
				f.write(f"# Día: {day_label}\n")
				f.write('Interval,Count\n')
				for lbl, cnt in counts_day_short.items():
					f.write(f"{lbl},{int(cnt)}\n")
			print(f"  Conteos <=1min {day_label} guardados en {out_txt_short}")
			plt.close()

		print(f"\nAnálisis por día completado. Carpeta: {por_dia_dir}")
		return all_results

	# ------------------------------------------------------------------
	# Modo normal (por_dia=False): análisis agregado
	# ------------------------------------------------------------------

	# pd.cut sobre la hora fraccionaria; usar right=False para [a,b)
	cats = pd.cut(df['__hour_frac'], bins=bins, labels=labels, include_lowest=True, right=False)

	counts = cats.value_counts().reindex(labels, fill_value=0)

	# Construir título con info de filtros
	filter_label = " | ".join(filter_desc_parts) if filter_desc_parts else ""
	title = 'Publicaciones por intervalos de 2 horas'
	if filter_label:
		title += f'\n({filter_label})'

	# Gráfico
	plt.figure(figsize=(12,5))
	sns.barplot(x=list(counts.index), y=counts.values, palette='viridis')
	plt.xticks(rotation=45, ha='right')
	plt.xlabel('Intervalo horario')
	plt.ylabel('Cantidad de videos publicados')
	plt.title(title)
	plt.tight_layout()

	suffix = f"_{re.sub(r'\\W+', '_', str(file_suffix)).strip('_')}" if file_suffix else ""
	out_file = script_dir / f"hour_2h_histogram{suffix}.png"
	plt.savefig(out_file, dpi=150)
	print(f"Histograma 2h guardado en {out_file}")

	# Guardar conteos a txt
	txt_file = script_dir / f"hour_2h_histogram{suffix}.txt"
	with open(txt_file, 'w', encoding='utf-8') as f:
		if filter_label:
			f.write(f"# Filtros: {filter_label}\n")
		f.write('Interval,Count\n')
		for lbl, cnt in counts.items():
			f.write(f"{lbl},{int(cnt)}\n")
	print(f"Conteos 2h guardados en {txt_file}")

	plt.close()
	return counts



# Función principal para ejecutar el análisis
def main() -> None:
	root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
	csv_path = os.path.join(root, 'dataset', 'videos.csv')

	if not os.path.exists(csv_path):
		raise FileNotFoundError(f'No se encontró el dataset en: {csv_path}')

	print(f'Cargando dataset desde: {csv_path}')
	df = load_csv(csv_path)

	#--------------------------------------------------
 	# EDA GENERAL
  	#--------------------------------------------------
	eda_dir = os.path.join(root, 'outputs', 'EDA')
	os.makedirs(eda_dir, exist_ok=True)

	# Analizar porcentaje de filas por año dentro de outputs/EDA
	analyze_yearly_percentage(df, output_dir=eda_dir)

	# Generar gráfico de distribución por duración y guardarlo también en outputs/EDA
	plot_duration_pie_buckets(df, output_dir=eda_dir)
	
 	# Generar gráfico de distribución por calidad/definición y guardarlo en outputs/EDA
	plot_definicion_pie(df=df, output_dir=eda_dir)

	#--------------------------------------------------
 	# TÍTULO Y DESCRIPCIÓN
  	#--------------------------------------------------
	title_dir = os.path.join(root, 'outputs', 'Title & Description')
	os.makedirs(title_dir, exist_ok=True)
	
	# Análisis general sin filtro de duración
	analyze_title_word_count(df, output_dir=title_dir)
	
 	# Análisis de longitud de descripción
	analyze_description_length(df, output_dir=title_dir)

	#--------------------------------------------------
	# TAGS
	#--------------------------------------------------
	tags_dir = os.path.join(root, 'outputs', 'Tags')
	os.makedirs(tags_dir, exist_ok=True)

	# Generar histogramas de tags para <=1min y (1min,16min]
	analyze_tags_intervals(df, output_dir=tags_dir)

	# Generar listado/plots de tags más frecuentes para <=1min y (1min,16min]
	analyze_most_frequent_tags(df, output_dir=tags_dir)

	#--------------------------------------------------
 	# WEEKDAYS
  	#--------------------------------------------------
	weekdays_dir = os.path.join(root, 'outputs', 'Weekdays')
	os.makedirs(weekdays_dir, exist_ok=True)

	# Videos de hasta 1 minuto (<= 60s)
	analyze_weekday_distribution(df, output_dir=weekdays_dir, file_suffix='le1min', duracion_filter=(None, 60))

	# Videos entre 1min (exclusive) y 16min (<= 960s)
	analyze_weekday_distribution(df, output_dir=weekdays_dir, file_suffix='1_16min', duracion_filter=(60, 16*60))

	#--------------------------------------------------
	# ANÁLISIS DEL DÍA 
	#--------------------------------------------------
	analisis_dia_dir = os.path.join(root, 'outputs', 'Análisis del día')
	os.makedirs(analisis_dia_dir, exist_ok=True)
 
	# Conteo de publicaciones por intervalos de 2 horas, con análisis separado por día de la semana
	analyze_publications_2h_intervals(df, output_dir=analisis_dia_dir, por_dia=True)




if __name__ == '__main__':
	main()

