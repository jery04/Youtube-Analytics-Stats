import os
from typing import Optional
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np

# Función para cargar el CSV de videos
def load_csv(path: str) -> pd.DataFrame:
	"""Carga el CSV de videos y devuelve un DataFrame.

	Espera que exista la columna numérica `durationSeconds`.
	"""
	return pd.read_csv(path, low_memory=False)

# Analizar el porcentaje de videos por canal en un dataset
def analyze_channel_percentage(df, root=None, output_dir=None, file_suffix=None, top_n=20):
	"""
	Calcula el porcentaje de videos por canal y guarda PNG y TXT.
	Guarda archivos en outputs/<script>/ y evita sobreescrituras usando `file_suffix`.
	
	Args:
		df: DataFrame con los datos
		root: Directorio raíz del proyecto
		output_dir: Directorio de salida personalizado (opcional)
		file_suffix: Sufijo para el nombre de archivo (opcional)
		top_n: Número de canales top a mostrar en el gráfico (default: 20)
	
	Returns:
		DataFrame con estadísticas por canal (canal, count, percent)
	"""
	if root is None:
		root = Path(__file__).resolve().parents[1]
	outputs_root = root / "outputs"
	outputs_root.mkdir(parents=True, exist_ok=True)
	if output_dir is not None:
		script_dir = Path(output_dir)
	else:
		script_dir = outputs_root / "EDA"
	script_dir.mkdir(parents=True, exist_ok=True)

	df = df.copy()

	# Buscar columna de canal
	channel_cols = [
		"canal",
		"canal_id",
		"channel",
		"channel_name",
		"channel_title",
		"channelTitle",
		"nombre_canal",
		"uploader",
		"creator",
	]
	found = None
	for c in channel_cols:
		if c in df.columns:
			found = c
			break
	if not found:
		# Buscar por patrones más generales
		for c in df.columns:
			if re.search(r'canal|channel|uploader|creator', c, re.I):
				found = c
				break
	if not found:
		print("No se encontró una columna de canal para el análisis.")
		return None

	# Limpiar datos
	df = df.dropna(subset=[found])
	df[found] = df[found].astype(str).str.strip()
	df = df[df[found] != '']
	
	if df.empty:
		print("No hay datos válidos con canales para el análisis.")
		return None

	# Contar videos por canal
	counts = df[found].value_counts()
	total = int(counts.sum())
	if total == 0:
		print("No hay filas con canal válido para el análisis.")
		return None

	perc = (counts / total * 100).round(4)

	# Preparar sufijo seguro
	suffix = f"_{re.sub(r'\\W+', '_', str(file_suffix)).strip('_')}" if file_suffix else ""

	# Crear DataFrame de resultados
	results_df = pd.DataFrame({
		'canal': counts.index,
		'count': counts.values,
		'percent': perc.values
	})

	# Gráfico de pastel con los top N canales
	top_counts = counts.head(top_n)
	top_perc = perc.head(top_n)
	
	# Agrupar el resto como "Otros"
	otros_count = counts.iloc[top_n:].sum() if len(counts) > top_n else 0
	
	plt.figure(figsize=(12, 10))
	
	labels = [str(c)[:25] for c in top_counts.index]
	sizes = top_counts.values.tolist()
	
	if otros_count > 0:
		labels.append('Otros')
		sizes.append(otros_count)
	
	colors = plt.cm.viridis([i / len(labels) for i in range(len(labels))])
	
	# Crear gráfico de pastel
	wedges, texts, autotexts = plt.pie(
		sizes, 
		labels=None,
		autopct=lambda p: f'{p:.1f}%' if p > 2 else '',
		colors=colors,
		startangle=90,
		pctdistance=0.8,
		wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'}
	)
	
	# Aumentar tamaño de fuentes para mejor legibilidad
	title_fs = 18
	legend_fs = 13
	autotext_fs = 12

	for t in autotexts:
		t.set_fontsize(autotext_fs)
		t.set_weight('bold')
		t.set_color('white')

	# Leyenda externa con fuente más grande
	plt.legend(wedges, labels, title="Canales", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=legend_fs, title_fontsize=legend_fs+1)
	plt.title(f'Distribución de Videos por Canal (Top {top_n})', fontsize=title_fs)
	
	plt.tight_layout()

	out_file = script_dir / f"channel_percentage{suffix}.png"
	plt.savefig(out_file, dpi=150)
	print(f"Gráfico de canales guardado en {out_file}")

	# Guardar txt con todos los canales
	txt_file = script_dir / f"channel_percentage{suffix}.txt"
	with open(txt_file, 'w', encoding='utf-8') as f:
		f.write('Canal,Count,Percent\n')
		for _, row in results_df.iterrows():
			f.write(f"{row['canal']},{int(row['count'])},{row['percent']:.4f}\n")
	print(f"Conteos por canal guardados en {txt_file}")
	
	# Mostrar resumen
	print(f"\n--- Resumen del análisis por canal ---")
	print(f"Total de videos: {total}")
	print(f"Total de canales únicos: {len(counts)}")
	print(f"\nTop 10 canales:")
	for i, (canal, cnt) in enumerate(counts.head(10).items(), 1):
		p = perc.loc[canal]
		print(f"  {i}. {canal}: {cnt} videos ({p:.2f}%)")
	
	plt.close()
	return results_df

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
			txt_file = out_file.with_suffix('.txt')

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
			txt_file = out_file.with_suffix('.txt')
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
			txt_file = out_file.with_suffix('.txt')
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
		txt_file = out_file.with_suffix('.txt')
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
	# Usar carpeta EDA por defecto
	script_dir = outputs_root / "EDA"
	script_dir.mkdir(parents=True, exist_ok=True)
	if output_dir is not None:
		script_dir = Path(output_dir)
		script_dir.mkdir(parents=True, exist_ok=True)

	df = df.copy()

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
		print("No se encontró una columna de fecha/hora para el análisis por año.")
		return None

	# Parsear fechas (format='mixed' soporta formatos mixtos como
	# '2026-02-28 06:05:00+00:00' y '2026-02-28T08:43:31Z' en la misma columna)
	try:
		df[found] = pd.to_datetime(df[found], format='mixed', errors='coerce')
	except Exception:
		df[found] = pd.to_datetime(df[found].astype(str), format='mixed', errors='coerce')

	df = df.dropna(subset=[found])
	if df.empty:
		print("No hay datos válidos con fechas para el análisis por año.")
		return None

	# Extraer año y contar
	df['__year'] = df[found].dt.year
	counts = df['__year'].value_counts().sort_index()
	total = int(counts.sum())
	if total == 0:
		print("No hay filas para contar por año.")
		return None

	# Gráfico de barras
	plt.figure(figsize=(10, 5))
	ax = sns.barplot(x=counts.index.astype(str), y=counts.values, palette='viridis')
	plt.xlabel('Año')
	plt.ylabel('Cantidad de videos')
	title = 'Cantidad de videos por año'
	plt.title(title)

	# Anotar porcentaje encima de cada barra
	for p in ax.patches:
		height = p.get_height()
		if total > 0:
			percent = height / total * 100
			ax.annotate(f"{percent:.1f}%", (p.get_x() + p.get_width() / 2, height),
					ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

	plt.tight_layout()

	suffix = f"_{re.sub(r'\\W+', '_', str(file_suffix)).strip('_')}" if file_suffix else ""
	out_file = script_dir / f"yearly_histogram{suffix}.png"
	plt.savefig(out_file, dpi=150)
	print(f"Histograma anual guardado en {out_file}")

	# Guardar TXT con Year, Count, Percent
	txt_file = script_dir / f"yearly_histogram{suffix}.txt"
	with open(txt_file, 'w', encoding='utf-8') as f:
		f.write('Year,Count,Percent\n')
		for year, cnt in counts.items():
			f.write(f"{int(year)},{int(cnt)},{cnt/total*100:.2f}\n")
	print(f"Conteos por año guardados en {txt_file}")

	plt.close()

	# Devolver Series con conteos (índice numérico de año)
	ser = pd.Series(data=counts.values, index=counts.index.astype(int))
	return ser

# Grafico poligonal: cantidad de videos publicados por mes
def plot_videos_por_mes(df=None, csv_path=None, start_month='2025-01', end_month=None, root=None, output_dir=None, file_suffix=None, save=True, show=False):
	"""
	Genera una línea poligonal con la cantidad de videos publicados por mes
	entre `start_month` y `end_month` (ambos inclusivos). Formato de meses: 'YYYY-MM'.
	Si `df` es None se carga `dataset/videos.csv` por defecto.
	Retorna un DataFrame con columnas `month` (YYYY-MM) y `count`.
	"""
	if root is None:
		root = Path(__file__).resolve().parents[1]
	outputs_root = root / "outputs"
	outputs_root.mkdir(parents=True, exist_ok=True)
	if output_dir is None:
		script_dir = outputs_root / "EDA"
	else:
		script_dir = Path(output_dir)
	script_dir.mkdir(parents=True, exist_ok=True)

	# Cargar dataframe si es necesario
	if df is None:
		if csv_path is None:
			csv_path = root / 'dataset' / 'videos.csv'
		try:
			df = pd.read_csv(csv_path, low_memory=False)
		except Exception as e:
			print(f"No se pudo cargar CSV desde {csv_path}: {e}")
			return None

	df = df.copy()

	# Detectar columna de fecha/hora (heurístico ya usado en el módulo)
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
		print("No se encontró una columna de fecha/hora para plot_videos_por_mes.")
		return None

	# Parsear fechas
	try:
		df[found] = pd.to_datetime(df[found], format='mixed', errors='coerce')
	except Exception:
		df[found] = pd.to_datetime(df[found].astype(str), format='mixed', errors='coerce')

	df = df.dropna(subset=[found])
	if df.empty:
		print("No hay datos válidos con fechas para plot_videos_por_mes.")
		return None

	# Normalizar diferencias de zona horaria: convertir a naive UTC si es necesario
	try:
		if pd.api.types.is_datetime64tz_dtype(df[found].dtype):
			df[found] = df[found].dt.tz_convert('UTC').dt.tz_localize(None)
	except Exception:
		# En caso de cualquier problema con tz conversion, forzamos a naive sin cambiar zonas
		try:
			df[found] = df[found].dt.tz_localize(None)
		except Exception:
			pass

	# Periodos de inicio y fin
	try:
		start = pd.Period(str(start_month), freq='M')
	except Exception:
		print(f"start_month inválido: {start_month}")
		return None

	if end_month is None:
		now = pd.Timestamp.now()
		end = pd.Period(now.strftime('%Y-%m'), freq='M')
	else:
		try:
			end = pd.Period(str(end_month), freq='M')
		except Exception:
			print(f"end_month inválido: {end_month}")
			return None

	if end < start:
		print("end_month debe ser igual o posterior a start_month.")
		return None

	# Filtrar por rango (incluir todo el mes final)
	mask = (df[found] >= start.to_timestamp()) & (df[found] < (end + 1).to_timestamp())
	df = df[mask]

	# Agrupar por periodo mensual
	df['__period'] = df[found].dt.to_period('M')
	counts = df['__period'].value_counts().sort_index()

	# Asegurar meses sin datos estén presentes
	full = pd.period_range(start=start, end=end, freq='M')
	counts = counts.reindex(full, fill_value=0)

	# Preparar Serie para graficar
	idx = counts.index.to_timestamp()
	vals = counts.values

	plt.figure(figsize=(12, 6))
	sns.lineplot(x=idx, y=vals, marker='o', color='tab:blue')
	plt.xlabel('Mes')
	plt.ylabel('Cantidad de videos publicados')
	plt.title(f'Videos publicados por mes ({str(start)} → {str(end)})')
	plt.xticks(rotation=45)
	pl_tight = plt.tight_layout()

	suffix = f"_{re.sub(r'\\W+', '_', str(file_suffix)).strip('_')}" if file_suffix else ""
	out_file = script_dir / f"monthly_videos{suffix}.png"
	if save:
		plt.savefig(out_file, dpi=150)
		print(f"Gráfico mensual guardado en {out_file}")

	# Guardar TXT con Month,Count
	txt_file = script_dir / f"monthly_videos{suffix}.txt"
	with open(txt_file, 'w', encoding='utf-8') as f:
		f.write('Month,Count\n')
		for per, cnt in counts.items():
			f.write(f"{str(per)},{int(cnt)}\n")
	print(f"Conteos mensuales guardados en {txt_file}")

	if show:
		plt.show()
	plt.close()

	res_df = pd.DataFrame({'month': [str(p) for p in counts.index], 'count': counts.values})
	return res_df

# Grafico poligonal: cantidad de videos publicados por semana
def plot_videos_por_semana(df=None, csv_path=None, start_week='2026-01-01', end_week=None, root=None, output_dir=None, file_suffix=None, save=True, show=False):
	"""
	Genera una línea poligonal con la cantidad de videos publicados por semana
	entre `start_week` y `end_week` (ambos inclusivos). `start_week` es la fecha
	exacta donde debe comenzar el conteo (puede ser cualquier día). La primera
	semana irá desde `start_week` hasta el primer domingo siguiente (inclusivo),
	y las semanas siguientes se consideran de lunes a domingo (ambos inclusive).
	Si `df` es None se carga `dataset/videos.csv` por defecto.
	Retorna un DataFrame con columnas `week_start` y `count`.
	"""
	if root is None:
		root = Path(__file__).resolve().parents[1]
	outputs_root = root / "outputs"
	outputs_root.mkdir(parents=True, exist_ok=True)
	if output_dir is None:
		script_dir = outputs_root / "EDA"
	else:
		script_dir = Path(output_dir)
	script_dir.mkdir(parents=True, exist_ok=True)

	# Cargar dataframe si es necesario
	if df is None:
		if csv_path is None:
			csv_path = root / 'dataset' / 'videos.csv'
		try:
			df = pd.read_csv(csv_path, low_memory=False)
		except Exception as e:
			print(f"No se pudo cargar CSV desde {csv_path}: {e}")
			return None

	df = df.copy()

	# Detectar columna de fecha/hora (reusar heurística)
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
		print("No se encontró una columna de fecha/hora para plot_videos_por_semana.")
		return None

	# Parsear fechas
	try:
		df[found] = pd.to_datetime(df[found], format='mixed', errors='coerce')
	except Exception:
		df[found] = pd.to_datetime(df[found].astype(str), format='mixed', errors='coerce')

	df = df.dropna(subset=[found])
	if df.empty:
		print("No hay datos válidos con fechas para plot_videos_por_semana.")
		return None

	# Normalizar tz
	try:
		if pd.api.types.is_datetime64tz_dtype(df[found].dtype):
			df[found] = df[found].dt.tz_convert('UTC').dt.tz_localize(None)
	except Exception:
		try:
			df[found] = df[found].dt.tz_localize(None)
		except Exception:
			pass

	# Parsear parámetros de fecha
	try:
		start_ts = pd.Timestamp(str(start_week))
	except Exception:
		print(f"start_week inválido: {start_week}")
		return None

	if end_week is None:
		end_ts = pd.Timestamp.now()
	else:
		try:
			end_ts = pd.Timestamp(str(end_week))
		except Exception:
			print(f"end_week inválido: {end_week}")
			return None

	if end_ts < start_ts:
		print("end_week debe ser igual o posterior a start_week.")
		return None

	# Filtrar por rango (inclusivo)
	mask = (df[found] >= start_ts) & (df[found] <= end_ts)
	df = df[mask]

	# Construir intervalos: primer intervalo desde start_ts hasta el primer domingo
	period_starts = []
	period_ends = []
	cur_start = start_ts.normalize()
	first_sunday = cur_start + pd.Timedelta(days=(6 - cur_start.weekday()))
	if first_sunday > end_ts:
		first_sunday = end_ts
	period_starts.append(cur_start)
	period_ends.append(first_sunday)
	cur_next = (first_sunday + pd.Timedelta(days=1)).normalize()

	while cur_next <= end_ts.normalize():
		cur_end = cur_next + pd.Timedelta(days=6)
		if cur_end > end_ts:
			cur_end = end_ts
		period_starts.append(cur_next)
		period_ends.append(cur_end)
		cur_next = (cur_end + pd.Timedelta(days=1)).normalize()

	# Contar por cada intervalo (inclusivo ambos extremos)
	counts_list = []
	for s, e in zip(period_starts, period_ends):
		mask_i = (df[found] >= s) & (df[found] <= e)
		counts_list.append(int(mask_i.sum()))

	# Preparar índice y serie
	idx = pd.to_datetime(period_starts)
	vals = np.array(counts_list, dtype=int)

	plt.figure(figsize=(12, 6))
	sns.lineplot(x=idx, y=vals, marker='o', color='tab:blue')
	plt.xlabel('Semana (inicio)')
	plt.ylabel('Cantidad de videos publicados')
	plt.title(f'Videos publicados por semana ({str(start_ts.date())} → {str(end_ts.date())})')

	# Formato del eje x y delimitadores mensuales
	ax = plt.gca()
	ax.xaxis.set_major_locator(mdates.AutoDateLocator())
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
	plt.xticks(rotation=45)

	# Dibujar líneas verticales en el inicio de cada mes dentro del rango
	if len(idx) > 0:
		month_starts = pd.date_range(start=idx.min().normalize(), end=idx.max().normalize(), freq='MS')
		for ms in month_starts:
			ax.axvline(ms, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

	plt.tight_layout()

	suffix = f"_{re.sub(r'\\W+', '_', str(file_suffix)).strip('_')}" if file_suffix else ""
	out_file = script_dir / f"weekly_videos{suffix}.png"
	if save:
		plt.savefig(out_file, dpi=150)
		print(f"Gráfico semanal guardado en {out_file}")

	# Guardar TXT con WeekStart,Count
	txt_file = script_dir / f"weekly_videos{suffix}.txt"
	with open(txt_file, 'w', encoding='utf-8') as f:
		f.write('WeekStart,Count\n')
		for s, cnt in zip(period_starts, counts_list):
			f.write(f"{str(s.date())},{int(cnt)}\n")
	print(f"Conteos semanales guardados en {txt_file}")

	if show:
		plt.show()
	plt.close()

	res_df = pd.DataFrame({'week_start': [str(p.date()) for p in period_starts], 'count': vals})
	return res_df

# Conteo de videos por día de la semana (Lun-Dom)
def analyze_weekday_distribution(df, root=None, output_dir=None, file_suffix=None, canal_filter=None, duracion_filter=None, menos_de_1min=True):
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
			Si es None se filtra por defecto por videos de menos de 1 minuto
			(<= 60s). Para cambiar a filtrar videos de más de 1 minuto pase
			`menos_de_1min=False`. Si se proporciona `duracion_filter`, este
			parámetro es ignorado.

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

	# Mapear el parámetro `menos_de_1min` a `duracion_filter` si no se
	# proporcionó un filtro explícito. Esto mantiene compatibilidad con
	# el uso previo de `duracion_filter`.
	if duracion_filter is None:
		if menos_de_1min:
			duracion_filter = (None, 60)
		else:
			duracion_filter = (61, None)

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

	# Parsear fechas (format='mixed' soporta formatos mixtos como
	# '2026-02-28 06:05:00+00:00' y '2026-02-28T08:43:31Z' en la misma columna)
	try:
		df[found] = pd.to_datetime(df[found], format='mixed', errors='coerce')
	except Exception:
		df[found] = pd.to_datetime(df[found].astype(str), format='mixed', errors='coerce')

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
	b3 = 3 * 60
	if duracion_filter is not None:
		# Si se proporciona un filtro explícito, usar solo ese
		buckets = [("custom", lambda s: (s >= (duracion_filter[0] or -float('inf'))) & (s <= (duracion_filter[1] or float('inf'))))]
	else:
		buckets = [
			("le1min", lambda s: s <= b1),
			("3_16min", lambda s: (s > b3) & (s <= b2)),
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
			Si es None se filtra por defecto por videos de menos de 1 minuto
			(<= 60s). Para cambiar a filtrar videos de más de 1 minuto pase
			`menos_de_1min=False`. Si se proporciona `duracion_filter`, este
			parámetro es ignorado.

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

	# Buckets por defecto si no hay duracion_filter: <=1min y (3min,16min]
	b1 = 60
	b2 = 16 * 60
	b3 = 3 * 60
	if duracion_filter is None:
		buckets = [
			("le1min", lambda s: s <= b1),
			("3_16min", lambda s: (s > b3) & (s <= b2)),
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
		# Buscar columna "tags" y contar elementos (separados por "|")
		for c in df.columns:
			if re.search(r'^tags$', c, re.I):
				df['__num_tags'] = df[c].apply(
					lambda x: 0 if pd.isna(x) or str(x).strip() in ('', '[]', 'nan')
					else len([t for t in str(x).split('|') if t.strip()])
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

	# Buckets por duración: <=1min y (3min,16min]
	b1 = 60
	b2 = 16 * 60
	b3 = 3 * 60
	buckets = [
		("le1min", lambda s: s <= b1),
		("3_16min", lambda s: (s > b3) & (s <= b2)),
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
	b3 = 3 * 60
	if duracion_filter is None and '__duration_s' in df.columns:
		buckets = [
			("le1min", lambda s: s <= b1),
			("3_16min", lambda s: (s > b3) & (s <= b2)),
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

	# Parsear fechas (format='mixed' soporta formatos mixtos como
	# '2026-02-28 06:05:00+00:00' y '2026-02-28T08:43:31Z' en la misma columna)
	try:
		df[found] = pd.to_datetime(df[found], format='mixed', errors='coerce')
	except Exception:
		df[found] = pd.to_datetime(df[found].astype(str), format='mixed', errors='coerce')

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

	# Etiquetas legibles (minutos si bin_size < 1, horas si >=1) – notación (a, b]
	def _label(a, b):
		if bin_size < 1.0:
			a_min = int(a * 60)
			b_min = int(b * 60)
			return f"({a_min}-{b_min}]min"
		else:
			a_h = int(a)
			b_h = int(b)
			return f"({a_h}-{b_h}]h"

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

		# Preparar subcarpetas para <=1min y (3min,16min] dentro del directorio por_dia
		long_dir = por_dia_dir / "3_16min"
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

			cats_day = pd.cut(df_day['__hour_frac'], bins=bins, labels=labels, include_lowest=True, right=True)
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

			# --- Guardar análisis para videos entre 3min y 16min en long_dir ---
			try:
				df_day_mid = df_day.copy()
				if '__dur_s' in df_day_mid.columns:
					df_day_mid = df_day_mid[(df_day_mid['__dur_s'] > 3 * 60) & (df_day_mid['__dur_s'] <= 16 * 60)]
				else:
					df_day_mid = df_day_mid.iloc[0:0]
			except Exception:
				df_day_mid = df_day.iloc[0:0]

			cats_day_mid = pd.cut(df_day_mid['__hour_frac'] if not df_day_mid.empty else pd.Series([], dtype=float), bins=bins, labels=labels, include_lowest=True, right=True)
			counts_day_mid = cats_day_mid.value_counts().reindex(labels, fill_value=0)

			plt.figure(figsize=(12,5))
			sns.barplot(x=list(counts_day_mid.index), y=counts_day_mid.values, palette='rocket')
			plt.xticks(rotation=45, ha='right')
			plt.xlabel('Intervalo horario')
			plt.ylabel('Cantidad de videos publicados')
			plt.title(f'Publicaciones por intervalos (3-16min] - {day_label}')
			plt.tight_layout()

			out_png_mid = long_dir / f"hour_2h_histogram_{day_name_file}.png"
			plt.savefig(out_png_mid, dpi=150)
			print(f"  Histograma 3-16min {day_label} guardado en {out_png_mid}")

			out_txt_mid = long_dir / f"hour_2h_histogram_{day_name_file}.txt"
			with open(out_txt_mid, 'w', encoding='utf-8') as f:
				if filter_label:
					f.write(f"# Filtros: {filter_label}\n")
				f.write(f"# Día: {day_label}\n")
				f.write('Interval,Count\n')
				for lbl, cnt in counts_day_mid.items():
					f.write(f"{lbl},{int(cnt)}\n")
			print(f"  Conteos 3-16min {day_label} guardados en {out_txt_mid}")
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

			cats_day_short = pd.cut(df_day_short['__hour_frac'] if not df_day_short.empty else pd.Series([], dtype=float), bins=bins, labels=labels, include_lowest=True, right=True)
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

	# pd.cut sobre la hora fraccionaria; usar right=True para (a,b]
	cats = pd.cut(df['__hour_frac'], bins=bins, labels=labels, include_lowest=True, right=True)

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

# Distribución de videos en intervalos de minutos (1-2,...,15-16) y segundos (0-10,...,50-60)
def analyze_duration_interval_distribution(df, root=None, output_dir=None, file_suffix=None, save=True, show=False, minute_bins=None, second_bins=None):
	"""
	Cuenta la cantidad de videos en intervalos de duración configurables.

	Args:
		df: DataFrame con los datos.
		root: Directorio raíz del proyecto.
		output_dir: Directorio de salida personalizado (opcional).
		file_suffix: Sufijo para los nombres de archivo (opcional).
		save: Guardar imágenes en disco.
		show: Mostrar figuras en pantalla.
		minute_bins: Opcional. Define los intervalos en minutos para la gráfica
			de minutos. Puede ser:
				- None (valor por defecto): crea [(1,2),(2,3),...,(15,16)].
				- lista de tuplas (low, high) en minutos.
				- lista de enteros: cada entero `n` genera (n, n+1).
		second_bins: Opcional. Define los intervalos en segundos para la gráfica
			de segundos. Puede ser:
				- None (valor por defecto): crea [(0,10),(10,20),...,(50,60)].
				- lista de tuplas (low, high) en segundos.
				- lista de enteros: cada entero `n` genera (n, n+10).

	Genera histogramas con el porcentaje encima de cada barra y guarda PNG/TXT en
	`outputs/Distribucion min` o `output_dir` si se provee.
	"""
	if root is None:
		root = Path(__file__).resolve().parents[1]
	outputs_root = root / "outputs"
	outputs_root.mkdir(parents=True, exist_ok=True)
	# Directorio destino
	if output_dir is None:
		script_dir = outputs_root / "Distribucion min"
	else:
		script_dir = Path(output_dir)
	script_dir.mkdir(parents=True, exist_ok=True)

	df = df.copy()

	# Heurística para detectar columna de duración
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

	# Función para parsear a segundos (compatible con otras funciones)
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

	if '__duration_s' in df.columns:
		# usar columna ya existente
		df['__duration_s'] = pd.to_numeric(df['__duration_s'], errors='coerce')
	else:
		if dur_col is None:
			print('No se encontró columna de duración para analyze_duration_interval_distribution.')
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
		print('No hay duraciones válidas para analyze_duration_interval_distribution.')
		return None

	# --- MINUTOS: intervalos configurables ---
	def _normalize_bins(bins, default_tuples, default_step):
		if bins is None:
			return default_tuples
		# lista de tuplas (low, high)
		if all(isinstance(b, (list, tuple)) and len(b) == 2 for b in bins):
			try:
				return [(int(a), int(b)) for a, b in bins]
			except Exception:
				return default_tuples
		# lista de números -> crear (n, n+step)
		try:
			nums = [int(x) for x in bins]
			return [(n, n + default_step) for n in nums]
		except Exception:
			return default_tuples

	default_minute = [(i, i+1) for i in range(1, 16)]
	minute_bins_norm = _normalize_bins(minute_bins, default_minute, 1)
	minute_labels = [f"{a}-{b}min" for a, b in minute_bins_norm]
	minute_counts = []
	for low, high in minute_bins_norm:
		cnt = int(((df['__duration_s'] > low*60) & (df['__duration_s'] <= high*60)).sum())
		minute_counts.append(cnt)
	min_total = sum(minute_counts)

	# Plot minutos
	plt.figure(figsize=(12,6))
	ax = sns.barplot(x=minute_labels, y=minute_counts, palette='viridis')
	plt.xlabel('Intervalo (min)')
	plt.ylabel('Cantidad de videos')
	plt.title('Distribución por duración (minutos)')

	# Anotar porcentaje encima de cada barra
	for i, v in enumerate(minute_counts):
		pct = (v / min_total * 100) if min_total > 0 else 0
		ax.annotate(f"{pct:.1f}%", (i, v), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9)

	plt.xticks(rotation=45, ha='right')
	plt.tight_layout()
	# Lógica para sufijos según reglas del usuario
	suffix_min = ""
	suffix_sec = ""
	if file_suffix:
		# Ambos bins especificados
		if minute_bins is not None and second_bins is not None:
			suffix_min = suffix_sec = f"_{re.sub(r'\\W+', '_', str(file_suffix)).strip('_')}"
		# Solo uno de los bins especificado
		elif minute_bins is not None and second_bins is None:
			suffix_min = f"_{re.sub(r'\\W+', '_', str(file_suffix)).strip('_')}"
		elif second_bins is not None and minute_bins is None:
			suffix_sec = f"_{re.sub(r'\\W+', '_', str(file_suffix)).strip('_')}"
		# Ningún bin especificado
		elif minute_bins is None and second_bins is None:
			suffix_min = suffix_sec = f"_{re.sub(r'\\W+', '_', str(file_suffix)).strip('_')}"

	out_png = script_dir / f"duration_distribution_minutes{suffix_min}.png"
	if save:
		plt.savefig(out_png, dpi=150)
		print(f"Histograma minutos guardado en {out_png}")
	out_txt = script_dir / f"duration_distribution_minutes{suffix_min}.txt"
	with open(out_txt, 'w', encoding='utf-8') as f:
		f.write('Interval,Count,Percent\n')
		for lbl, cnt in zip(minute_labels, minute_counts):
			pct = (cnt / min_total * 100) if min_total > 0 else 0
			f.write(f"{lbl},{int(cnt)},{pct:.2f}\n")
	plt.close()

	# --- SEGUNDOS: intervalos configurables ---
	default_seconds = [(i, i+10) for i in range(0, 60, 10)]
	second_bins_norm = _normalize_bins(second_bins, default_seconds, 10)
	second_labels = [f"{a}-{b}s" for a, b in second_bins_norm]
	second_counts = []
	for low, high in second_bins_norm:
		# primer intervalo: >0 hasta <=high, otros: >low <=high
		if low == 0:
			cnt = int(((df['__duration_s'] > 0) & (df['__duration_s'] <= high)).sum())
		else:
			cnt = int(((df['__duration_s'] > low) & (df['__duration_s'] <= high)).sum())
		second_counts.append(cnt)
	sec_total = sum(second_counts)

	plt.figure(figsize=(10,5))
	ax2 = sns.barplot(x=second_labels, y=second_counts, palette='rocket')
	plt.xlabel('Intervalo (s)')
	plt.ylabel('Cantidad de videos')
	plt.title('Distribución por duración (segundos, hasta 60s)')

	for i, v in enumerate(second_counts):
		pct = (v / sec_total * 100) if sec_total > 0 else 0
		ax2.annotate(f"{pct:.1f}%", (i, v), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9)

	plt.tight_layout()
	out_png2 = script_dir / f"duration_distribution_seconds{suffix_sec}.png"
	if save:
		plt.savefig(out_png2, dpi=150)
		print(f"Histograma segundos guardado en {out_png2}")
	out_txt2 = script_dir / f"duration_distribution_seconds{suffix_sec}.txt"
	with open(out_txt2, 'w', encoding='utf-8') as f:
		f.write('Interval,Count,Percent\n')
		for lbl, cnt in zip(second_labels, second_counts):
			pct = (cnt / sec_total * 100) if sec_total > 0 else 0
			f.write(f"{lbl},{int(cnt)},{pct:.2f}\n")

	plt.close()

	# Devolver diccionario con resultados
	return {
		'minutes': pd.DataFrame({'interval': minute_labels, 'count': minute_counts, 'percent': [round((c/min_total*100) if min_total>0 else 0,2) for c in minute_counts]}),
		'seconds': pd.DataFrame({'interval': second_labels, 'count': second_counts, 'percent': [round((c/sec_total*100) if sec_total>0 else 0,2) for c in second_counts]}),
	}

# Genera un gráfico de pastel con la distribución de categorías en el dataset
def plot_category_pie(df=None, csv_path=None, root=None, output_dir=None, file_suffix=None, save=True, show=False):
	"""
	Detecta la columna de categoría del DataFrame (heurística) y genera
	un gráfico de pastel con la distribución de las distintas categorías.
	Guarda PNG y TXT en `outputs/EDA` por defecto o en `output_dir` si se
	proporciona.

	Retorna DataFrame con columnas ['category','count','percent'] o None.
	"""
	if root is None:
		root = Path(__file__).resolve().parents[1]
	outputs_root = root / "outputs"
	outputs_root.mkdir(parents=True, exist_ok=True)
	# Por defecto usar outputs/EDA
	if output_dir is None:
		script_dir = outputs_root / "EDA"
	else:
		script_dir = Path(output_dir)
	script_dir.mkdir(parents=True, exist_ok=True)

	# Cargar dataframe si no se recibió
	if df is None:
		if csv_path is None:
			csv_path = root / 'dataset' / 'videos.csv'
		try:
			df = pd.read_csv(csv_path, low_memory=False)
		except Exception as e:
			print(f"No se pudo cargar CSV desde {csv_path}: {e}")
			return None

	df = df.copy()

	# Heurística para detectar columna de categoría
	candidate_cols = [
		'category', 'categoria', 'category_id', 'categoryId', 'categoria_id',
		'category_name', 'categoryName', 'categoria_name'
	]
	found = None
	for c in candidate_cols:
		if c in df.columns:
			found = c
			break
	if not found:
		for c in df.columns:
			if re.search(r'cat(egor|egoria|egory)|category', c, re.I):
				found = c
				break
	if not found:
		print("No se encontró una columna de categoría en el dataset.")
		return None

	# Normalizar y contar
	df = df.dropna(subset=[found])
	if df.empty:
		print("No hay valores de categoría para generar el gráfico.")
		# Generar imagen indicativa
		labels = ['No data']
		sizes = [1]
		plt.figure(figsize=(8,8))
		wedges, texts, autotexts = plt.pie(sizes, labels=None, autopct=lambda p: '', colors=['#cccccc'], startangle=90, wedgeprops={'linewidth':0.5,'edgecolor':'white'})
		plt.legend(wedges, labels, title="Categoría", loc="center left", bbox_to_anchor=(1, 0.5))
		plt.title('Distribución por categoría')
		plt.tight_layout()
		suffix = f"_{re.sub(r'\\W+', '_', str(file_suffix)).strip('_')}" if file_suffix else ""
		out_file = script_dir / f"category_pie{suffix}.png"
		if save:
			plt.savefig(out_file, dpi=150)
			print(f"Gráfico indicativo de categorías guardado en {out_file}")
		txt_file = out_file.with_suffix('.txt')
		with open(txt_file, 'w', encoding='utf-8') as f:
			f.write('Category,Count,Percent\n')
		plt.close()
		return None

	df[found] = df[found].astype(str).str.strip()
	df = df[df[found] != '']
	if df.empty:
		print("Después de normalizar no hay categorías válidas.")
		return None

	counts = df[found].value_counts()
	total = int(counts.sum())
	if total == 0:
		print("No hay filas con categoría para contar.")
		return None

	perc = (counts / total * 100).round(2)

	# Preparar gráfico
	labels = [str(l) for l in counts.index]
	sizes = counts.values.tolist()
	colors = plt.cm.tab20([i / max(1, len(labels)-1) for i in range(len(labels))])

	plt.figure(figsize=(8,8))
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

	plt.legend(wedges, labels, title="Categoría", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10, title_fontsize=12)
	plt.title('Distribución por categoría')
	plt.tight_layout()

	suffix = f"_{re.sub(r'\\W+', '_', str(file_suffix)).strip('_')}" if file_suffix else ""
	out_file = script_dir / f"category_pie{suffix}.png"
	if save:
		plt.savefig(out_file, dpi=150)
		print(f"Gráfico de categorías guardado en {out_file}")

	txt_file = out_file.with_suffix('.txt')
	with open(txt_file, 'w', encoding='utf-8') as f:
		f.write('Category,Count,Percent\n')
		for cat, cnt, p in zip(labels, counts.values, perc.values):
			f.write(f"{cat},{int(cnt)},{p:.2f}\n")
	print(f"Conteos de categorías guardados en {txt_file}")

	if show:
		plt.show()
	plt.close()

	return pd.DataFrame({'category': counts.index, 'count': counts.values, 'percent': perc.values})

# Simulación Monte Carlo para comparar intervalos de duración según viewCount
def monte_carlo_duration_intervals(
	df: pd.DataFrame,
	output_dir: str | None = None,
	n_rounds: int = 200_000,
	seed: int = 42,
	save: bool = True,
	show: bool = False,
) -> dict:
	"""
	Simulación Monte Carlo por intervalos de duración.

	Para cada ronda se elige un video aleatorio de cada intervalo de duración;
	el intervalo cuyo video seleccionado tiene el mayor viewCount gana esa ronda.
	Se repite `n_rounds` veces y se genera un histograma con el porcentaje de
	victorias de cada intervalo.

	Se generan dos simulaciones:
	  1) Intervalos en segundos: [0-5], [5-10], …, [50-60]
	  2) Intervalos en minutos: [1-2], [2-3], …, [15-16]

	Los resultados se guardan en `output_dir` (por defecto outputs/Random).
	"""

	if output_dir is None:
		root = Path(__file__).resolve().parents[1]
		output_dir = root / "outputs" / "Random"
	else:
		output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# ── Resolver columna de duración en segundos ────────────────────────
	duration_candidates = [
		"durationSeconds", "duracion_segundos", "duration_seconds",
		"duration_sec", "length_seconds",
	]
	dur_col = None
	for c in duration_candidates:
		if c in df.columns:
			dur_col = c
			break
	if dur_col is None:
		for c in df.columns:
			if re.search(r'duraci|duration|length', c, re.I):
				dur_col = c
				break
	if dur_col is None:
		print("monte_carlo: No se encontró columna de duración.")
		return {}

	# ── Resolver columna de vistas ──────────────────────────────────────
	view_candidates = [
		"viewCount", "views", "vistas", "visualizaciones",
		"view_count", "reproducciones",
	]
	view_col = None
	for c in view_candidates:
		if c in df.columns:
			view_col = c
			break
	if view_col is None:
		for c in df.columns:
			if re.search(r'view|vista|visual|reproduc', c, re.I):
				view_col = c
				break
	if view_col is None:
		print("monte_carlo: No se encontró columna de vistas/viewCount.")
		return {}

	# ── Preparar datos ──────────────────────────────────────────────────
	tmp = df[[dur_col, view_col]].copy()
	tmp["dur_s"] = pd.to_numeric(tmp[dur_col], errors="coerce")
	tmp["views"] = pd.to_numeric(tmp[view_col], errors="coerce")
	tmp = tmp.dropna(subset=["dur_s", "views"])
	tmp["dur_s"] = tmp["dur_s"].astype(float)
	tmp["views"] = tmp["views"].astype(int)

	# ── Definir intervalos ──────────────────────────────────────────────
	second_bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40), (40, 50), (50, 60)]
	minute_bins = [(m, m + 1) for m in range(3, 16)]  # [3-4], [4-5], …, [15-16]

	# ── Función interna de simulación ───────────────────────────────────
	def _run_simulation(bins_def, unit="s"):
		"""
		bins_def: lista de tuplas (lo, hi) en la unidad indicada.
		unit: 's' para segundos, 'm' para minutos.
		Retorna dict {label: win_pct}.
		"""
		# Agrupar videos por intervalo
		interval_videos: dict[str, np.ndarray] = {}  # label -> array de views
		for lo, hi in bins_def:
			if unit == "s":
				mask = (tmp["dur_s"] > lo) & (tmp["dur_s"] <= hi)
				label = f"({lo}-{hi}]s"
			else:  # minutos
				mask = (tmp["dur_s"] > lo * 60) & (tmp["dur_s"] <= hi * 60)
				label = f"({lo}-{hi}]min"
			views_arr = tmp.loc[mask, "views"].values
			if len(views_arr) > 0:
				interval_videos[label] = views_arr

		if len(interval_videos) < 2:
			print(f"monte_carlo ({unit}): Menos de 2 intervalos con videos; se omite.")
			return {}, {}

		labels = list(interval_videos.keys())
		arrays = [interval_videos[l] for l in labels]
		n_intervals = len(labels)

		# Contar videos por intervalo
		counts = {labels[i]: len(arrays[i]) for i in range(n_intervals)}

		# Simulación vectorizada con NumPy para velocidad
		rng = np.random.default_rng(seed)
		wins = np.zeros(n_intervals, dtype=np.int64)

		# Generar índices aleatorios para todas las rondas de una vez
		idx_matrix = [rng.integers(0, len(arr), size=n_rounds) for arr in arrays]
		# Obtener vistas correspondientes
		views_matrix = np.column_stack(
			[arr[idx] for arr, idx in zip(arrays, idx_matrix)]
		)  # shape: (n_rounds, n_intervals)

		# Determinar ganador de cada ronda (mayor viewCount)
		winners = np.argmax(views_matrix, axis=1)  # shape: (n_rounds,)
		for i in range(n_intervals):
			wins[i] = np.sum(winners == i)

		win_pct = wins / n_rounds * 100
		return {labels[i]: float(win_pct[i]) for i in range(n_intervals)}, counts

	# ── Función interna para generar histograma ─────────────────────────
	def _plot_histogram(results: dict, title: str, filename: str):
		if not results:
			return
		labels_sorted = list(results.keys())
		pcts = [results[l] for l in labels_sorted]

		fig, ax = plt.subplots(figsize=(12, 6))
		bars = ax.bar(
			range(len(labels_sorted)), pcts,
			color=sns.color_palette("viridis", len(labels_sorted)),
			edgecolor="black",
		)
		ax.set_xticks(range(len(labels_sorted)))
		ax.set_xticklabels(labels_sorted, rotation=30, ha="right", fontsize=10)
		ax.set_ylabel("% de victorias", fontsize=12)
		ax.set_xlabel("Intervalo de duración", fontsize=12)
		ax.set_title(title, fontsize=14, fontweight="bold")

		# Etiquetas sobre las barras
		for bar, pct in zip(bars, pcts):
			ax.text(
				bar.get_x() + bar.get_width() / 2,
				bar.get_height() + 0.3,
				f"{pct:.2f}%",
				ha="center", va="bottom", fontsize=9, fontweight="bold",
			)

		plt.tight_layout()
		if save:
			img_path = output_dir / f"{filename}.png"
			fig.savefig(img_path, dpi=150)
			print(f"Imagen guardada: {img_path}")
		if show:
			plt.show()
		plt.close(fig)

	# ── Función interna para guardar TXT ────────────────────────────────
	def _save_txt(results: dict, filename: str, title: str, counts: dict | None = None):
		if not results:
			return
		txt_path = output_dir / f"{filename}.txt"
		with open(txt_path, "w", encoding="utf-8") as f:
			f.write(f"{title}\n")
			f.write(f"Rondas: {n_rounds:,}\n")
			f.write(f"{'Intervalo':<18}{'Count':>8}{'Victorias':>12}{'%':>10}\n")
			f.write("-" * 48 + "\n")
			for label, pct in results.items():
				wins_count = int(round(pct / 100 * n_rounds))
				count_val = counts.get(label, 0) if counts else 0
				f.write(f"{label:<18}{count_val:>8,}{wins_count:>12,}{pct:>9.2f}%\n")
		print(f"Datos guardados: {txt_path}")

	# ── Bins adicionales para videos ≤ 1 min ───────────────────────────
	le1min_bins_v1 = [(0, 10), (11, 20), (20, 30), (30, 45), (45, 60)]
	le1min_bins_v2 = [(0, 15), (15, 30), (30, 45), (45, 60)]

	# ── Ejecutar simulaciones ───────────────────────────────────────────
	print(f"\n{'='*60}")
	print(f"Monte Carlo – Intervalos de duración ({n_rounds:,} rondas)")
	print(f"{'='*60}")

	results_all = {}

	# 1) Segundos
	print("\n→ Simulación con intervalos en SEGUNDOS...")
	res_sec, counts_sec = _run_simulation(second_bins, unit="s")
	if res_sec:
		_plot_histogram(
			res_sec,
			f"Monte Carlo – Victorias por intervalo de duración (segundos)\n{n_rounds:,} rondas",
			"monte_carlo_seconds",
		)
		_save_txt(res_sec, "monte_carlo_seconds", "Monte Carlo – Intervalos en Segundos", counts=counts_sec)
		results_all["seconds"] = res_sec

	# 2) Minutos
	print("→ Simulación con intervalos en MINUTOS...")
	res_min, counts_min = _run_simulation(minute_bins, unit="m")
	if res_min:
		_plot_histogram(
			res_min,
			f"Monte Carlo – Victorias por intervalo de duración (minutos)\n{n_rounds:,} rondas",
			"monte_carlo_minutes",
		)
		_save_txt(res_min, "monte_carlo_minutes", "Monte Carlo – Intervalos en Minutos", counts=counts_min)
		results_all["minutes"] = res_min

	# 3) ≤1 min – bins v1: (0,10), (11,20), (20,30), (30,45), (45,60)
	print("→ Simulación ≤1 min con bins v1...")
	res_le1_v1, counts_le1_v1 = _run_simulation(le1min_bins_v1, unit="s")
	if res_le1_v1:
		_plot_histogram(
			res_le1_v1,
			f"Monte Carlo – ≤1 min bins v1\n{n_rounds:,} rondas",
			"monte_carlo_seconds_le1min_v1",
		)
		_save_txt(res_le1_v1, "monte_carlo_seconds_le1min_v1",
				  "Monte Carlo – ≤1 min bins v1", counts=counts_le1_v1)
		results_all["le1min_v1"] = res_le1_v1

	# 4) ≤1 min – bins v2: (0,15), (15,30), (30,45), (45,60)
	print("→ Simulación ≤1 min con bins v2...")
	res_le1_v2, counts_le1_v2 = _run_simulation(le1min_bins_v2, unit="s")
	if res_le1_v2:
		_plot_histogram(
			res_le1_v2,
			f"Monte Carlo – ≤1 min bins v2\n{n_rounds:,} rondas",
			"monte_carlo_seconds_le1min_v2",
		)
		_save_txt(res_le1_v2, "monte_carlo_seconds_le1min_v2",
				  "Monte Carlo – ≤1 min bins v2", counts=counts_le1_v2)
		results_all["le1min_v2"] = res_le1_v2

	print(f"{'='*60}\n")
	return results_all

# Simulación Monte Carlo para comparar intervalos de duración según viewCount
def monte_carlo_weekdays(
	df: pd.DataFrame,
	output_dir: str | None = None,
	n_rounds: int = 200_000,
	seed: int = 42,
	save: bool = True,
	show: bool = False,
) -> dict:
	"""
	Simulación Monte Carlo por día de la semana.

	Para cada ronda se elige un video aleatorio de cada día de la semana
	(lunes a domingo); el día cuyo video seleccionado tiene el mayor
	viewCount gana esa ronda.  Se repite `n_rounds` veces y se genera un
	histograma con el porcentaje de victorias de cada día.

	Los resultados se guardan en `output_dir` (por defecto outputs/Random).
	"""

	if output_dir is None:
		root = Path(__file__).resolve().parents[1]
		output_dir = root / "outputs" / "Random"
	else:
		output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# ── Resolver columna de fecha/hora ──────────────────────────────────
	date_candidates = [
		"publishedAt", "published_at", "fecha_publicacion",
		"fecha", "date", "upload_date",
	]
	date_col = None
	for c in date_candidates:
		if c in df.columns:
			date_col = c
			break
	if date_col is None:
		for c in df.columns:
			if re.search(r'publish|upload|date|time|fecha|publica|publicación|publicacion', c, re.I):
				date_col = c
				break
	if date_col is None:
		print("monte_carlo_weekdays: No se encontró columna de fecha/hora.")
		return {}

	# ── Resolver columna de vistas ──────────────────────────────────────
	view_candidates = [
		"viewCount", "views", "vistas", "visualizaciones",
		"view_count", "reproducciones",
	]
	view_col = None
	for c in view_candidates:
		if c in df.columns:
			view_col = c
			break
	if view_col is None:
		for c in df.columns:
			if re.search(r'view|vista|visual|reproduc', c, re.I):
				view_col = c
				break
	if view_col is None:
		print("monte_carlo_weekdays: No se encontró columna de vistas/viewCount.")
		return {}

	# ── Resolver columna de duración en segundos ───────────────────────
	duration_candidates = [
		"durationSeconds", "duracion_segundos", "duration_seconds",
		"duration_sec", "length_seconds",
	]
	dur_col = None
	for c in duration_candidates:
		if c in df.columns:
			dur_col = c
			break
	if dur_col is None:
		for c in df.columns:
			if re.search(r'duraci|duration|length', c, re.I):
				dur_col = c
				break
	if dur_col is None:
		print("monte_carlo_weekdays: No se encontró columna de duración.")
		return {}

	# ── Preparar datos ──────────────────────────────────────────────────
	tmp = df[[date_col, view_col, dur_col]].copy()
	try:
		tmp[date_col] = pd.to_datetime(tmp[date_col], format='mixed', errors='coerce')
	except Exception:
		tmp[date_col] = pd.to_datetime(tmp[date_col].astype(str), format='mixed', errors='coerce')
	tmp["views"] = pd.to_numeric(tmp[view_col], errors="coerce")
	tmp["dur_s"] = pd.to_numeric(tmp[dur_col], errors="coerce")
	tmp = tmp.dropna(subset=[date_col, "views", "dur_s"])
	if tmp.empty:
		print("monte_carlo_weekdays: No hay datos válidos con fechas, vistas y duración.")
		return {}
	tmp["views"] = tmp["views"].astype(int)
	tmp["dur_s"] = tmp["dur_s"].astype(float)
	tmp["weekday"] = tmp[date_col].dt.dayofweek  # Monday=0 .. Sunday=6

	weekday_labels = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']

	# ── Definir buckets de duración ─────────────────────────────────────
	duration_buckets = [
		("le1min",  lambda s: s <= 60,               "≤1min"),
		("3_16min", lambda s: (s > 180) & (s <= 960), "3-16min"),
	]

	# ── Histograma ──────────────────────────────────────────────────────
	def _plot_histogram(res: dict, title: str, filename: str):
		if not res:
			return
		lbls = list(res.keys())
		pcts = [res[l] for l in lbls]

		fig, ax = plt.subplots(figsize=(12, 6))
		bars = ax.bar(
			range(len(lbls)), pcts,
			color=sns.color_palette("viridis", len(lbls)),
			edgecolor="black",
		)
		ax.set_xticks(range(len(lbls)))
		ax.set_xticklabels(lbls, rotation=30, ha="right", fontsize=10)
		ax.set_ylabel("% de victorias", fontsize=12)
		ax.set_xlabel("Día de la semana", fontsize=12)
		ax.set_title(title, fontsize=14, fontweight="bold")

		for bar, pct in zip(bars, pcts):
			ax.text(
				bar.get_x() + bar.get_width() / 2,
				bar.get_height() + 0.3,
				f"{pct:.2f}%",
				ha="center", va="bottom", fontsize=9, fontweight="bold",
			)

		plt.tight_layout()
		if save:
			img_path = output_dir / f"{filename}.png"
			fig.savefig(img_path, dpi=150)
			print(f"Imagen guardada: {img_path}")
		if show:
			plt.show()
		plt.close(fig)

	# ── Guardar TXT ─────────────────────────────────────────────────────
	def _save_txt(res: dict, filename: str, title: str, counts: dict | None = None):
		if not res:
			return
		txt_path = output_dir / f"{filename}.txt"
		with open(txt_path, "w", encoding="utf-8") as f:
			f.write(f"{title}\n")
			f.write(f"Rondas: {n_rounds:,}\n")
			f.write(f"{'Día':<18}{'Count':>8}{'Victorias':>12}{'%':>10}\n")
			f.write("-" * 48 + "\n")
			for label, pct in res.items():
				wins_count = int(round(pct / 100 * n_rounds))
				count_val = counts.get(label, 0) if counts else 0
				f.write(f"{label:<18}{count_val:>8,}{wins_count:>12,}{pct:>9.2f}%\n")
		print(f"Datos guardados: {txt_path}")

	# ── Ejecutar simulación por cada bucket de duración ─────────────────
	print(f"\n{'='*60}")
	print(f"Monte Carlo – Días de la semana ({n_rounds:,} rondas)")
	print(f"{'='*60}")

	all_results = {}

	for bucket_name, bucket_cond, bucket_label in duration_buckets:
		print(f"\n→ Simulación para videos {bucket_label}...")
		tmp_filtered = tmp[bucket_cond(tmp["dur_s"])].copy()

		if tmp_filtered.empty:
			print(f"  Sin datos para bucket {bucket_label}; omitido.")
			continue

		# Agrupar videos por día de la semana
		weekday_videos: dict[str, np.ndarray] = {}
		for idx, label in enumerate(weekday_labels):
			views_arr = tmp_filtered.loc[tmp_filtered["weekday"] == idx, "views"].values
			if len(views_arr) > 0:
				weekday_videos[label] = views_arr

		if len(weekday_videos) < 2:
			print(f"  monte_carlo_weekdays ({bucket_label}): Menos de 2 días con videos; omitido.")
			continue

		labels = list(weekday_videos.keys())
		arrays = [weekday_videos[l] for l in labels]
		n_days = len(labels)

		# Simulación vectorizada
		rng = np.random.default_rng(seed)
		wins = np.zeros(n_days, dtype=np.int64)

		idx_matrix = [rng.integers(0, len(arr), size=n_rounds) for arr in arrays]
		views_matrix = np.column_stack(
			[arr[idx] for arr, idx in zip(arrays, idx_matrix)]
		)  # shape: (n_rounds, n_days)

		winners = np.argmax(views_matrix, axis=1)
		for i in range(n_days):
			wins[i] = np.sum(winners == i)

		win_pct = wins / n_rounds * 100
		results = {labels[i]: float(win_pct[i]) for i in range(n_days)}

		weekday_counts = {label: len(arr) for label, arr in weekday_videos.items()}

		_plot_histogram(
			results,
			f"Monte Carlo – Victorias por día de la semana ({bucket_label})\n{n_rounds:,} rondas",
			f"monte_carlo_weekdays_{bucket_name}",
		)
		_save_txt(
			results,
			f"monte_carlo_weekdays_{bucket_name}",
			f"Monte Carlo – Días de la semana ({bucket_label})",
			counts=weekday_counts,
		)

		all_results[bucket_name] = results

	print(f"{'='*60}\n")
	return all_results

# Simulación Monte Carlo por número de palabras en el título
def monte_carlo_title_words(
	df: pd.DataFrame,
	output_dir: str | None = None,
	n_rounds: int = 200_000,
	seed: int = 42,
	save: bool = True,
	show: bool = False,
) -> dict:
	"""
	Simulación Monte Carlo por número de palabras en el título.

	Para cada ronda se elige un video aleatorio de cada intervalo de
	palabras en el título; el intervalo cuyo video seleccionado tiene el
	mayor viewCount gana esa ronda. Se repite `n_rounds` veces y se genera
	un histograma con el porcentaje de victorias de cada intervalo.

	Intervalos: 1-3, 4-6, 7-9, 10-12, 13-15, 16-18, 19-21, 22-24 palabras.

	Se generan dos simulaciones separadas por duración:
	  1) Videos ≤1 min
	  2) Videos 1-16 min
	"""

	if output_dir is None:
		root = Path(__file__).resolve().parents[1]
		output_dir = root / "outputs" / "Random"
	else:
		output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# ── Resolver columna de título ──────────────────────────────────────
	title_col = None
	for c in ["title", "titulo", "título"]:
		if c in df.columns:
			title_col = c
			break
	if title_col is None:
		for c in df.columns:
			if re.search(r'title|titulo|título', c, re.I):
				title_col = c
				break
	if title_col is None:
		print("monte_carlo_title_words: No se encontró columna de título.")
		return {}

	# ── Resolver columna de vistas ──────────────────────────────────────
	view_candidates = [
		"viewCount", "views", "vistas", "visualizaciones",
		"view_count", "reproducciones",
	]
	view_col = None
	for c in view_candidates:
		if c in df.columns:
			view_col = c
			break
	if view_col is None:
		for c in df.columns:
			if re.search(r'view|vista|visual|reproduc', c, re.I):
				view_col = c
				break
	if view_col is None:
		print("monte_carlo_title_words: No se encontró columna de vistas/viewCount.")
		return {}

	# ── Resolver columna de duración en segundos ───────────────────────
	duration_candidates = [
		"durationSeconds", "duracion_segundos", "duration_seconds",
		"duration_sec", "length_seconds",
	]
	dur_col = None
	for c in duration_candidates:
		if c in df.columns:
			dur_col = c
			break
	if dur_col is None:
		for c in df.columns:
			if re.search(r'duraci|duration|length', c, re.I):
				dur_col = c
				break
	if dur_col is None:
		print("monte_carlo_title_words: No se encontró columna de duración.")
		return {}

	# ── Preparar datos ──────────────────────────────────────────────────
	tmp = df[[title_col, view_col, dur_col]].copy()
	tmp["views"] = pd.to_numeric(tmp[view_col], errors="coerce")
	tmp["dur_s"] = pd.to_numeric(tmp[dur_col], errors="coerce")
	tmp = tmp.dropna(subset=["views", "dur_s"])
	tmp["views"] = tmp["views"].astype(int)
	tmp["dur_s"] = tmp["dur_s"].astype(float)
	tmp["word_count"] = tmp[title_col].astype(str).str.split().str.len().fillna(0).astype(int)

	# ── Definir intervalos de palabras ──────────────────────────────────
	word_bins = [(1, 3), (4, 6), (7, 9), (10, 12), (13, 15), (16, 18), (19, 21), (22, 24)]

	# ── Definir buckets de duración ─────────────────────────────────────
	duration_buckets = [
		("le1min",  lambda s: s <= 60,               "≤1min"),
		("3_16min", lambda s: (s > 180) & (s <= 960), "3-16min"),
	]

	# ── Histograma ──────────────────────────────────────────────────────
	def _plot_histogram(res: dict, title: str, filename: str):
		if not res:
			return
		lbls = list(res.keys())
		pcts = [res[l] for l in lbls]

		fig, ax = plt.subplots(figsize=(12, 6))
		bars = ax.bar(
			range(len(lbls)), pcts,
			color=sns.color_palette("viridis", len(lbls)),
			edgecolor="black",
		)
		ax.set_xticks(range(len(lbls)))
		ax.set_xticklabels(lbls, rotation=30, ha="right", fontsize=10)
		ax.set_ylabel("% de victorias", fontsize=12)
		ax.set_xlabel("Nº de palabras en el título", fontsize=12)
		ax.set_title(title, fontsize=14, fontweight="bold")

		for bar, pct in zip(bars, pcts):
			ax.text(
				bar.get_x() + bar.get_width() / 2,
				bar.get_height() + 0.3,
				f"{pct:.2f}%",
				ha="center", va="bottom", fontsize=9, fontweight="bold",
			)

		plt.tight_layout()
		if save:
			img_path = output_dir / f"{filename}.png"
			fig.savefig(img_path, dpi=150)
			print(f"Imagen guardada: {img_path}")
		if show:
			plt.show()
		plt.close(fig)

	# ── Guardar TXT ─────────────────────────────────────────────────────
	def _save_txt(res: dict, filename: str, title: str, counts_d: dict | None = None):
		if not res:
			return
		txt_path = output_dir / f"{filename}.txt"
		with open(txt_path, "w", encoding="utf-8") as f:
			f.write(f"{title}\n")
			f.write(f"Rondas: {n_rounds:,}\n")
			f.write(f"{'Intervalo':<18}{'Count':>8}{'Victorias':>12}{'%':>10}\n")
			f.write("-" * 48 + "\n")
			for label, pct in res.items():
				wins_count = int(round(pct / 100 * n_rounds))
				count_val = counts_d.get(label, 0) if counts_d else 0
				f.write(f"{label:<18}{count_val:>8,}{wins_count:>12,}{pct:>9.2f}%\n")
		print(f"Datos guardados: {txt_path}")

	# ── Ejecutar simulación por cada bucket de duración ─────────────────
	print(f"\n{'='*60}")
	print(f"Monte Carlo – Nº palabras en título ({n_rounds:,} rondas)")
	print(f"{'='*60}")

	all_results = {}

	for bucket_name, bucket_cond, bucket_label in duration_buckets:
		print(f"\n→ Simulación para videos {bucket_label}...")
		tmp_filtered = tmp[bucket_cond(tmp["dur_s"])].copy()

		if tmp_filtered.empty:
			print(f"  Sin datos para bucket {bucket_label}; omitido.")
			continue

		# Agrupar videos por intervalo de palabras
		interval_videos: dict[str, np.ndarray] = {}
		for lo, hi in word_bins:
			mask = (tmp_filtered["word_count"] >= lo) & (tmp_filtered["word_count"] <= hi)
			label = f"[{lo}-{hi}]"
			views_arr = tmp_filtered.loc[mask, "views"].values
			if len(views_arr) > 0:
				interval_videos[label] = views_arr

		if len(interval_videos) < 2:
			print(f"  monte_carlo_title_words ({bucket_label}): Menos de 2 intervalos con videos; omitido.")
			continue

		labels = list(interval_videos.keys())
		arrays = [interval_videos[l] for l in labels]
		n_intervals = len(labels)
		counts = {labels[i]: len(arrays[i]) for i in range(n_intervals)}

		# Simulación vectorizada
		rng = np.random.default_rng(seed)
		wins = np.zeros(n_intervals, dtype=np.int64)

		idx_matrix = [rng.integers(0, len(arr), size=n_rounds) for arr in arrays]
		views_matrix = np.column_stack(
			[arr[idx] for arr, idx in zip(arrays, idx_matrix)]
		)

		winners = np.argmax(views_matrix, axis=1)
		for i in range(n_intervals):
			wins[i] = np.sum(winners == i)

		win_pct = wins / n_rounds * 100
		results = {labels[i]: float(win_pct[i]) for i in range(n_intervals)}

		_plot_histogram(
			results,
			f"Monte Carlo – Victorias por nº de palabras en título ({bucket_label})\n{n_rounds:,} rondas",
			f"monte_carlo_title_words_{bucket_name}",
		)
		_save_txt(
			results,
			f"monte_carlo_title_words_{bucket_name}",
			f"Monte Carlo – Nº de palabras en el título ({bucket_label})",
			counts_d=counts,
		)

		all_results[bucket_name] = results

	print(f"{'='*60}\n")
	return all_results

# Simulación Monte Carlo por número de tags
def monte_carlo_tag_count(
	df: pd.DataFrame,
	output_dir: str | None = None,
	n_rounds: int = 200_000,
	seed: int = 42,
	save: bool = True,
	show: bool = False,
) -> dict:
	"""
	Simulación Monte Carlo por cantidad de tags.

	Para cada ronda se elige un video aleatorio de cada intervalo de
	cantidad de tags; el intervalo cuyo video seleccionado tiene el mayor
	viewCount gana esa ronda. Se repite `n_rounds` veces y se genera un
	histograma con el porcentaje de victorias de cada intervalo.

	Intervalos: 0-5, 6-10, 11-15, 16-20, 21-30, 31-40, 41-60 tags.
	Los tags se cuentan separando la columna "tags" por el carácter "|".

	Se generan dos simulaciones separadas por duración:
	  1) Videos ≤1 min
	  2) Videos 1-16 min
	"""

	if output_dir is None:
		root = Path(__file__).resolve().parents[1]
		output_dir = root / "outputs" / "Random"
	else:
		output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# ── Resolver columna de tags ────────────────────────────────────────
	tag_col = None
	for c in ["tags", "etiquetas", "keywords"]:
		if c in df.columns:
			tag_col = c
			break
	if tag_col is None:
		for c in df.columns:
			if re.search(r'tag|etiqueta|keyword', c, re.I):
				tag_col = c
				break
	if tag_col is None:
		print("monte_carlo_tag_count: No se encontró columna de tags.")
		return {}

	# ── Resolver columna de vistas ──────────────────────────────────────
	view_candidates = [
		"viewCount", "views", "vistas", "visualizaciones",
		"view_count", "reproducciones",
	]
	view_col = None
	for c in view_candidates:
		if c in df.columns:
			view_col = c
			break
	if view_col is None:
		for c in df.columns:
			if re.search(r'view|vista|visual|reproduc', c, re.I):
				view_col = c
				break
	if view_col is None:
		print("monte_carlo_tag_count: No se encontró columna de vistas/viewCount.")
		return {}

	# ── Resolver columna de duración en segundos ───────────────────────
	duration_candidates = [
		"durationSeconds", "duracion_segundos", "duration_seconds",
		"duration_sec", "length_seconds",
	]
	dur_col = None
	for c in duration_candidates:
		if c in df.columns:
			dur_col = c
			break
	if dur_col is None:
		for c in df.columns:
			if re.search(r'duraci|duration|length', c, re.I):
				dur_col = c
				break
	if dur_col is None:
		print("monte_carlo_tag_count: No se encontró columna de duración.")
		return {}

	# ── Preparar datos ──────────────────────────────────────────────────
	tmp = df[[tag_col, view_col, dur_col]].copy()
	tmp["views"] = pd.to_numeric(tmp[view_col], errors="coerce")
	tmp["dur_s"] = pd.to_numeric(tmp[dur_col], errors="coerce")
	tmp = tmp.dropna(subset=["views", "dur_s"])
	tmp["views"] = tmp["views"].astype(int)
	tmp["dur_s"] = tmp["dur_s"].astype(float)

	# Contar tags: separar por "|" y contar elementos
	def _count_tags(val):
		if pd.isna(val) or str(val).strip() == "":
			return 0
		return len(str(val).split("|"))

	tmp["tag_count"] = tmp[tag_col].apply(_count_tags)

	# ── Definir intervalos de tags ──────────────────────────────────────
	tag_bins = [(0, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 40), (41, 60)]

	# ── Definir buckets de duración ─────────────────────────────────────
	duration_buckets = [
		("le1min",  lambda s: s <= 60,               "≤1min"),
		("3_16min", lambda s: (s > 180) & (s <= 960), "3-16min"),
	]

	# ── Histograma ──────────────────────────────────────────────────────
	def _plot_histogram(res: dict, title: str, filename: str):
		if not res:
			return
		lbls = list(res.keys())
		pcts = [res[l] for l in lbls]

		fig, ax = plt.subplots(figsize=(12, 6))
		bars = ax.bar(
			range(len(lbls)), pcts,
			color=sns.color_palette("viridis", len(lbls)),
			edgecolor="black",
		)
		ax.set_xticks(range(len(lbls)))
		ax.set_xticklabels(lbls, rotation=30, ha="right", fontsize=10)
		ax.set_ylabel("% de victorias", fontsize=12)
		ax.set_xlabel("Nº de tags", fontsize=12)
		ax.set_title(title, fontsize=14, fontweight="bold")

		for bar, pct in zip(bars, pcts):
			ax.text(
				bar.get_x() + bar.get_width() / 2,
				bar.get_height() + 0.3,
				f"{pct:.2f}%",
				ha="center", va="bottom", fontsize=9, fontweight="bold",
			)

		plt.tight_layout()
		if save:
			img_path = output_dir / f"{filename}.png"
			fig.savefig(img_path, dpi=150)
			print(f"Imagen guardada: {img_path}")
		if show:
			plt.show()
		plt.close(fig)

	# ── Guardar TXT ─────────────────────────────────────────────────────
	def _save_txt(res: dict, filename: str, title: str, counts_d: dict | None = None):
		if not res:
			return
		txt_path = output_dir / f"{filename}.txt"
		with open(txt_path, "w", encoding="utf-8") as f:
			f.write(f"{title}\n")
			f.write(f"Rondas: {n_rounds:,}\n")
			f.write(f"{'Intervalo':<18}{'Count':>8}{'Victorias':>12}{'%':>10}\n")
			f.write("-" * 48 + "\n")
			for label, pct in res.items():
				wins_count = int(round(pct / 100 * n_rounds))
				count_val = counts_d.get(label, 0) if counts_d else 0
				f.write(f"{label:<18}{count_val:>8,}{wins_count:>12,}{pct:>9.2f}%\n")
		print(f"Datos guardados: {txt_path}")

	# ── Ejecutar simulación por cada bucket de duración ─────────────────
	print(f"\n{'='*60}")
	print(f"Monte Carlo – Nº de tags ({n_rounds:,} rondas)")
	print(f"{'='*60}")

	all_results = {}

	for bucket_name, bucket_cond, bucket_label in duration_buckets:
		print(f"\n→ Simulación para videos {bucket_label}...")
		tmp_filtered = tmp[bucket_cond(tmp["dur_s"])].copy()

		if tmp_filtered.empty:
			print(f"  Sin datos para bucket {bucket_label}; omitido.")
			continue

		# Agrupar videos por intervalo de tags
		interval_videos: dict[str, np.ndarray] = {}
		for lo, hi in tag_bins:
			mask = (tmp_filtered["tag_count"] >= lo) & (tmp_filtered["tag_count"] <= hi)
			label = f"[{lo}-{hi}]"
			views_arr = tmp_filtered.loc[mask, "views"].values
			if len(views_arr) > 0:
				interval_videos[label] = views_arr

		if len(interval_videos) < 2:
			print(f"  monte_carlo_tag_count ({bucket_label}): Menos de 2 intervalos con videos; omitido.")
			continue

		labels = list(interval_videos.keys())
		arrays = [interval_videos[l] for l in labels]
		n_intervals = len(labels)
		counts = {labels[i]: len(arrays[i]) for i in range(n_intervals)}

		# Simulación vectorizada
		rng = np.random.default_rng(seed)
		wins = np.zeros(n_intervals, dtype=np.int64)

		idx_matrix = [rng.integers(0, len(arr), size=n_rounds) for arr in arrays]
		views_matrix = np.column_stack(
			[arr[idx] for arr, idx in zip(arrays, idx_matrix)]
		)

		winners = np.argmax(views_matrix, axis=1)
		for i in range(n_intervals):
			wins[i] = np.sum(winners == i)

		win_pct = wins / n_rounds * 100
		results = {labels[i]: float(win_pct[i]) for i in range(n_intervals)}

		_plot_histogram(
			results,
			f"Monte Carlo – Victorias por nº de tags ({bucket_label})\n{n_rounds:,} rondas",
			f"monte_carlo_tag_count_{bucket_name}",
		)
		_save_txt(
			results,
			f"monte_carlo_tag_count_{bucket_name}",
			f"Monte Carlo – Nº de tags ({bucket_label})",
			counts_d=counts,
		)

		all_results[bucket_name] = results

	print(f"{'='*60}\n")
	return all_results

# Simulación Monte Carlo por longitud de descripción en caracteres
def monte_carlo_description_length(
	df: pd.DataFrame,
	output_dir: str | None = None,
	n_rounds: int = 200_000,
	seed: int = 42,
	save: bool = True,
	show: bool = False,
) -> dict:
	"""
	Simulación Monte Carlo por longitud de descripción (caracteres).

	Para cada ronda se elige un video aleatorio de cada intervalo de
	longitud de descripción; el intervalo cuyo video seleccionado tiene el
	mayor viewCount gana esa ronda. Se repite `n_rounds` veces y se genera
	un histograma con el porcentaje de victorias de cada intervalo.

	Intervalos: 0-100, 100-250, 250-500, 500-750, 750-1000, 1000-1500,
	            1500-2000, 2000-3000, 3000-4000, 4000-5000, 5000+ caracteres.

	Se generan dos simulaciones separadas por duración:
	  1) Videos ≤1 min
	  2) Videos 1-16 min
	"""

	if output_dir is None:
		root = Path(__file__).resolve().parents[1]
		output_dir = root / "outputs" / "Random"
	else:
		output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# ── Resolver columna de descripción ─────────────────────────────────
	desc_col = None
	for c in ["description", "descripcion", "descripción", "desc", "about"]:
		if c in df.columns:
			desc_col = c
			break
	if desc_col is None:
		for c in df.columns:
			if re.search(r'descrip|about|sinopsis', c, re.I):
				desc_col = c
				break
	if desc_col is None:
		print("monte_carlo_description_length: No se encontró columna de descripción.")
		return {}

	# ── Resolver columna de vistas ──────────────────────────────────────
	view_candidates = [
		"viewCount", "views", "vistas", "visualizaciones",
		"view_count", "reproducciones",
	]
	view_col = None
	for c in view_candidates:
		if c in df.columns:
			view_col = c
			break
	if view_col is None:
		for c in df.columns:
			if re.search(r'view|vista|visual|reproduc', c, re.I):
				view_col = c
				break
	if view_col is None:
		print("monte_carlo_description_length: No se encontró columna de vistas/viewCount.")
		return {}

	# ── Resolver columna de duración en segundos ───────────────────────
	duration_candidates = [
		"durationSeconds", "duracion_segundos", "duration_seconds",
		"duration_sec", "length_seconds",
	]
	dur_col = None
	for c in duration_candidates:
		if c in df.columns:
			dur_col = c
			break
	if dur_col is None:
		for c in df.columns:
			if re.search(r'duraci|duration|length', c, re.I):
				dur_col = c
				break
	if dur_col is None:
		print("monte_carlo_description_length: No se encontró columna de duración.")
		return {}

	# ── Preparar datos ──────────────────────────────────────────────────
	tmp = df[[desc_col, view_col, dur_col]].copy()
	tmp["views"] = pd.to_numeric(tmp[view_col], errors="coerce")
	tmp["dur_s"] = pd.to_numeric(tmp[dur_col], errors="coerce")
	tmp = tmp.dropna(subset=["views", "dur_s"])
	tmp["views"] = tmp["views"].astype(int)
	tmp["dur_s"] = tmp["dur_s"].astype(float)

	# Calcular longitud de descripción en caracteres
	tmp["desc_len"] = tmp[desc_col].apply(
		lambda x: len(str(x)) if pd.notna(x) and str(x).strip() not in ('', 'nan') else 0
	)

	# ── Definir intervalos de longitud de descripción ──────────────────
	# Tuplas (lo, hi) donde hi=None significa sin límite superior
	desc_bins = [
		(0, 100), (100, 250), (250, 500), (500, 750), (750, 1000),
		(1000, 1500), (1500, 2000), (2000, 3000), (3000, 4000),
		(4000, 5000), (5000, None),
	]

	# ── Definir buckets de duración ─────────────────────────────────────
	duration_buckets = [
		("le1min",  lambda s: s <= 60,               "≤1min"),
		("3_16min", lambda s: (s > 180) & (s <= 960), "3-16min"),
	]

	# ── Histograma ──────────────────────────────────────────────────────
	def _plot_histogram(res: dict, title: str, filename: str):
		if not res:
			return
		lbls = list(res.keys())
		pcts = [res[l] for l in lbls]

		fig, ax = plt.subplots(figsize=(14, 6))
		bars = ax.bar(
			range(len(lbls)), pcts,
			color=sns.color_palette("viridis", len(lbls)),
			edgecolor="black",
		)
		ax.set_xticks(range(len(lbls)))
		ax.set_xticklabels(lbls, rotation=35, ha="right", fontsize=9)
		ax.set_ylabel("% de victorias", fontsize=12)
		ax.set_xlabel("Intervalo de longitud de descripción (caracteres)", fontsize=12)
		ax.set_title(title, fontsize=14, fontweight="bold")

		for bar, pct in zip(bars, pcts):
			ax.text(
				bar.get_x() + bar.get_width() / 2,
				bar.get_height() + 0.3,
				f"{pct:.2f}%",
				ha="center", va="bottom", fontsize=9, fontweight="bold",
			)

		plt.tight_layout()
		if save:
			img_path = output_dir / f"{filename}.png"
			fig.savefig(img_path, dpi=150)
			print(f"Imagen guardada: {img_path}")
		if show:
			plt.show()
		plt.close(fig)

	# ── Guardar TXT ─────────────────────────────────────────────────────
	def _save_txt(res: dict, filename: str, title: str, counts_d: dict | None = None):
		if not res:
			return
		txt_path = output_dir / f"{filename}.txt"
		with open(txt_path, "w", encoding="utf-8") as f:
			f.write(f"{title}\n")
			f.write(f"Rondas: {n_rounds:,}\n")
			f.write(f"{'Intervalo':<18}{'Count':>8}{'Victorias':>12}{'%':>10}\n")
			f.write("-" * 48 + "\n")
			for label, pct in res.items():
				wins_count = int(round(pct / 100 * n_rounds))
				count_val = counts_d.get(label, 0) if counts_d else 0
				f.write(f"{label:<18}{count_val:>8,}{wins_count:>12,}{pct:>9.2f}%\n")
		print(f"Datos guardados: {txt_path}")

	# ── Ejecutar simulación por cada bucket de duración ─────────────────
	print(f"\n{'='*60}")
	print(f"Monte Carlo – Longitud de descripción ({n_rounds:,} rondas)")
	print(f"{'='*60}")

	all_results = {}

	for bucket_name, bucket_cond, bucket_label in duration_buckets:
		print(f"\n→ Simulación para videos {bucket_label}...")
		tmp_filtered = tmp[bucket_cond(tmp["dur_s"])].copy()

		if tmp_filtered.empty:
			print(f"  Sin datos para bucket {bucket_label}; omitido.")
			continue

		# Agrupar videos por intervalo de longitud de descripción
		interval_videos: dict[str, np.ndarray] = {}
		for lo, hi in desc_bins:
			if hi is None:
				mask = tmp_filtered["desc_len"] >= lo
				label = f"{lo}+"
			else:
				mask = (tmp_filtered["desc_len"] >= lo) & (tmp_filtered["desc_len"] < hi)
				label = f"{lo}-{hi}"
			views_arr = tmp_filtered.loc[mask, "views"].values
			if len(views_arr) > 0:
				interval_videos[label] = views_arr

		if len(interval_videos) < 2:
			print(f"  monte_carlo_description_length ({bucket_label}): Menos de 2 intervalos con videos; omitido.")
			continue

		labels = list(interval_videos.keys())
		arrays = [interval_videos[l] for l in labels]
		n_intervals = len(labels)
		counts = {labels[i]: len(arrays[i]) for i in range(n_intervals)}

		# Simulación vectorizada
		rng = np.random.default_rng(seed)
		wins = np.zeros(n_intervals, dtype=np.int64)

		idx_matrix = [rng.integers(0, len(arr), size=n_rounds) for arr in arrays]
		views_matrix = np.column_stack(
			[arr[idx] for arr, idx in zip(arrays, idx_matrix)]
		)

		winners = np.argmax(views_matrix, axis=1)
		for i in range(n_intervals):
			wins[i] = np.sum(winners == i)

		win_pct = wins / n_rounds * 100
		results = {labels[i]: float(win_pct[i]) for i in range(n_intervals)}

		_plot_histogram(
			results,
			f"Monte Carlo – Victorias por longitud de descripción ({bucket_label})\n{n_rounds:,} rondas",
			f"monte_carlo_description_length_{bucket_name}",
		)
		_save_txt(
			results,
			f"monte_carlo_description_length_{bucket_name}",
			f"Monte Carlo – Longitud de descripción ({bucket_label})",
			counts_d=counts,
		)

		all_results[bucket_name] = results

	print(f"{'='*60}\n")
	return all_results

# Monte Carlo – Intervalos de 2 h por día de la semana (separado por duración)
def monte_carlo_hourly_by_weekday(
	df: pd.DataFrame,
	output_dir: str | None = None,
	n_rounds: int = 200_000,
	seed: int = 42,
	save: bool = True,
	show: bool = False,
) -> dict:
	"""
	Simulación Monte Carlo por intervalos de 2 h, ejecutada por separado para
	cada día de la semana y cada bucket de duración (≤1 min y 3-16 min).

	Para un día dado se divide la jornada (hora militar) en intervalos de 2 h:
	(0-2], (2-4], …, (22-24].  En cada ronda se elige un video aleatorio de
	cada intervalo que contenga al menos un video y el intervalo cuyo video
	seleccionado tiene el mayor *viewCount* gana esa ronda.  Los intervalos
	vacíos se ignoran; la ronda se lleva a cabo igualmente.

	Se generan dos carpetas dentro de ``output_dir`` (por defecto
	*outputs/Random*):

	* ``menos_1min/``  – videos con duración ≤ 60 s
	* ``3_16min/``     – videos con 60*3 < duración ≤ 960 s

	Dentro de cada carpeta se guarda un archivo TXT y una imagen PNG por día.
	"""

	if output_dir is None:
		root = Path(__file__).resolve().parents[1]
		output_dir = root / "outputs" / "Random"
	else:
		output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# ── Resolver columnas ────────────────────────────────────────────────
	# Fecha/hora
	date_candidates = [
		"publishedAt", "published_at", "fecha_publicacion",
		"fecha", "date", "upload_date",
	]
	date_col = None
	for c in date_candidates:
		if c in df.columns:
			date_col = c
			break
	if date_col is None:
		for c in df.columns:
			if re.search(r'publish|upload|date|time|fecha|publica|publicación|publicacion', c, re.I):
				date_col = c
				break
	if date_col is None:
		print("monte_carlo_hourly_by_weekday: No se encontró columna de fecha/hora.")
		return {}

	# Vistas
	view_candidates = [
		"viewCount", "views", "vistas", "visualizaciones",
		"view_count", "reproducciones",
	]
	view_col = None
	for c in view_candidates:
		if c in df.columns:
			view_col = c
			break
	if view_col is None:
		for c in df.columns:
			if re.search(r'view|vista|visual|reproduc', c, re.I):
				view_col = c
				break
	if view_col is None:
		print("monte_carlo_hourly_by_weekday: No se encontró columna de vistas.")
		return {}

	# Duración
	duration_candidates = [
		"durationSeconds", "duracion_segundos", "duration_seconds",
		"duration_sec", "length_seconds",
	]
	dur_col = None
	for c in duration_candidates:
		if c in df.columns:
			dur_col = c
			break
	if dur_col is None:
		for c in df.columns:
			if re.search(r'duraci|duration|length', c, re.I):
				dur_col = c
				break
	if dur_col is None:
		print("monte_carlo_hourly_by_weekday: No se encontró columna de duración.")
		return {}

	# ── Preparar datos ──────────────────────────────────────────────────
	tmp = df[[date_col, view_col, dur_col]].copy()
	try:
		tmp[date_col] = pd.to_datetime(tmp[date_col], format='mixed', errors='coerce')
	except Exception:
		tmp[date_col] = pd.to_datetime(tmp[date_col].astype(str), format='mixed', errors='coerce')
	tmp["views"] = pd.to_numeric(tmp[view_col], errors="coerce")
	tmp["dur_s"] = pd.to_numeric(tmp[dur_col], errors="coerce")
	tmp = tmp.dropna(subset=[date_col, "views", "dur_s"])
	if tmp.empty:
		print("monte_carlo_hourly_by_weekday: No hay datos válidos.")
		return {}
	tmp["views"] = tmp["views"].astype(int)
	tmp["dur_s"] = tmp["dur_s"].astype(float)
	tmp["weekday"] = tmp[date_col].dt.dayofweek   # Monday=0 … Sunday=6
	tmp["hour_frac"] = (
		tmp[date_col].dt.hour.astype(float)
		+ tmp[date_col].dt.minute.astype(float) / 60.0
		+ tmp[date_col].dt.second.astype(float) / 3600.0
	)

	weekday_labels = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
	hour_bins = [(h, h + 2) for h in range(0, 24, 2)]  # 12 intervalos de 2 h
	hour_labels = [f"({lo:02d}-{hi:02d}]" for lo, hi in hour_bins]

	duration_buckets = [
		("menos_1min", lambda s: s <= 60,               "≤1min"),
		("3_16min",    lambda s: (s > 180) & (s <= 960), "3-16min"),
	]

	# ── Funciones internas ──────────────────────────────────────────────
	def _plot_histogram(res: dict, title: str, filepath: Path):
		if not res:
			return
		lbls = list(res.keys())
		pcts = [res[l] for l in lbls]

		fig, ax = plt.subplots(figsize=(14, 6))
		bars = ax.bar(
			range(len(lbls)), pcts,
			color=sns.color_palette("viridis", len(lbls)),
			edgecolor="black",
		)
		ax.set_xticks(range(len(lbls)))
		ax.set_xticklabels(lbls, rotation=30, ha="right", fontsize=10)
		ax.set_ylabel("% de victorias", fontsize=12)
		ax.set_xlabel("Intervalo horario (hora militar)", fontsize=12)
		ax.set_title(title, fontsize=13, fontweight="bold")

		for bar, pct in zip(bars, pcts):
			ax.text(
				bar.get_x() + bar.get_width() / 2,
				bar.get_height() + 0.3,
				f"{pct:.2f}%",
				ha="center", va="bottom", fontsize=9, fontweight="bold",
			)

		plt.tight_layout()
		if save:
			fig.savefig(filepath, dpi=150)
			print(f"  Imagen guardada: {filepath}")
		if show:
			plt.show()
		plt.close(fig)

	def _save_txt(res: dict, filepath: Path, title: str, counts_d: dict | None = None):
		if not res:
			return
		with open(filepath, "w", encoding="utf-8") as f:
			f.write(f"{title}\n")
			f.write(f"Rondas: {n_rounds:,}\n")
			f.write(f"{'Intervalo':<18}{'Count':>8}{'Victorias':>12}{'%':>10}\n")
			f.write("-" * 48 + "\n")
			for label, pct in res.items():
				wins_count = int(round(pct / 100 * n_rounds))
				count_val = counts_d.get(label, 0) if counts_d else 0
				f.write(f"{label:<18}{count_val:>8,}{wins_count:>12,}{pct:>9.2f}%\n")
		print(f"  Datos guardados: {filepath}")

	# ── Simulación ──────────────────────────────────────────────────────
	print(f"\n{'='*60}")
	print(f"Monte Carlo – Intervalos de 2 h por día de la semana ({n_rounds:,} rondas)")
	print(f"{'='*60}")

	all_results: dict = {}

	for bucket_name, bucket_cond, bucket_label in duration_buckets:
		bucket_dir = output_dir / bucket_name
		bucket_dir.mkdir(parents=True, exist_ok=True)

		print(f"\n■ Bucket {bucket_label}  →  {bucket_dir}")
		tmp_bucket = tmp[bucket_cond(tmp["dur_s"])].copy()
		if tmp_bucket.empty:
			print(f"  Sin datos para {bucket_label}; omitido.")
			continue

		bucket_results: dict = {}

		for wd_idx, wd_label in enumerate(weekday_labels):
			day_df = tmp_bucket[tmp_bucket["weekday"] == wd_idx]
			if day_df.empty:
				print(f"  {wd_label}: sin videos; omitido.")
				continue

			# Agrupar por intervalo de 2 h – notación (a, b]
			interval_videos: dict[str, np.ndarray] = {}
			for (lo, hi), hlabel in zip(hour_bins, hour_labels):
				if lo == 0:
					mask = (day_df["hour_frac"] >= lo) & (day_df["hour_frac"] <= hi)
				else:
					mask = (day_df["hour_frac"] > lo) & (day_df["hour_frac"] <= hi)
				varr = day_df.loc[mask, "views"].values
				if len(varr) > 0:
					interval_videos[hlabel] = varr

			if len(interval_videos) < 2:
				print(f"  {wd_label}: menos de 2 intervalos con videos; omitido.")
				continue

			labels = list(interval_videos.keys())
			arrays = [interval_videos[l] for l in labels]
			n_int = len(labels)
			counts = {labels[i]: len(arrays[i]) for i in range(n_int)}

			# Simulación vectorizada
			rng = np.random.default_rng(seed)
			wins = np.zeros(n_int, dtype=np.int64)

			idx_matrix = [rng.integers(0, len(arr), size=n_rounds) for arr in arrays]
			views_matrix = np.column_stack(
				[arr[idx] for arr, idx in zip(arrays, idx_matrix)]
			)  # (n_rounds, n_int)

			winners = np.argmax(views_matrix, axis=1)
			for i in range(n_int):
				wins[i] = np.sum(winners == i)

			win_pct = wins / n_rounds * 100
			results = {labels[i]: float(win_pct[i]) for i in range(n_int)}

			# Guardar
			safe_day = wd_label.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
			fname = f"monte_carlo_hourly_{safe_day}"

			_plot_histogram(
				results,
				f"Monte Carlo – Victorias por franja horaria · {wd_label} ({bucket_label})\n{n_rounds:,} rondas",
				bucket_dir / f"{fname}.png",
			)
			_save_txt(
				results,
				bucket_dir / f"{fname}.txt",
				f"Monte Carlo – Franja horaria · {wd_label} ({bucket_label})",
				counts_d=counts,
			)

			bucket_results[wd_label] = results

		all_results[bucket_name] = bucket_results

	print(f"{'='*60}\n")
	return all_results

# Monte Carlo – Intervalos de 2 h (sin separar por día de la semana, solo por duración)
def monte_carlo_hourly_2h(
	df: pd.DataFrame,
	output_dir: str | None = None,
	n_rounds: int = 200_000,
	seed: int = 42,
	save: bool = True,
	show: bool = False,
) -> dict:
	"""
	Simulación Monte Carlo por intervalos de 2 h (sin separar por día de la
	semana).  Genera **una imagen** por bucket de duración:

	* ``monte_carlo_hourly_2h_le1min.png``  – videos con duración ≤ 60 s
	* ``monte_carlo_hourly_2h_3_16min.png`` – videos con 60*3 < duración ≤ 960 s

	En cada ronda se elige un video aleatorio de cada intervalo de 2 h que
	contenga al menos un video y el intervalo cuyo video seleccionado tiene el
	mayor *viewCount* gana esa ronda.
	"""

	if output_dir is None:
		root = Path(__file__).resolve().parents[1]
		output_dir = root / "outputs" / "Random"
	else:
		output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# ── Resolver columnas ────────────────────────────────────────────────
	date_candidates = [
		"publishedAt", "published_at", "fecha_publicacion",
		"fecha", "date", "upload_date",
	]
	date_col = None
	for c in date_candidates:
		if c in df.columns:
			date_col = c
			break
	if date_col is None:
		for c in df.columns:
			if re.search(r'publish|upload|date|time|fecha|publica|publicación|publicacion', c, re.I):
				date_col = c
				break
	if date_col is None:
		print("monte_carlo_hourly_2h: No se encontró columna de fecha/hora.")
		return {}

	view_candidates = [
		"viewCount", "views", "vistas", "visualizaciones",
		"view_count", "reproducciones",
	]
	view_col = None
	for c in view_candidates:
		if c in df.columns:
			view_col = c
			break
	if view_col is None:
		for c in df.columns:
			if re.search(r'view|vista|visual|reproduc', c, re.I):
				view_col = c
				break
	if view_col is None:
		print("monte_carlo_hourly_2h: No se encontró columna de vistas.")
		return {}

	duration_candidates = [
		"durationSeconds", "duracion_segundos", "duration_seconds",
		"duration_sec", "length_seconds",
	]
	dur_col = None
	for c in duration_candidates:
		if c in df.columns:
			dur_col = c
			break
	if dur_col is None:
		for c in df.columns:
			if re.search(r'duraci|duration|length', c, re.I):
				dur_col = c
				break
	if dur_col is None:
		print("monte_carlo_hourly_2h: No se encontró columna de duración.")
		return {}

	# ── Preparar datos ──────────────────────────────────────────────────
	tmp = df[[date_col, view_col, dur_col]].copy()
	try:
		tmp[date_col] = pd.to_datetime(tmp[date_col], format='mixed', errors='coerce')
	except Exception:
		tmp[date_col] = pd.to_datetime(tmp[date_col].astype(str), format='mixed', errors='coerce')
	tmp["views"] = pd.to_numeric(tmp[view_col], errors="coerce")
	tmp["dur_s"] = pd.to_numeric(tmp[dur_col], errors="coerce")
	tmp = tmp.dropna(subset=[date_col, "views", "dur_s"])
	if tmp.empty:
		print("monte_carlo_hourly_2h: No hay datos válidos.")
		return {}
	tmp["views"] = tmp["views"].astype(int)
	tmp["dur_s"] = tmp["dur_s"].astype(float)
	tmp["hour_frac"] = (
		tmp[date_col].dt.hour.astype(float)
		+ tmp[date_col].dt.minute.astype(float) / 60.0
		+ tmp[date_col].dt.second.astype(float) / 3600.0
	)

	hour_bins = [(h, h + 2) for h in range(0, 24, 2)]  # 12 intervalos
	hour_labels = [f"({lo:02d}-{hi:02d}]" for lo, hi in hour_bins]

	duration_buckets = [
		("le1min",  lambda s: s <= 60,               "≤1min"),
		("3_16min", lambda s: (s > 180) & (s <= 960), "3-16min"),
	]

	# ── Funciones internas ──────────────────────────────────────────────
	def _plot(res: dict, title: str, filepath: Path, counts_d: dict):
		if not res:
			return
		lbls = list(res.keys())
		pcts = [res[l] for l in lbls]

		fig, ax = plt.subplots(figsize=(14, 6))
		bars = ax.bar(
			range(len(lbls)), pcts,
			color=sns.color_palette("viridis", len(lbls)),
			edgecolor="black",
		)
		ax.set_xticks(range(len(lbls)))
		ax.set_xticklabels(lbls, rotation=30, ha="right", fontsize=10)
		ax.set_ylabel("% de victorias", fontsize=12)
		ax.set_xlabel("Intervalo horario (hora militar)", fontsize=12)
		ax.set_title(title, fontsize=13, fontweight="bold")

		for bar, pct in zip(bars, pcts):
			ax.text(
				bar.get_x() + bar.get_width() / 2,
				bar.get_height() + 0.3,
				f"{pct:.2f}%",
				ha="center", va="bottom", fontsize=9, fontweight="bold",
			)

		plt.tight_layout()
		if save:
			fig.savefig(filepath, dpi=150)
			print(f"  Imagen guardada: {filepath}")
		if show:
			plt.show()
		plt.close(fig)

	def _save_txt(res: dict, filepath: Path, title: str, counts_d: dict):
		if not res:
			return
		with open(filepath, "w", encoding="utf-8") as f:
			f.write(f"{title}\n")
			f.write(f"Rondas: {n_rounds:,}\n")
			f.write(f"{'Intervalo':<18}{'Count':>8}{'Victorias':>12}{'%':>10}\n")
			f.write("-" * 48 + "\n")
			for label, pct in res.items():
				wins_count = int(round(pct / 100 * n_rounds))
				count_val = counts_d.get(label, 0)
				f.write(f"{label:<18}{count_val:>8,}{wins_count:>12,}{pct:>9.2f}%\n")
		print(f"  Datos guardados: {filepath}")

	# ── Simulación ──────────────────────────────────────────────────────
	print(f"\n{'='*60}")
	print(f"Monte Carlo – Intervalos de 2 h global ({n_rounds:,} rondas)")
	print(f"{'='*60}")

	all_results: dict = {}

	for bucket_name, bucket_cond, bucket_label in duration_buckets:
		print(f"\n■ Bucket {bucket_label}")
		tmp_bucket = tmp[bucket_cond(tmp["dur_s"])].copy()
		if tmp_bucket.empty:
			print(f"  Sin datos para {bucket_label}; omitido.")
			continue

		# Agrupar por intervalo de 2 h – notación (a, b]
		interval_videos: dict[str, np.ndarray] = {}
		for (lo, hi), hlabel in zip(hour_bins, hour_labels):
			if lo == 0:
				mask = (tmp_bucket["hour_frac"] >= lo) & (tmp_bucket["hour_frac"] <= hi)
			else:
				mask = (tmp_bucket["hour_frac"] > lo) & (tmp_bucket["hour_frac"] <= hi)
			varr = tmp_bucket.loc[mask, "views"].values
			if len(varr) > 0:
				interval_videos[hlabel] = varr

		if len(interval_videos) < 2:
			print(f"  Menos de 2 intervalos con videos; omitido.")
			continue

		labels = list(interval_videos.keys())
		arrays = [interval_videos[l] for l in labels]
		n_int = len(labels)
		counts = {labels[i]: len(arrays[i]) for i in range(n_int)}

		# Simulación vectorizada
		rng = np.random.default_rng(seed)
		wins = np.zeros(n_int, dtype=np.int64)

		idx_matrix = [rng.integers(0, len(arr), size=n_rounds) for arr in arrays]
		views_matrix = np.column_stack(
			[arr[idx] for arr, idx in zip(arrays, idx_matrix)]
		)

		winners = np.argmax(views_matrix, axis=1)
		for i in range(n_int):
			wins[i] = np.sum(winners == i)

		win_pct = wins / n_rounds * 100
		results = {labels[i]: float(win_pct[i]) for i in range(n_int)}

		# Guardar imagen y txt
		fname = f"monte_carlo_hourly_2h_{bucket_name}"
		_plot(
			results,
			f"Monte Carlo – Victorias por franja horaria ({bucket_label})\n{n_rounds:,} rondas",
			output_dir / f"{fname}.png",
			counts,
		)
		_save_txt(
			results,
			output_dir / f"{fname}.txt",
			f"Monte Carlo – Franja horaria ({bucket_label})",
			counts,
		)

		all_results[bucket_name] = results

	print(f"{'='*60}\n")
	return all_results

# ── Monte Carlo: comparación directa entre dos datasets ────────────────────
def monte_carlo_two_datasets(
	df_a: pd.DataFrame,
	df_b: pd.DataFrame,
	label_a: str = "filtered",
	label_b: str = "remainder",
	output_dir: str | None = None,
	file_suffix: str | None = None,
	n_rounds: int = 200_000,
	seed: int = 42,
	save: bool = True,
	show: bool = False,
) -> dict:
	"""
	Simulación Monte Carlo para comparar dos datasets por viewCount.

	En cada ronda se elige aleatoriamente un video de `df_a` y otro de `df_b`;
	gana el que tenga mayor viewCount.  Se acumulan resultados a lo largo de
	`n_rounds` rondas y se genera un gráfico de barras con el % de victorias
	para cada dataset más el % de empates.

	Args:
		df_a:        Dataset A (p. ej. videos filtrados / shorts).
		df_b:        Dataset B (p. ej. videos restantes / largos).
		label_a:     Etiqueta para el dataset A en el gráfico.
		label_b:     Etiqueta para el dataset B en el gráfico.
		output_dir:  Directorio de salida.  Por defecto outputs/Random.
		file_suffix: Sufijo para el nombre de archivo de salida.
		n_rounds:    Número de rondas de la simulación (default 200 000).
		seed:        Semilla para reproducibilidad.
		save:        Si True, guarda PNG y TXT.
		show:        Si True, muestra el gráfico en pantalla.

	Returns:
		dict con claves 'label_a', 'label_b', 'ties' y su % de victorias.
	"""

	if output_dir is None:
		root = Path(__file__).resolve().parents[1]
		output_dir = root / "outputs" / "Random"
	else:
		output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# ── Resolver columna de vistas ──────────────────────────────────────
	view_candidates = [
		"viewCount", "views", "vistas", "visualizaciones",
		"view_count", "reproducciones",
	]

	def _find_view_col(df: pd.DataFrame) -> str | None:
		for c in view_candidates:
			if c in df.columns:
				return c
		for c in df.columns:
			if re.search(r'view|vista|visual|reproduc', c, re.I):
				return c
		return None

	view_col_a = _find_view_col(df_a)
	view_col_b = _find_view_col(df_b)

	if view_col_a is None:
		print("monte_carlo_two_datasets: No se encontró columna de vistas en df_a.")
		return {}
	if view_col_b is None:
		print("monte_carlo_two_datasets: No se encontró columna de vistas en df_b.")
		return {}

	# ── Preparar arrays de vistas ───────────────────────────────────────
	views_a = pd.to_numeric(df_a[view_col_a], errors="coerce").dropna().astype(np.int64).values
	views_b = pd.to_numeric(df_b[view_col_b], errors="coerce").dropna().astype(np.int64).values

	if len(views_a) == 0:
		print("monte_carlo_two_datasets: df_a no tiene vistas válidas.")
		return {}
	if len(views_b) == 0:
		print("monte_carlo_two_datasets: df_b no tiene vistas válidas.")
		return {}

	# ── Simulación vectorizada ──────────────────────────────────────────
	print(f"\n{'='*60}")
	print(f"Monte Carlo – {label_a} vs {label_b} ({n_rounds:,} rondas)")
	print(f"  {label_a}: {len(views_a):,} videos  |  {label_b}: {len(views_b):,} videos")
	print(f"{'='*60}")

	rng = np.random.default_rng(seed)
	idx_a = rng.integers(0, len(views_a), size=n_rounds)
	idx_b = rng.integers(0, len(views_b), size=n_rounds)

	sampled_a = views_a[idx_a]
	sampled_b = views_b[idx_b]

	wins_a   = int(np.sum(sampled_a > sampled_b))
	wins_b   = int(np.sum(sampled_b > sampled_a))
	ties     = int(np.sum(sampled_a == sampled_b))

	pct_a    = wins_a   / n_rounds * 100
	pct_b    = wins_b   / n_rounds * 100
	pct_ties = ties     / n_rounds * 100

	print(f"  {label_a:<20} → {pct_a:>6.2f}%  ({wins_a:,} victorias)")
	print(f"  {label_b:<20} → {pct_b:>6.2f}%  ({wins_b:,} victorias)")
	print(f"  {'ties':<20} → {pct_ties:>6.2f}%  ({ties:,} empates)")
	print(f"{'='*60}\n")

	results = {label_a: pct_a, label_b: pct_b, "ties": pct_ties}

	# ── Nombre de archivo base ──────────────────────────────────────────
	base_name = f"monte_carlo_{label_a}_vs_{label_b}"
	if file_suffix:
		base_name = f"{base_name}_{file_suffix}"

	# ── Gráfico ─────────────────────────────────────────────────────────
	if save or show:
		labels_plot = [label_a, label_b, "ties"]
		pcts_plot   = [pct_a,   pct_b,   pct_ties]
		# Colores: azul para A, naranja para B, gris muy claro para ties
		colors = ["#1f77b4", "#d2874a", "#7f7f7f"]

		fig, ax = plt.subplots(figsize=(9, 6))
		bars = ax.bar(labels_plot, pcts_plot, color=colors, width=0.5)

		ax.set_ylim(0, 100)
		ax.set_ylabel("% victorias", fontsize=12)
		ax.set_title(
			f"Monte Carlo: {n_rounds:,} rondas — % victorias",
			fontsize=13, fontweight="bold",
		)
		ax.spines["top"].set_visible(False)
		ax.spines["right"].set_visible(False)

		# Etiquetas sobre barras
		for bar, pct in zip(bars, pcts_plot):
			ax.text(
				bar.get_x() + bar.get_width() / 2,
				bar.get_height() + 0.6,
				f"{pct:.2f}%",
				ha="center", va="bottom", fontsize=11, fontweight="bold",
			)

		plt.tight_layout()

		if save:
			img_path = output_dir / f"{base_name}.png"
			fig.savefig(img_path, dpi=150)
			print(f"Imagen guardada: {img_path}")

		if show:
			plt.show()

		plt.close(fig)

	# ── TXT ─────────────────────────────────────────────────────────────
	if save:
		txt_path = output_dir / f"{base_name}.txt"
		with open(txt_path, "w", encoding="utf-8") as f:
			f.write(f"Monte Carlo: {label_a} vs {label_b}\n")
			f.write(f"Rondas: {n_rounds:,}\n")
			f.write(f"{'Dataset':<20}{'Victorias':>12}{'%':>10}\n")
			f.write("-" * 42 + "\n")
			f.write(f"{label_a:<20}{wins_a:>12,}{pct_a:>9.2f}%\n")
			f.write(f"{label_b:<20}{wins_b:>12,}{pct_b:>9.2f}%\n")
			f.write(f"{'ties':<20}{ties:>12,}{pct_ties:>9.2f}%\n")
		print(f"Datos guardados: {txt_path}")

	return results

# Filtrado de videos por múltiples propiedades
def filter_videos_by_properties(df, title_word_intervals=None, hour_intervals=None,
								days_list=None, duration_intervals=None,
								tag_count_intervals=None, duration_filter_mode=3):
	"""
	Filtra `df` devolviendo (filtrados, restantes).

	Selecciona las filas que cumplen TODOS los parámetros. Para cada parámetro
	que recibe una lista de intervalos se acepta que la fila cumpla al menos
	uno de los intervalos de esa lista. Parámetros vacíos/None se ignoran
	(no filtran).

	Args:
		df: DataFrame de entrada.
		title_word_intervals: lista de tuplas (a,b) para cantidad de palabras del título.
		hour_intervals: lista de tuplas (a,b) para hora del día (0-23). Soporta intervalos
			"envolventes" (ej. (22,2)).
		days_list: lista de días en español, p.ej. ["Lunes","Martes",...].
		duration_intervals: lista de tuplas (a,b) en segundos.
		tag_count_intervals: lista de tuplas (a,b) para cantidad de tags.
		duration_filter_mode: int (1|2|3|4).
			1 → trabaja solo con videos de duración <= 60 s  (≤ 1 min).
			2 → trabaja solo con videos de duración (60, 960] s  (1 min–16 min].
			3 → sin prefiltro de duración (por defecto).
			4 → trabaja solo con videos de duración < 180 s  (< 3 min).

	Returns:
		(filtered_df, remainder_df)
	"""
	df = df.copy()

	# ── Helpers para detectar columnas ───────────────────────────────────
	def find_column(candidates, pattern=None):
		for c in candidates:
			if c in df.columns:
				return c
		if pattern:
			for c in df.columns:
				if re.search(pattern, c, re.I):
					return c
		return None

	# Column candidates
	title_col = find_column(
		['title', 'titulo', 'video_title', 'name', 'titulo_video'], r'titl|titulo')
	publish_col = find_column([
		'publish_time', 'publish_date', 'publishTimestamp', 'publishedAt',
		'upload_time', 'uploaded_at', 'fecha_publicacion', 'fecha_publicación',
		'fecha', 'published_at'], r'publish|upload|date|time|fecha|publica')
	duration_col = find_column([
		'duration', 'duracion', 'duracion_iso', 'duracion_segundos',
		'duracion_legible', 'video_duration', 'length', 'duration_sec',
		'length_seconds', 'duration_seconds', 'duration_ms'], r'duraci|duration|length')
	tags_col = find_column(
		['tags', 'etiquetas', 'tag_list', 'tag', 'keywords'], r'tag|keyword|etiquet')

	# ── Parsear publish time si hace falta ───────────────────────────────
	if publish_col:
		try:
			is_dt = pd.api.types.is_datetime64_any_dtype(df[publish_col].dtype)
		except Exception:
			is_dt = False
		if not is_dt:
			try:
				df[publish_col] = pd.to_datetime(df[publish_col], errors='coerce')
			except Exception:
				df[publish_col] = pd.to_datetime(
					df[publish_col].astype(str), errors='coerce')

	# ── Mapeo de días en español ─────────────────────────────────────────
	dias_es = {
		0: 'Lunes', 1: 'Martes', 2: 'Miércoles',
		3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo',
	}

	# ── Contadores / parsers ─────────────────────────────────────────────
	def count_title_words(x):
		if pd.isna(x):
			return 0
		return len(re.findall(r"\w+", str(x), flags=re.UNICODE))

	def to_seconds(x):
		if pd.isna(x):
			return None
		if isinstance(x, (int, float)) and not isinstance(x, bool):
			val = float(x)
			if val > 1e6:          # probablemente milisegundos
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
			parts = [p for p in s.split(':') if p.strip() != '']
			try:
				if len(parts) == 3:
					h, mm, ss = parts
					return int(h) * 3600 + int(mm) * 60 + int(float(ss))
				if len(parts) == 2:
					mm, ss = parts
					return int(mm) * 60 + int(float(ss))
			except Exception:
				pass
		m_hms = re.search(r'(?:(\d+)\s*h)', s, re.I)
		m_min = re.search(r'(?:(\d+)\s*m)', s, re.I)
		m_sec = re.search(r'(?:(\d+)\s*s)', s, re.I)
		if m_hms or m_min or m_sec:
			h = int(m_hms.group(1)) if m_hms else 0
			mm = int(m_min.group(1)) if m_min else 0
			ss = int(m_sec.group(1)) if m_sec else 0
			return h * 3600 + mm * 60 + ss
		m_digits = re.search(r'^(\d+(?:\.\d+)?)$', s)
		if m_digits:
			try:
				return int(float(m_digits.group(1)))
			except Exception:
				return None
		return None

	def count_tags(x):
		if pd.isna(x):
			return 0
		if isinstance(x, (list, tuple, set)):
			return len(x)
		s = str(x).strip()
		try:
			import ast
			v = ast.literal_eval(s)
			if isinstance(v, (list, tuple, set)):
				return len(v)
		except Exception:
			pass
		parts = re.split(r'[\|,;]+', s)
		parts = [p.strip() for p in parts if p.strip() != '']
		return len(parts)

	# ── Prefiltro por duration_filter_mode ───────────────────────────────
	if int(duration_filter_mode) in (1, 2, 4) and duration_col and duration_col in df.columns:
		dur_secs = df[duration_col].apply(to_seconds).fillna(-1).astype(int)
		if int(duration_filter_mode) == 1:
			# Solo videos con duración <= 60 s (≤ 1 min)
			df = df[dur_secs <= 60].copy()
		elif int(duration_filter_mode) == 2:
			# Solo videos con duración (60, 960] s  — (1 min, 16 min]
			df = df[(dur_secs > 60) & (dur_secs <= 960)].copy()
		elif int(duration_filter_mode) == 4:
			# Solo videos con duración < 180 s  (< 3 min)
			df = df[dur_secs < 180].copy()

	# ── Construir máscara (True = cumple todos los filtros) ──────────────
	mask = pd.Series(True, index=df.index)

	# Filtro título
	if title_word_intervals:
		counts = (df[title_col].apply(count_title_words)
				  if title_col and title_col in df.columns
				  else pd.Series(0, index=df.index))
		m = pd.Series(False, index=df.index)
		for a, b in title_word_intervals:
			m |= counts.between(int(a), int(b), inclusive='both')
		mask &= m

	# Filtro hora
	if hour_intervals:
		if not publish_col or publish_col not in df.columns:
			mask &= pd.Series(False, index=df.index)
		else:
			hours = df[publish_col].dt.hour.fillna(-1).astype(int)
			m = pd.Series(False, index=df.index)
			for a, b in hour_intervals:
				a_i, b_i = int(a), int(b)
				if a_i <= b_i:
					m |= hours.between(a_i, b_i, inclusive='both')
				else:
					m |= (hours >= a_i) | (hours <= b_i)
			mask &= m

	# Filtro días
	if days_list:
		if not publish_col or publish_col not in df.columns:
			mask &= pd.Series(False, index=df.index)
		else:
			weekday_nums = df[publish_col].dt.weekday

			def _weekday_to_name(x):
				if pd.isna(x):
					return ''
				try:
					return dias_es.get(int(x), '')
				except Exception:
					return ''

			nombres = weekday_nums.map(_weekday_to_name)
			mask &= nombres.isin(days_list)

	# Filtro duración (intervalos explícitos adicionales)
	if duration_intervals:
		if not duration_col or duration_col not in df.columns:
			mask &= pd.Series(False, index=df.index)
		else:
			secs = df[duration_col].apply(to_seconds).fillna(-1).astype(int)
			m = pd.Series(False, index=df.index)
			for a, b in duration_intervals:
				m |= secs.between(int(a), int(b), inclusive='both')
			mask &= m

	# Filtro tags
	if tag_count_intervals:
		if not tags_col or tags_col not in df.columns:
			mask &= pd.Series(False, index=df.index)
		else:
			tcounts = df[tags_col].apply(count_tags)
			m = pd.Series(False, index=df.index)
			for a, b in tag_count_intervals:
				m |= tcounts.between(int(a), int(b), inclusive='both')
			mask &= m

	filtered = df[mask].copy()
	remainder = df[~mask].copy()

	return filtered, remainder

# Análisis de viralidad por canal
def analyze_virality_ratio(
	df: pd.DataFrame,
	min_videos: int = 50,
	k: float = 1.5,
	root=None,
	output_dir=None,
	file_suffix=None,
	save: bool = True,
	show: bool = False,
) -> pd.DataFrame | None:
	"""
	Calcula el ratio de viralidad por canal y el promedio ponderado global.

	Un video se considera "viral" si su viewCount > Q3 + k×IQR, donde Q3 es
	el tercer cuartil de vistas del canal, IQR = Q3 − Q1 y k es el factor de
	holgura (predeterminado 1.5).

	Para evitar falsos outliers (videos que superan el umbral pero forman
	parte de la cola natural de la distribución), se aplica una confirmación
	basada en saltos (gap-based filtering): se ordenan las vistas del canal,
	y solo se cuentan como verdaderamente virales aquellos videos que estén
	separados del resto por un salto ≥ IQR en la secuencia ordenada.  Si los
	valores por encima del umbral transicionan suavemente desde la masa de
	datos (sin un salto significativo), se consideran falsos outliers.

	Solo se incluyen canales con más de `min_videos` videos.

	Pasos:
	  1. Para cada canal elegible se calcula Q3 de viewCount y el umbral.
	  2. Se ordenan las vistas y se identifican candidatos (vistas > umbral).
	  3. Se recorren los saltos entre valores consecutivos desde el último
	     valor ≤ umbral; solo los videos tras el primer salto ≥ IQR son
	     verdaderamente virales (confirmación de gap).
	  4. Se obtiene el ratio de viralidad por canal = virales / total.
	  5. Se calcula el promedio ponderado global =
	     Σ(virales_i) / Σ(total_i) sobre todos los canales elegibles.

	Args:
		df:          DataFrame con columnas channelTitle (o similar) y viewCount.
		min_videos:  Mínimo de videos para que un canal sea incluido (default 50).
		k:           Factor de holgura sobre IQR para definir "viral" (default 1.5).
		root:        Directorio raíz del proyecto.
		output_dir:  Directorio de salida; por defecto outputs/Viralidad.
		file_suffix: Sufijo para los archivos de salida.
		save:        Si True, guarda PNG y TXT.
		show:        Si True, muestra el gráfico en pantalla.

	Returns:
		DataFrame con columnas: canal, total_videos, q3_views, umbral_viral,
		videos_virales, ratio_viralidad.  Última fila = TOTAL PONDERADO.
		None si no hay datos válidos.
	"""
	if root is None:
		root = Path(__file__).resolve().parents[1]
	outputs_root = root / "outputs"
	outputs_root.mkdir(parents=True, exist_ok=True)
	if output_dir is None:
		script_dir = outputs_root / "Viralidad"
	else:
		script_dir = Path(output_dir)
	script_dir.mkdir(parents=True, exist_ok=True)

	df = df.copy()

	# ── Detectar columna de canal ────────────────────────────────────────
	channel_candidates = [
		"channelTitle", "canal", "canal_id", "channel", "channel_name",
		"channel_title", "nombre_canal", "uploader", "creator",
	]
	channel_col = None
	for c in channel_candidates:
		if c in df.columns:
			channel_col = c
			break
	if channel_col is None:
		for c in df.columns:
			if re.search(r"canal|channel|uploader|creator", c, re.I):
				channel_col = c
				break
	if channel_col is None:
		print("No se encontró una columna de canal para el análisis de viralidad.")
		return None

	# ── Detectar columna de vistas ───────────────────────────────────────
	view_candidates = ["viewCount", "views", "vistas", "view_count"]
	view_col = None
	for c in view_candidates:
		if c in df.columns:
			view_col = c
			break
	if view_col is None:
		for c in df.columns:
			if re.search(r"view|vista", c, re.I):
				view_col = c
				break
	if view_col is None:
		print("No se encontró una columna de vistas para el análisis de viralidad.")
		return None

	# ── Limpieza ─────────────────────────────────────────────────────────
	df[channel_col] = df[channel_col].astype(str).str.strip()
	df[view_col] = pd.to_numeric(df[view_col], errors="coerce")
	df = df.dropna(subset=[channel_col, view_col])
	df = df[df[channel_col] != ""]

	if df.empty:
		print("No hay datos válidos para el análisis de viralidad.")
		return None

	# ── Filtrar canales con más de min_videos ────────────────────────────
	channel_counts = df[channel_col].value_counts()
	eligible_channels = channel_counts[channel_counts > min_videos].index
	if eligible_channels.empty:
		print(f"Ningún canal tiene más de {min_videos} videos.")
		return None

	df = df[df[channel_col].isin(eligible_channels)]

	# ── Cálculo por canal ────────────────────────────────────────────────
	rows = []
	for canal, grp in df.groupby(channel_col):
		views = grp[view_col]
		q1 = views.quantile(0.25)
		q3 = views.quantile(0.75)
		iqr = q3 - q1
		umbral = q3 + k * iqr
		total = len(grp)

		# ── Filtrar falsos outliers mediante detección de saltos ────────
		# Solo se cuentan como virales los videos que, además de superar el
		# umbral, están separados de la masa de datos por un salto ≥ IQR
		# en la secuencia ordenada de vistas.
		if iqr > 0:
			sorted_views = np.sort(views.values)
			candidates = sorted_views[sorted_views > umbral]
			if len(candidates) == 0:
				virales = 0
			else:
				non_candidates = sorted_views[sorted_views <= umbral]
				if len(non_candidates) > 0:
					sequence = np.concatenate([[non_candidates[-1]], candidates])
				else:
					sequence = candidates
				gaps = np.diff(sequence)
				virales = 0
				for idx, gap in enumerate(gaps):
					if gap >= iqr:
						virales = len(candidates) - idx
						break
		else:
			virales = 0

		ratio = virales / total if total > 0 else 0.0
		rows.append({
			"canal": canal,
			"total_videos": total,
			"q3_views": round(q3, 2),
			"umbral_viral": round(umbral, 2),
			"videos_virales": virales,
			"ratio_viralidad": round(ratio, 6),
		})

	result_df = pd.DataFrame(rows).sort_values("ratio_viralidad", ascending=False)

	# ── Promedio ponderado global ────────────────────────────────────────
	sum_virales = result_df["videos_virales"].sum()
	sum_total = result_df["total_videos"].sum()
	ratio_global = sum_virales / sum_total if sum_total > 0 else 0.0

	global_row = pd.DataFrame([{
		"canal": "── TOTAL PONDERADO ──",
		"total_videos": int(sum_total),
		"q3_views": np.nan,
		"umbral_viral": np.nan,
		"videos_virales": int(sum_virales),
		"ratio_viralidad": round(ratio_global, 6),
	}])
	result_df = pd.concat([result_df, global_row], ignore_index=True)

	# ── Consola ──────────────────────────────────────────────────────────
	print(f"\n{'='*70}")
	print(f"Análisis de Viralidad  (min_videos={min_videos}, k={k})")
	print(f"{'='*70}")
	print(f"Canales elegibles: {len(eligible_channels)}")
	print(f"Total de videos analizados: {sum_total:,}")
	print(f"Total de videos virales: {sum_virales:,}")
	print(f"Ratio de viralidad ponderado global: {ratio_global:.4%}")
	print(f"\nDetalle por canal (ordenado de mayor a menor ratio):")
	for _, r in result_df.iterrows():
		if r["canal"] == "── TOTAL PONDERADO ──":
			continue
		print(f"  {r['canal'][:35]:<36} "
			  f"total={int(r['total_videos']):>6,}  "
			  f"umbral={r['umbral_viral']:>12,.0f}  "
			  f"virales={int(r['videos_virales']):>5,}  "
			  f"ratio={r['ratio_viralidad']:.4%}")
	print(f"\n  {'── TOTAL PONDERADO ──':<36} "
		  f"total={sum_total:>6,}  "
		  f"{'':>12}  "
		  f"virales={sum_virales:>5,}  "
		  f"ratio={ratio_global:.4%}")

	# ── Preparar sufijo ──────────────────────────────────────────────────
	suffix = f"_{re.sub(r'\\W+', '_', str(file_suffix)).strip('_')}" if file_suffix else ""

	# ── Gráfico de barras horizontales ───────────────────────────────────
	plot_df = result_df[result_df["canal"] != "── TOTAL PONDERADO ──"].copy()
	plot_df = plot_df.sort_values("ratio_viralidad", ascending=True)

	fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.45)))
	bars = ax.barh(
		plot_df["canal"].str[:30],
		plot_df["ratio_viralidad"] * 100,
		color=plt.cm.viridis(np.linspace(0.25, 0.85, len(plot_df))),
		edgecolor="white",
		linewidth=0.5,
	)
	# Línea del ratio global
	ax.axvline(ratio_global * 100, color="red", linestyle="--", linewidth=1.2,
			   label=f"Promedio ponderado: {ratio_global:.2%}")

	for bar in bars:
		width = bar.get_width()
		ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2,
				f"{width:.2f}%", va="center", fontsize=9)

	ax.set_xlabel("Ratio de viralidad (%)", fontsize=12)
	ax.set_title(f"Ratio de Viralidad por Canal  (umbral = Q3 + {k}×IQR, mín {min_videos} videos)",
				 fontsize=14)
	ax.legend(fontsize=11)
	plt.tight_layout()

	if save:
		png_path = script_dir / f"virality_ratio{suffix}.png"
		plt.savefig(png_path, dpi=150)
		print(f"\nGráfico guardado en {png_path}")

		txt_path = script_dir / f"virality_ratio{suffix}.txt"
		with open(txt_path, "w", encoding="utf-8") as f:
			f.write(f"Análisis de Viralidad  (min_videos={min_videos}, k={k})\n")
			f.write(f"Canales elegibles: {len(eligible_channels)}\n")
			f.write(f"Ratio ponderado global: {ratio_global:.6f} ({ratio_global:.4%})\n\n")
			header = (f"{'Canal':<35} {'Total':>7} {'Q3+k·IQR':>12} "
					  f"{'Virales':>8} {'Ratio':>10}\n")
			f.write(header)
			f.write("-" * len(header) + "\n")
			for _, r in result_df.iterrows():
				umb_str = f"{r['umbral_viral']:>12,.0f}" if pd.notna(r["umbral_viral"]) else f"{'':>12}"
				f.write(f"{str(r['canal'])[:35]:<35} {int(r['total_videos']):>7,} "
						f"{umb_str} "
						f"{int(r['videos_virales']):>8,} "
						f"{r['ratio_viralidad']:>9.4%}\n")
		print(f"Datos guardados en {txt_path}")

	if show:
		plt.show()
	plt.close()

	return result_df

# Análisis de crecimiento de vistas por edad del video
def analyze_views_growth_by_age(
	df: pd.DataFrame,
	root=None,
	output_dir=None,
	file_suffix=None,
	min_videos_per_bin: int = 30,
	save: bool = True,
	show: bool = False,
) -> pd.DataFrame | None:
	"""
	Agrupa los videos por edad (semanas desde publicación) y calcula la mediana
	de vistas por grupo.  Luego calcula la tasa de crecimiento entre grupos
	consecutivos:  (V_t − V_(t−1)) / V_(t−1).

	Esquema de grupos:
	  1 sem, 2 sem, 3 sem, 1 mes, 1 mes y 1 sem, …, N meses y 3 sem.

	Genera un gráfico de barras con el % de crecimiento por intervalo.

	Args:
		df:                 DataFrame con columnas publishedAt y viewCount.
		root:               Directorio raíz del proyecto.
		output_dir:         Directorio de salida (default: outputs/Crecimiento).
		file_suffix:        Sufijo para archivos de salida.
		min_videos_per_bin: Mínimo de videos por grupo para incluirlo.
		save:               Si True, guarda PNG y TXT.
		show:               Si True, muestra el gráfico en pantalla.

	Returns:
		DataFrame con columnas: label, age_days, median_views, videos_count,
		growth_pct.  None si no hay datos válidos.
	"""
	if root is None:
		root = Path(__file__).resolve().parents[1]
	outputs_root = root / "outputs"
	outputs_root.mkdir(parents=True, exist_ok=True)
	if output_dir is None:
		script_dir = outputs_root / "Crecimiento"
	else:
		script_dir = Path(output_dir)
	script_dir.mkdir(parents=True, exist_ok=True)

	df = df.copy()

	# ── Detectar columnas ────────────────────────────────────────────────
	if "publishedAt" not in df.columns:
		print("No se encontró la columna 'publishedAt'.")
		return None
	view_col = None
	for c in ["viewCount", "views", "vistas", "view_count"]:
		if c in df.columns:
			view_col = c
			break
	if view_col is None:
		print("No se encontró una columna de vistas.")
		return None

	# ── Parsear fechas y calcular edad en días ───────────────────────────
	df["_pub"] = pd.to_datetime(df["publishedAt"], utc=True, errors="coerce")
	df = df.dropna(subset=["_pub", view_col])
	df[view_col] = pd.to_numeric(df[view_col], errors="coerce")
	df = df.dropna(subset=[view_col])

	reference_date = df["_pub"].max()
	df["_age_days"] = (reference_date - df["_pub"]).dt.days

	# ── Construir bins semanales ─────────────────────────────────────────
	max_age = int(df["_age_days"].max())
	max_weeks = max_age // 7 + 1

	def _week_label(i: int) -> str:
		"""Etiqueta legible para el bin semanal i (0-based)."""
		if i < 3:
			return f"{i + 1} sem"
		offset = i - 3
		month = offset // 4 + 1
		week = offset % 4
		m_str = f"{month} mes" if month == 1 else f"{month} meses"
		if week == 0:
			return m_str
		return f"{m_str} y {week} sem"

	# Asignar bin semanal (0-based): edad 0-6 → bin 0, 7-13 → bin 1, …
	df["_week_bin"] = df["_age_days"] // 7

	# ── Mediana de vistas por bin ────────────────────────────────────────
	grouped = (
		df.groupby("_week_bin")[view_col]
		.agg(median_views="median", videos_count="count")
		.reset_index()
		.rename(columns={"_week_bin": "week_bin"})
	)

	# Filtrar bins con pocos videos
	grouped = grouped[grouped["videos_count"] >= min_videos_per_bin].copy()
	grouped = grouped.sort_values("week_bin").reset_index(drop=True)

	if len(grouped) < 2:
		print("No hay suficientes grupos con datos para calcular crecimiento.")
		return None

	# Etiquetas y días
	grouped["label"] = grouped["week_bin"].apply(_week_label)
	grouped["age_days"] = grouped["week_bin"] * 7

	# ── Tasa de crecimiento ──────────────────────────────────────────────
	grouped["growth_pct"] = (
		grouped["median_views"].pct_change() * 100
	)  # (V_t - V_(t-1)) / V_(t-1) × 100

	print(f"\n{'─'*60}")
	print("Crecimiento de vistas por edad del video")
	print(f"{'─'*60}")
	print(f"{'Edad':<22} {'Mediana':>12} {'Videos':>8} {'Δ%':>10}")
	print(f"{'─'*22} {'─'*12} {'─'*8} {'─'*10}")
	for _, r in grouped.iterrows():
		g = f"{r['growth_pct']:>+9.2f}%" if pd.notna(r["growth_pct"]) else "       —"
		print(
			f"{r['label']:<22} {r['median_views']:>12,.0f} "
			f"{int(r['videos_count']):>8,} {g}"
		)
	print(f"{'─'*60}")

	# ── Gráfico ──────────────────────────────────────────────────────────
	plot_df = grouped.dropna(subset=["growth_pct"]).copy()

	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, len(grouped) * 0.5), 12),
									gridspec_kw={"height_ratios": [1, 1.3]})

	# ---- Subplot 1: Mediana de vistas por edad ----
	ax1.plot(
		grouped["label"], grouped["median_views"],
		marker="o", linewidth=2, color="#2196F3", markersize=6,
	)
	ax1.set_ylabel("Mediana de vistas", fontsize=12)
	ax1.set_title("Mediana de vistas según edad del video", fontsize=14, weight="bold")
	ax1.tick_params(axis="x", rotation=90, labelsize=8)
	ax1.grid(axis="y", alpha=0.3)

	# ---- Subplot 2: % de crecimiento semanal ----
	colors = ["#4CAF50" if v >= 0 else "#F44336" for v in plot_df["growth_pct"]]
	bars = ax2.bar(plot_df["label"], plot_df["growth_pct"], color=colors, edgecolor="white", linewidth=0.5)

	# Anotar valores sobre las barras
	for bar, val in zip(bars, plot_df["growth_pct"]):
		y = bar.get_height()
		ax2.text(
			bar.get_x() + bar.get_width() / 2,
			y + (0.5 if y >= 0 else -1.5),
			f"{val:+.1f}%",
			ha="center", va="bottom" if y >= 0 else "top",
			fontsize=7, weight="bold",
		)

	ax2.axhline(0, color="black", linewidth=0.8)
	ax2.set_ylabel("Crecimiento %  (V_t − V_{t−1}) / V_{t−1}", fontsize=12)
	ax2.set_xlabel("Edad del video", fontsize=12)
	ax2.set_title(
		"% de crecimiento de vistas entre periodos consecutivos",
		fontsize=14, weight="bold",
	)
	ax2.tick_params(axis="x", rotation=90, labelsize=8)
	ax2.grid(axis="y", alpha=0.3)

	plt.tight_layout()

	# ── Guardar ──────────────────────────────────────────────────────────
	suffix = f"_{re.sub(r'\\W+', '_', str(file_suffix)).strip('_')}" if file_suffix else ""

	if save:
		png_path = script_dir / f"views_growth_by_age{suffix}.png"
		plt.savefig(png_path, dpi=150, bbox_inches="tight")
		print(f"Gráfico guardado en {png_path}")

		txt_path = script_dir / f"views_growth_by_age{suffix}.txt"
		with open(txt_path, "w", encoding="utf-8") as f:
			f.write("label,age_days,median_views,videos_count,growth_pct\n")
			for _, r in grouped.iterrows():
				g = f"{r['growth_pct']:.4f}" if pd.notna(r["growth_pct"]) else ""
				f.write(
					f"{r['label']},{int(r['age_days'])},"
					f"{r['median_views']:.2f},{int(r['videos_count'])},{g}\n"
				)
		print(f"Datos guardados en {txt_path}")

	if show:
		plt.show()
	plt.close()

	return grouped[["label", "age_days", "median_views", "videos_count", "growth_pct"]]

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

	# Analizar porcentaje de videos por canal y guardar en outputs/EDA
	analyze_channel_percentage(df, output_dir=eda_dir)

	# Generar gráfico de distribución por duración y guardarlo también en outputs/EDA
	plot_duration_pie_buckets(df, output_dir=eda_dir)
	# Generar gráfico de distribución por calidad/definición y guardarlo en outputs/EDA
	plot_definicion_pie(df=df, output_dir=eda_dir)

	# Generar gráfico de videos publicados por mes y guardarlo en outputs/EDA
	plot_videos_por_mes(df=df, output_dir=eda_dir)

	# Generar gráfico de videos publicados por semana desde enero 2026 hasta la actualidad
	plot_videos_por_semana(df=df, start_week='2025-12-29', output_dir=eda_dir)

	# Generar gráfico de distribución por categorías y guardarlo en outputs/EDA
	plot_category_pie(df=df, output_dir=eda_dir)

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
	analyze_weekday_distribution(df, output_dir=weekdays_dir, file_suffix='3_16min', duracion_filter=(60*3, 16*60))

	#--------------------------------------------------
	# ANÁLISIS DEL DÍA 
	#--------------------------------------------------
	analisis_dia_dir = os.path.join(root, 'outputs', 'Análisis del día')
	os.makedirs(analisis_dia_dir, exist_ok=True)
 
	# Conteo de publicaciones por intervalos de 2 horas, con análisis separado por día de la semana
	analyze_publications_2h_intervals(df, output_dir=analisis_dia_dir, por_dia=True)

	#--------------------------------------------------
	# DISTRIBUCIÓN DE DURACIÓN EN INTERVALOS DE MINUTOS Y SEGUNDOS
	#--------------------------------------------------
	dist_dir = os.path.join(root, 'outputs', 'Distribucion min')
	os.makedirs(dist_dir, exist_ok=True)
 
	# Generar histogramas de distribución de duración en intervalos de minutos (1-2,...,15-16)
	# y segundos personalizados para <=1min
	second_bins_custom_1 = [(0,10), (11,20), (20,30), (30,45), (45,60)]
	second_bins_custom_2 = [(0,15), (15,30), (30,45), (45,60)]
	second_bins_custom_3 = [(0,5), (5,10), (10,15), (15,20), (20,30), (30,40), (40,50), (50,60)]
 
	analyze_duration_interval_distribution(df, output_dir=dist_dir, second_bins=second_bins_custom_1, file_suffix='sbins_v1')
	analyze_duration_interval_distribution(df, output_dir=dist_dir, second_bins=second_bins_custom_2, file_suffix='sbins_v2')
	analyze_duration_interval_distribution(df, output_dir=dist_dir, second_bins=second_bins_custom_3, file_suffix='sbins_v3')

	#--------------------------------------------------
	# MONTE CARLO – INTERVALOS DE DURACIÓN
	#--------------------------------------------------
	random_dir = os.path.join(root, 'outputs', 'Random')
	os.makedirs(random_dir, exist_ok=True)

	monte_carlo_duration_intervals(df, output_dir=random_dir)
	monte_carlo_weekdays(df, output_dir=random_dir)
	monte_carlo_title_words(df, output_dir=random_dir)
	monte_carlo_tag_count(df, output_dir=random_dir)
	monte_carlo_description_length(df, output_dir=random_dir)
	monte_carlo_hourly_by_weekday(df, output_dir=random_dir)
	monte_carlo_hourly_2h(df, output_dir=random_dir)

	#--------------------------------------------------
	# FILTRADO Y MONTE CARLO DOS DATASETS
	#--------------------------------------------------
	
	resultados_dir = os.path.join(root, 'outputs', 'Resultados')
	os.makedirs(resultados_dir, exist_ok=True)
 	
	df_filtrados, df_restantes = filter_videos_by_properties(
		df,
		title_word_intervals=[(4, 6), (16, 18), (19, 21), (7, 9)],
		hour_intervals=[(10, 12), (12, 14), (14, 16), (20, 22), (16, 18), (22, 24), (18, 20)],
		days_list=[ "Viernes", "Miércoles", "Jueves"],
		duration_intervals=[(45, 50), (50, 60), (20, 40)],
		tag_count_intervals=[(21, 30), (41, 60)],
		duration_filter_mode=4,
	)

	monte_carlo_two_datasets(
		df_a=df_filtrados,
		df_b=df_restantes,
		label_a="filtrados",
		label_b="restantes",
		output_dir=resultados_dir,
		file_suffix="filtro_propiedades",
		n_rounds=200_000,
		seed=42,
		save=True,
		show=False,
	)

	#--------------------------------------------------
	# VIRALIDAD
	#--------------------------------------------------
	viralidad_dir = os.path.join(root, 'outputs', 'Viralidad')
	os.makedirs(viralidad_dir, exist_ok=True)

	analyze_virality_ratio(df, min_videos=50, k=5, output_dir=viralidad_dir)

	#--------------------------------------------------
	# CRECIMIENTO DE VISTAS POR EDAD
	#--------------------------------------------------
	crecimiento_dir = os.path.join(root, 'outputs', 'Crecimiento')
	os.makedirs(crecimiento_dir, exist_ok=True)

	analyze_views_growth_by_age(df, output_dir=crecimiento_dir)

if __name__ == '__main__':
	main()

