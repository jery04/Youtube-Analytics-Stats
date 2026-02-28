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

#  Genera un gráfico de pastel con los porcentajes por duración.
def plot_duration_pie(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
	"""Genera un gráfico de pastel con los porcentajes por duración.

	Categorías:
	- menos de 1min (< 60s)
	- entre 1-15min (60s - 900s)
	- otros (> 900s)
	"""
	if 'durationSeconds' in df.columns:
		secs = pd.to_numeric(df['durationSeconds'], errors='coerce').fillna(0)
	else:
		# Fallback: intentar parsear columna `duration` (ISO 8601) no implementado aquí
		raise ValueError('La columna durationSeconds no existe en el dataset')

	less_1 = int((secs < 60).sum())
	between_1_15 = int(((secs >= 60) & (secs <= 15 * 60)).sum())
	others = int((secs > 15 * 60).sum())

	sizes = [less_1, between_1_15, others]
	labels = ['<1 min', '1–15 min', 'Otros']

	fig, ax = plt.subplots(figsize=(6, 6))
	# Calcular porcentajes
	total = sum(sizes) if sum(sizes) > 0 else 1
	percents = [s / total * 100 for s in sizes]

	# Función autopct que muestra porcentaje y conteo dentro de la porción
	def make_autopct(values):
		def my_autopct(pct):
			absolute = int(round(pct * sum(values) / 100.0))
			return f"{pct:.1f}%\n({absolute})"
		return my_autopct

	wedges, texts, autotexts = ax.pie(
		sizes,
		labels=labels,
		autopct=make_autopct(sizes),
		startangle=90,
		pctdistance=0.6,
		labeldistance=1.05,
	)

	# Mejorar legibilidad del texto dentro de las porciones
	for t in autotexts:
		t.set_color('white')
		t.set_fontsize(9)
		t.set_fontweight('bold')
	ax.set_title('Distribución de duración de videos')
	ax.axis('equal')

	if save_path:
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		plt.savefig(save_path, bbox_inches='tight')
		# Guardar archivo TXT con conteos y porcentajes
		try:
			from pathlib import Path
			out_dir = Path(save_path).parent
			txt_file = out_dir / (Path(save_path).stem + '.txt')
			with open(txt_file, 'w', encoding='utf-8') as f:
				f.write('Category,Count,Percent\n')
				for lab, s, p in zip(labels, sizes, percents):
					f.write(f"{lab},{int(s)},{p:.2f}\n")
			print(f"Archivo de duración guardado en {txt_file}")
		except Exception:
			pass
		plt.close(fig)
	else:
		plt.show()

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
	script_dir = outputs_root / Path(__file__).stem
	script_dir.mkdir(parents=True, exist_ok=True)
	if output_dir is not None:
		script_dir = Path(output_dir)
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



# Función principal para ejecutar el análisis
def main() -> None:
	root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
	csv_path = os.path.join(root, 'dataset', 'videos.csv')

	if not os.path.exists(csv_path):
		raise FileNotFoundError(f'No se encontró el dataset en: {csv_path}')

	print(f'Cargando dataset desde: {csv_path}')
	df = load_csv(csv_path)

	out_dir = os.path.join(root, 'outputs')
	os.makedirs(out_dir, exist_ok=True)
	save_path = os.path.join(out_dir, 'duration_pie.png')

    # Generar gráfico de pastel y guardarlo
	plot_duration_pie(df, save_path=save_path)

	# Analizar porcentaje de filas por año
	analyze_yearly_percentage(df, output_dir=out_dir)



if __name__ == '__main__':
	main()

