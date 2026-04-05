# YouTube Entretenimiento — Análisis técnico y resultados 📊🎬🧠

Descripción técnica breve
-------------------------
Proyecto de análisis cuantitativo y simulación sobre un corpus de canales y vídeos de YouTube centrado en entretenimiento. El objetivo es caracterizar propiedades temporales, de duración, tags, títulos y métricas de viralidad mediante estadística descriptiva, análisis temporal y simulaciones Monte Carlo. El repositorio contiene el conjunto de datos crudo, scripts ETL/análisis y una colección abundante de resultados en texto para reproducibilidad y auditoría. 🧾🔬⚙️

Objetivos principales 🎯
- **Characterizar** distribución de duraciones y horarios de publicación.
- **Medir** señales de viralidad (ratio, crecimiento por edad del vídeo).
- **Simular** comportamientos aleatorios vs observados (Monte Carlo) para validar hallazgos.
- **Proveer** artefactos reproducibles (archivos de salida, scripts, datasets).

Estructura y flujo de datos (arquitectura) 🔁
1. Ingesta: `dataset/videos.csv` y los CSV por canal en `dataset/channels/` contienen la metadata de los vídeos (identificador, título, descripción, duración, fecha de publicación, vistas y campos asociados). 💾
2. ETL / Descarga: `scripts/download_channels.py` gestiona la adquisición / refresco de datos de canales. 🌐
3. Orquestación: `scripts/index.py` / `scripts/program.py` sirven como puntos de entrada para ejecutar pipelines completos (descarga → limpieza → análisis). 🛠️
4. Análisis: `scripts/analysis.py` y `scripts/channel_analysis.py` realizan agregaciones, cálculos estadísticos y generan ficheros de resultados en `outputs/`. 📈
5. Salidas: `outputs/` contiene carpetas temáticas (EDA, Distribución, Viralidad, Monte Carlo, etc.) con .txt que resumen métricas y distribuciones. 📁

Contenido destacable del repositorio 📂✨
- `dataset/` — datos fuentes y backups (`videos.csv`, `state.json`, `channels/*.csv`).
- `scripts/` — scripts de ingestión y análisis: [scripts/analysis.py](scripts/analysis.py), [scripts/channel_analysis.py](scripts/channel_analysis.py), [scripts/download_channels.py](scripts/download_channels.py), [scripts/index.py](scripts/index.py), [scripts/program.py](scripts/program.py).
- `outputs/` — resultados derivados organizados por tema (EDA, Random/MonteCarlo, Viralidad, Distribución min, Weekdays, Crecimiento, etc.). Ejemplos: [outputs/Viralidad/virality_ratio.txt](outputs/Viralidad/virality_ratio.txt), [outputs/Crecimiento/views_growth_by_age.txt](outputs/Crecimiento/views_growth_by_age.txt).
- `API YOUTUBE (LofiAPI).txt` — notas / configuración relacionada con la API usada (si aplica). 🧾

Descripción técnica de los análisis principales 🧠🔎
- Distribuciones de duración: se calculan histogramas y bucketizaciones en segundos/minutos para comparar subpoblaciones (p.ej. `menos_1min/` vs `3_16min/`). Resultados almacenados en `outputs/Distribucion min/` y `outputs/Duration/`. ⏱️
- Horarios y días de la semana: agregaciones por hora/ventana de 2h y por weekday para identificar ventanas de publicación con mayor densidad y correlación con vistas. Resultados en `outputs/Weekdays/` y `outputs/Análisis del día/`. 🕒📅
- Monte Carlo: para cada métrica (longitud de título, número de tags, descripción, distribución horaria) se ejecutan simulaciones aleatorias que generan distribuciones nulas; los archivos `monte_carlo_*.txt` contienen percentiles y p-values aproximados para evaluar si los observados se desvían de la aleatoriedad. 🧪🎲
- Viralidad y crecimiento: `virality_ratio.txt` y `views_growth_by_age.txt` resumen métricas que combinan vistas, edad del vídeo y crecimiento temporal para identificar outliers de rápido crecimiento y patrones replicables. 📈🔥

Formato y convenciones de los artefactos
- Ficheros de salida son texto plano con columnas o series resumidas; nombre de archivo sigue la convención `{tema}/{indicador}.txt`.
- Backups de datasets mantienen timestamps en el sufijo (p.ej. `videos.csv.bak_20260313_014547`). 🕊️

Interpretación práctica de resultados (cómo leerlos) 🧭
- Archivos `*_histogram_*.txt` contienen bins y conteos; comparar percentiles para entender dispersión.
- `monte_carlo_*` presentan intervalo de confianza empírico: si la métrica observada cae fuera del intervalo 95% de la simulación, es estadísticamente notable. ✨
- `virality_ratio.txt` proporciona un indicador normalizado (por edad/vistas esperadas); valores altos marcan candidatos para estudio manual. 🔍

Cómo reproducir (entorno mínimo) 🧰
Recomendado: Python 3.10+ en entorno virtual. Ejemplo rápido:

```bash
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt  # si existe, o instalar pandas, numpy, matplotlib
python scripts/index.py       # ejecutar pipeline principal (según configuración)
```

Buenas prácticas y notas para desarrollo 🛡️
- Versionar `dataset/videos.csv` solo para snapshots; preferir backups con timestamp.
- Mantener `state.json` como registro de estado del pipeline (última ejecución, offsets, fallos recuperables).
- Añadir `requirements.txt` si se incorporan dependencias explícitas. 📦

Limitaciones conocidas ⚠️
- Calidad y completitud de la metadata dependen de la fuente (API/CSV manual). Algunos campos pueden faltar o estar inconsistentes.
- Las simulaciones Monte Carlo asumen independencias específicas; validar hipótesis antes de generalizar conclusiones. 🧾

Siguientes pasos sugeridos 🛠️➡️
- Automatizar CI para regenerar `outputs/` con cada push (reproducibilidad).
- Serializar resultados clave a formatos estructurados (JSON/Parquet) para análisis posteriores.
- Implementar notebooks de exploración reproducible (Jupyter) con visualizaciones interactivas. 📊

Contacto y mantenimiento
- Autor / Mantainer: revisar metadatos del repositorio o contactar al responsable del proyecto.

¡Listo! Este README sintetiza la arquitectura, artefactos y resultados del proyecto. 🚀🔧📚
