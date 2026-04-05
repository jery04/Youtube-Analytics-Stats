
# YouTube Entertainment — Technical analysis and results 📊🎬🧠

Brief technical description
---------------------------
This repository contains a quantitative analysis and simulation project focused on a corpus of YouTube channels and videos in the entertainment genre. The goal is to characterize temporal properties, durations, tags, titles, and virality metrics using descriptive statistics, time-based analysis, and Monte Carlo simulations. The repository includes the raw dataset, ETL/analysis scripts, and an extensive collection of text-based results for reproducibility and auditing. 🧾🔬⚙️

Main objectives 🎯
- **Characterize** the distribution of video durations and publishing times.
- **Measure** signals of virality (virality ratio, growth by video age).
- **Simulate** random vs observed behavior (Monte Carlo) to validate findings.
- **Provide** reproducible artifacts (output files, scripts, datasets).

Data structure and workflow (architecture) 🔁
1. Ingestion: `dataset/videos.csv` and per-channel CSVs in `dataset/channels/` contain video metadata (id, title, description, duration, publish date, views, and related fields). 💾
2. ETL / Download: `scripts/download_channels.py` handles acquisition and refresh of channel data. 🌐
3. Orchestration: `scripts/index.py` and `scripts/program.py` act as entry points to run full pipelines (download → clean → analyze). 🛠️
4. Analysis: `scripts/analysis.py` and `scripts/channel_analysis.py` perform aggregations, statistical calculations, and generate result files in `outputs/`. 📈
5. Outputs: `outputs/` contains thematic folders (EDA, Distribution, Virality, Monte Carlo, etc.) with .txt files summarizing metrics and distributions. 📁

Notable repository contents 📂✨
- `dataset/` — source data and backups (`videos.csv`, `state.json`, `channels/*.csv`).
- `scripts/` — ingestion and analysis scripts: [scripts/analysis.py](scripts/analysis.py), [scripts/channel_analysis.py](scripts/channel_analysis.py), [scripts/download_channels.py](scripts/download_channels.py), [scripts/index.py](scripts/index.py), [scripts/program.py](scripts/program.py).
- `outputs/` — derived results organized by topic (EDA, Random/MonteCarlo, Virality, Duration Distribution, Weekdays, Growth, etc.). Examples: [outputs/Viralidad/virality_ratio.txt](outputs/Viralidad/virality_ratio.txt), [outputs/Crecimiento/views_growth_by_age.txt](outputs/Crecimiento/views_growth_by_age.txt).
- `API YOUTUBE (LofiAPI).txt` — notes / configuration related to the API used (if applicable). 🧾

Technical description of the main analyses 🧠🔎
- Duration distributions: histograms and bucketizations in seconds/minutes are used to compare subpopulations (e.g. `menos_1min/` vs `3_16min/`). Results are stored in `outputs/Distribucion min/` and `outputs/Duration/`. ⏱️
- Publishing times and weekdays: aggregations by hour / 2-hour windows and by weekday identify publishing windows with higher density and potential correlations with views. Results are in `outputs/Weekdays/` and `outputs/Análisis del día/`. 🕒📅
- Monte Carlo: for each metric (title length, tag count, description length, hourly distribution) random simulations generate null distributions; `monte_carlo_*.txt` files contain percentiles and approximate p-values to assess whether observed values deviate from randomness. 🧪🎲
- Virality and growth: `virality_ratio.txt` and `views_growth_by_age.txt` summarize metrics that combine views, video age and temporal growth to identify fast-growing outliers and replicable patterns. 📈🔥

Format and conventions of artifacts
- Output files are plain text with columns or summarized series; filenames follow the convention `{topic}/{indicator}.txt`.
- Dataset backups include timestamps in the suffix (e.g. `videos.csv.bak_20260313_014547`). 🕊️

Practical interpretation of results (how to read them) 🧭
- `*_histogram_*.txt` files contain bins and counts; compare percentiles to understand dispersion.
- `monte_carlo_*` files present empirical confidence intervals: if the observed metric falls outside the 95% simulation interval, it is statistically notable. ✨
- `virality_ratio.txt` provides a normalized indicator (by age / expected views); high values flag candidates for manual inspection. 🔍

How to reproduce (minimum environment) 🧰
Recommended: Python 3.10+ in a virtual environment. Quick example:

```bash
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt  # if present, or install pandas, numpy, matplotlib
python scripts/index.py       # run the main pipeline (depending on configuration)
```
