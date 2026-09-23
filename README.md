
# Reproducibilidad del Jupyter Book

El notebook maestro es `model_evaluation.ipynb`. Conserva el experimento ejecutado con Python 3.10.21 y el entorno `vol_venv`. La publicación utiliza las salidas existentes; no requiere reentrenar los 100 MLP.

## Entorno original y ejecución completa documentada

Para una reproducción integral deliberada en un entorno compatible con las versiones fijadas:

```powershell
py -3.10 -m venv vol_venv
.\vol_venv\Scripts\python.exe -m pip install -r requirements.txt
.\vol_venv\Scripts\python.exe -m jupyter lab model_evaluation.ipynb
```

La ejecución completa, desde Jupyter y en el orden de las celdas del maestro, reproduce el experimento y entrena sus modelos. Se documenta como procedimiento; no se ejecutó para completar o publicar este libro. `requirements.txt` se conserva sin cambios y su instalación completa no se verificó en esta actualización.

## Datos y artefactos

- `data/processed/crypto_binance_master_1h.csv`: caché local. Con `FORCE_DOWNLOAD = False` y el archivo existente, el maestro carga el CSV; una caché ausente o una descarga forzada activa la adquisición. Las fechas se interpretan en UTC.
- `results/`: métricas, diagnóstico temporal y predicciones guardadas; `results/volatility_predictions/` contiene los NPZ.
- `notebooks/figs/`: figuras originales del EDA y de evaluación, incluidas las curvas real vs predicho.
- `eda.ipynb`, `model_base.ipynb` y `conclusions.ipynb`: capítulos con las salidas existentes y los incisos complementarios de la guía.
- `scripts/eda_supplement.py`: únicamente EDA ligero de lectura de la caché y cálculo en memoria; imprime JSON, sin escribir CSV ni figuras ni entrenar modelos. Sus versiones de ejecución están registradas en EDA 2.6; son distintas del entorno histórico del maestro.

## Publicar desde las salidas guardadas

Con Node.js y npm disponibles, desde la raíz del proyecto:

```shell
npx mystmd build
```

En PowerShell, si la política impide ejecutar `npx.ps1`, el comando equivalente es `npx.cmd mystmd build`. No se añade `--execute`; las celdas existentes conservan `skip-execution`. El TOC está en `myst.yml`. El build valida y genera los documentos MyST en `_build/` usando las salidas guardadas.

## EDA complementario sin entrenamientos

En un intérprete con NumPy, pandas, SciPy y statsmodels disponibles:

```shell
python scripts/eda_supplement.py
```

Este comando no ejecuta ninguna celda del maestro. Los perfiles usan todas las observaciones útiles; ADF/KPSS y PACF usan el segmento continuo mayor por activo para evitar cruzar huecos. Los resultados se publican como tablas Markdown nuevas en el EDA.
