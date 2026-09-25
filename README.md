
# Reproducibilidad del Jupyter Book

El notebook maestro es `model_evaluation.ipynb`: contiene adquisición, preparación, EDA, entrenamiento de 100 MLP multisalida, evaluación, R², persistencia, figuras y conclusiones calculadas. `model_evaluation_compilado.ipynb` es la copia de entrega con las mismas salidas ejecutadas. La publicación del Jupyter Book utiliza las salidas de sus capítulos y no requiere entrenar modelos.

## Reproducir el notebook maestro

Desde la raíz del repositorio:

1. Crear un entorno con Python 3.10 (versión de referencia: 3.10.21).
2. Instalar las versiones fijadas en `requirements.txt`.
3. Registrar el kernel y abrir `model_evaluation.ipynb`.
4. Seleccionar **Python (vol_venv)** en Jupyter.
5. Reiniciar el kernel y pulsar **Run All**; esperar a que terminen las 100 redes y las figuras.
6. La primera ejecución descarga automáticamente los cinco activos de Binance Spot, entre 2020-01-01 y 2026-09-20 inclusive, en UTC.
7. Las siguientes ejecuciones reutilizan la caché local cuando `FORCE_DOWNLOAD = False`.

```powershell
py -3.10 -m venv vol_venv
.\vol_venv\Scripts\python.exe -m pip install -r requirements.txt
.\vol_venv\Scripts\python.exe -m ipykernel install --user --name vol_venv --display-name "Python (vol_venv)"
.\vol_venv\Scripts\python.exe -m jupyter lab model_evaluation.ipynb
```

También puede utilizarse un entorno Conda con Python 3.10.21 y ejecutar los mismos comandos de `pip`, `ipykernel` y `jupyter` desde él. El notebook imprime las versiones al comenzar y advierte si Python no es 3.10.x. `requirements.txt` conserva las versiones del entorno original; reproducir también el entorno numérico ayuda a conservar los resultados.

Las celdas se ejecutan en orden: dependen de las anteriores del propio notebook, pero no de variables de otra sesión, otros notebooks ni scripts auxiliares. No se necesita copiar datos ni resultados manualmente. Cada Run All **entrena los 100 modelos** y sobrescribe los artefactos generados en `results/` y `notebooks/figs/`. La descarga inicial requiere conexión a Binance; una caché existente permite ejecutar el análisis sin consultar la API. La preparación conserva `ddof=1`, segmentación temporal, cinco folds expansivos con purga y escaladores ajustados solo con train.

La ventana se selecciona por RMSE TEST y después se valida contra BTC=28, ETH=28, BNB=7, XRP=28 y SOL=7. Esa selección es retrospectiva y no sustituye una evaluación externa. R² se calcula directamente con los valores reales y las predicciones TEST; un valor alto no demuestra ausencia de leakage. El target rolling de 30 horas posee persistencia y solapamiento temporal, por lo que se compara también con persistencia.

## Datos y artefactos

- `data/processed/crypto_binance_master_1h.csv`: caché local. Con `FORCE_DOWNLOAD = False` y el archivo existente, el maestro carga el CSV; una caché ausente o una descarga forzada activa la adquisición. Las fechas se interpretan en UTC.
- `results/`: métricas, diagnóstico temporal, R², persistencia, EDA complementario y `master_run_report.json`. Los NPZ de `results/volatility_predictions/` generados por el maestro contienen `pred_train`, `pred_val`, `pred_test`, `y_train`, `y_val`, `y_test` e índices; `validation_indices` se conserva junto al alias `val_indices`.
- `notebooks/figs/`: figuras originales del EDA y de evaluación, incluidas las curvas real vs predicho.
- `eda.ipynb`, `model_base.ipynb` y `conclusions.ipynb`: capítulos con las salidas existentes y los incisos complementarios de la guía.
- `scripts/eda_supplement.py`: únicamente EDA ligero de lectura de la caché y cálculo en memoria; imprime JSON, sin escribir CSV ni figuras ni entrenar modelos. Sus versiones de ejecución están registradas en EDA 2.6; son distintas del entorno histórico del maestro.

Los scripts auxiliares de análisis histórico no son necesarios para Run All: sus cálculos relevantes están integrados en el maestro. `data/raw/` y `data/processed/` permanecen excluidos de Git.

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
