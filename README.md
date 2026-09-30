# Entregable 1: pronóstico de volatilidad en criptomonedas

Informe de investigación para BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT y SOLUSDT con datos horarios de Binance Spot. Compara Persistence y SVR lineal mediante validación temporal. TEST permanece reservado.

## Lectura

- Informe: `book/entregable1_master.md` y `book/sections/`.
- Conclusiones: `book/sections/05_conclusiones.md`.
- Notebooks con resultados: `notebooks/`; modelo base: `18_base_model.ipynb`.
- **Notebook completo para entregar:** [Entregable_1_Completo.ipynb](notebooks/Entregable_1_Completo.ipynb). Reúne el informe, figuras como salidas Jupyter, código y archivos de entrada incorporados. Puede ejecutarse desde Descargas sin la carpeta del repositorio: recupera las entradas en un directorio temporal y recalcula los análisis y los 170 ajustes usando solo DEVELOPMENT. Requiere Python con las dependencias y Node.js.
- Sitio público: https://jassan-bit.github.io/MachineLearning_202630/ (consultar el registro de publicación para identificar la versión verificada).

El SVR obtuvo RMSE medio 0,623259 frente a 0,389006 de Persistence, en puntos porcentuales. Son resultados de validación interna, no de TEST. La entrega conserva las limitaciones metodológicas detalladas en el informe.

## Reproducción

Se verificó con Python 3.10.21 en Windows. Desde la raíz:

```powershell
python -m venv .venv-repro
.venv-repro/Scripts/python.exe -m pip install -r requirements-lock-windows-py310.txt
.venv-repro/Scripts/python.exe -m unittest discover -s tests -v
.venv-repro/Scripts/python.exe src/24_clean_reproduction.py
.venv-repro/Scripts/python.exe src/25_validate_clean_notebooks.py
```

El script 24 crea una carpeta nueva, copia únicamente DEVELOPMENT y recalcula calidad, EDA, modelos y diagnósticos. Requiere Node para dos auditorías JavaScript. El script 25 ejecuta los seis notebooks de comprobación dentro de esa carpeta. El registro incluido documenta la ejecución ya realizada; las rutas absolutas en él identifican el equipo de origen. La igualdad numérica se comprueba con tolerancias, no se exige igualdad binaria entre plataformas.

Para la entrega unificada, abrir `notebooks/Entregable_1_Completo.ipynb` con el kernel del entorno anterior y seleccionar **Restart Kernel and Run All Cells**. Para ejecutarlo automáticamente y guardar sus salidas: `.venv-repro/Scripts/python.exe src/29_execute_compiled_notebook.py`. El script `28_build_compiled_notebook.py` regenera su contenido desde el informe y el código; ejecutarlo borra las salidas del notebook, por lo que debe ir seguido del script 29. El registro de ejecución unificada queda en `outputs/tables/compiled_notebook_execution.json`.

El snapshot `data/splits/development_80.csv` tiene SHA-256 `5fbbc54a12e976551b35652b72c989be8d1a120ef41cd7c6a4b50dcaf5fa87a2`. El paquete reproducible no contiene TEST ni el archivo maestro. Los scripts de descarga y partición se conservan como procedencia; no son pasos de la reproducción de DEVELOPMENT. Los derechos y términos del proveedor se describen en la sección 1 del informe.

Para construir el Book, con Node/npm:

```powershell
cd book
npm exec --yes --package=mystmd@1.11.0 -- myst build --html
python -m http.server 8000 --directory _build/html
```

Abrir `http://localhost:8000`. El HTML necesita un servidor local para resolver correctamente rutas y recursos. La compilación puede descargar MyST y su plantilla; se verificó con Node 24.19.0. Los generadores de secciones deben ejecutarse secuencialmente; cada sección documenta su procedimiento de reproducción.

## Organización

`src/` contiene cálculos y generadores; `tests/`, controles de fronteras y diagnósticos; `outputs/tables/`, protocolos, métricas y metadatos; `outputs/models/`, los pipelines finales sobre DEVELOPMENT; `book/_static/figures/`, las figuras utilizadas. Los scripts anteriores identificados como antecedentes no sustituyen los procedimientos vigentes enlazados en cada sección.
