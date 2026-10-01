# Ejecutar el dashboard del Entregable 2

Desde la raíz, en el entorno Python del proyecto:

```powershell
python -m pip install -r requirements-dashboard.txt
python app.py
```

Abrir http://127.0.0.1:8050. En este equipo también se puede ejecutar con `.venv-repro/Scripts/python.exe app.py`.

## Render: servicio web manual

- Repositorio: `jassan-bit/MachineLearning_202630`, rama `main`.
- Lenguaje: Python 3; variable de entorno `PYTHON_VERSION=3.11.11`.
- Root Directory: vacío.
- Build Command: `pip install -r requirements-dashboard.txt`.
- Start Command: `gunicorn app:server --bind 0.0.0.0:$PORT --workers 1`.
- Compute: Free.

El archivo de dependencias del dashboard es diferente del utilizado para reproducir el entrenamiento. Evita instalar el entorno científico completo en el servidor web. Como alternativa, `render.yaml` contiene esta configuración para Blueprint.

El servicio requiere `app.py`, `assets/dashboard.css`, `requirements-dashboard.txt` y los cinco CSV de `dashboard_data/`. No necesita TEST, credenciales de Binance ni modelos serializados. Usa resultados guardados de DEVELOPMENT.

## Informe y comprobaciones

El capítulo `book/sections/06_dashboard.md` interpreta las seis gráficas y enlaza la metodología existente. El menú del Book incluye este capítulo. `BOOK_URL` permite cambiar la dirección del informe enlazado por el dashboard.

```powershell
python -m unittest discover -s tests -p test_dashboard.py -v
```

Las pruebas contrastan los filtros con las tablas de origen para cinco activos, seis selecciones de fold y cuatro métricas, además de comprobar rutas HTTP y cuantiles. El despliegue Linux con Gunicorn se debe confirmar en los registros de Render; la ejecución local Windows utiliza el servidor de desarrollo de Dash.
