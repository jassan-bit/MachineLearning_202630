# 5. Reproducibilidad y referencias

## Reproducción y archivos

Desde la raíz, con las dependencias de `requirements.txt` y `requirements-dashboard.txt` instaladas:

```bash
python src/optimize_minute_svr.py
python src/report_optimized_svr.py
python src/apply_optimized_delivery.py
```

La entrega incluye los datos procesados de minuto, los 16 modelos, las 640 búsquedas, calendarios, auditoría nativa, predicciones, métricas, figuras y notebook ejecutado. Para regenerar los datos desde Binance se incluyen los scripts de descarga y preparación y el registro SHA-256 de fuentes. El entrenamiento tarda varios minutos; no es necesario repetirlo para consultar los resultados.

Notebook vigente: [Entregable_1_Minuto.ipynb](../../notebooks/Entregable_1_Minuto.ipynb). Paquete: [Entregable1_optimizado_2023_2025.zip](../../delivery/Entregable1_optimizado_2023_2025.zip). Resultados: `results/optimized_minute_2023_2025`. El dashboard/API anteriores continúan asociados al modelo original y no sirven los modelos optimizados. El informe vigente se consulta en el HTML incluido o en localhost:8051.


## Referencias

- Binance. [Datos públicos de mercado](https://data.binance.vision/) y [API](https://www.binance.com/en/binance-api).
- [timeseries-cv](https://pypi.org/project/timeseries-cv/), versión 0.1.5.
- Ramos, F. (2021). *Data Science na Modelação e Previsão de Séries Económico-financeiras: das Metodologias Clássicas ao Deep Learning*. Tesis doctoral, Instituto Universitário de Lisboa, ISCTE Business School.
- Scikit-learn. [LinearSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html).
