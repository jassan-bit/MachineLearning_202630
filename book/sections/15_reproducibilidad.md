# 5. Reproducibilidad y referencias

## 5.1 Reproducción y archivos

Desde la raíz, con las dependencias de `requirements.txt` y `requirements-dashboard.txt` instaladas:

```bash
python src/optimize_minute_svr.py
python src/report_optimized_svr.py
python src/apply_optimized_delivery.py
```

La entrega incluye los datos procesados de minuto, los 16 modelos, las 640 búsquedas, calendarios, auditoría nativa, predicciones, métricas, figuras y notebook ejecutado. Para regenerar los datos desde Binance se incluyen los scripts de descarga y preparación y el registro SHA-256 de fuentes. El entrenamiento tarda varios minutos; no es necesario repetirlo para consultar los resultados.

Notebook vigente: [Entregable_1_Minuto.ipynb](../../notebooks/Entregable_1_Minuto.ipynb). Paquete: [Entregable1_optimizado_2023_2025.zip](../../delivery/Entregable1_optimizado_2023_2025.zip). Resultados: `results/optimized_minute_2023_2025`. El dashboard/API anteriores continúan asociados al modelo original y no sirven los modelos optimizados. El informe vigente se consulta en el HTML incluido o en localhost:8051.


## 5.2 Auditoría adicional ejecutada el 4 de octubre de 2026

La revisión se calculó sobre los datos y los 16 modelos vigentes. Sus
resultados complementan la entrega original: calidad RAW y procesada,
EDA de desarrollo, diagnósticos de residuos, intervalos temporales,
curva de aprendizaje e interpretación de coeficientes. Los capítulos
1–4 incluyen tablas, figuras, método y descargas de cada resultado.

Desde la raíz, para reproducir los diagnósticos:

La auditoría de calidad requiere los 144 archivos RAW identificados en
`results/minute_2023_2025/sources.csv`. Si no están en la copia local,
`python src/download_three_years.py` reconstruye esas fuentes y verifica sus
checksums oficiales. Los análisis restantes utilizan los datos procesados
y los modelos incluidos en la entrega.

```bash
python src/audit_current_licence.py
python src/audit_current_dataset.py
python src/audit_current_eda.py --check
python src/audit_current_eda.py
python src/audit_current_model.py --repetitions 2000
python src/audit_current_model.py --additional-bootstrap --repetitions 2000
```

Las curvas de aprendizaje realizan ajustes auxiliares exclusivamente
con 2023–2024; los pronósticos originales y sus modelos siguen siendo los
artefactos de referencia. Se verificaron límites temporales, alineación
de las muestras, conservación de hashes, métricas recalculadas e inversión
de los escaladores. Las pruebas nuevas se ejecutan mediante
`python -m unittest discover -s tests -p 'test_current_*_audit.py'`.

La evaluación adicional usa **enero–agosto de 2026**: 236 orígenes, de
1 de enero a 24 de agosto, con siete objetivos completos. El
[protocolo](../../results/current_delivery_audit/holdout/protocol.json)
y su [SHA-256](../../results/current_delivery_audit/holdout/protocol.sha256)
se fijaron antes de descargar esos datos. Se reutilizan los modelos guardados,
sin selección ni nuevo entrenamiento. La descarga se puede reconstruir con
`python src/evaluate_frozen_holdout.py download`; el modo `evaluate` comprueba
el protocolo y evita sobrescribir una evaluación ya completada. En una
copia de trabajo separada se puede retirar únicamente su resumen de salida
para regenerar los resultados con el mismo protocolo. El modo `freeze`
sirve para registrar un estudio nuevo antes de su primera descarga y rechaza
sobrescribir este registro.

Se publican [fuentes y hashes de 2026](../../results/current_delivery_audit/holdout/sources.csv),
[controles por archivo](../../results/current_delivery_audit/holdout/quality.csv),
[predicciones](../../results/current_delivery_audit/holdout/predictions.csv.gz)
y [verificación de resultados](../../results/current_delivery_audit/holdout/summary.json).
Los archivos nuevos se guardan localmente en carpetas separadas
`data/raw/frozen_holdout_2026` y `data/processed/frozen_holdout_2026`.
El informe distingue este periodo adicional de la evaluación retrospectiva
de 2025 y del posible uso previo de 2026 fuera de los artefactos auditados.

Descargar el [suplemento completo de evidencias, scripts y figuras](../../delivery/Entregable1_auditoria_actualizada.zip)
y el [registro de resolución de requisitos](../../results/current_delivery_audit/resolution.csv).
Incluye [atribución y licencia de datos y derivados](../../results/current_delivery_audit/licence/DATA_LICENSE.txt).
El paquete original conserva sus resultados de 2025; el suplemento contiene
los análisis adicionales y los resultados de 2026.

(reproduccion-retroalimentacion)=
## 5.3 Ampliación en respuesta a la retroalimentación

:::{div}
:class: correccion-profesor

Los análisis del **18 de octubre de 2026** se incorporan al Jupyter Book
en esta revisión documental del **21 de octubre de 2026**. No se repitió
el entrenamiento para elaborar esta revisión: se contrastaron resultados
guardados, manifiestos, protocolos y pruebas del código que los produjo.

| Evidencia | Contenido y procedencia |
| --- | --- |
| [Metadatos del EDA complementario](../../results/professor_feedback_20261018/diagnostics/metadata.json) | Periodos accedidos, métodos, tamaños, versiones y hashes de entrada y salida para información mutua, VIF y perfiles intradía. |
| [Referencia HAR diario](../../results/professor_feedback_20261018/daily_har/selection_freeze.json) | Registro de congelación del HAR diario OLS de escalas 1/7/28 y sus modelos. |
| [Referencias HAR intradía](../../results/professor_feedback_20261018/har_baseline/selection_freeze.json) | Registro de los ajustes OLS y Ridge de escalas 1/7/28 sobre variación intradía. |
| [Corrección AR](../../results/professor_feedback_20261018/residuals/selection_freeze.json) | Selección y congelación de los correctores para errores maduros de HAR-Ridge + XGBoost. |
| [Protocolo de comparación](../../results/professor_feedback_20261018/evaluation_protocol.json) | Familias de contrastes, calendario y regla de sustitución de la ampliación. |
| [Resumen verificado](../../results/professor_feedback_20261018/summary.json) | Métricas, contrastes y hashes; 209 archivos protegidos permanecieron sin cambios durante el experimento. |
| [Manifiesto de revisión](../../results/professor_feedback_20261018/review_manifest.json) | Hashes de scripts, pruebas e informe y registro de 23 pruebas aprobadas en la ejecución original. |

Los campos `LOCAL ONLY` y `github_or_render_updated=false` de los manifiestos
describen el estado al ejecutar los análisis el 18 de octubre. No son
constancia de la publicación de esta revisión documental.

Las pruebas comprueban causalidad temporal de las entradas, disponibilidad
de los errores por horizonte, exclusión de etiquetas futuras, integridad
de calendarios y emparejamiento de predicciones. Se ejecutan desde la raíz:

```powershell
.venv-repro/Scripts/python.exe -m unittest discover -s tests -p 'test_professor_feedback*.py' -v
```

Para inspeccionar el método sin volver a entrenar, se incluyen los scripts
[diagnósticos](../../src/professor_feedback_diagnostics.py),
[HAR diario](../../src/professor_feedback_daily_har.py),
[HAR intradía](../../src/professor_feedback_har_baseline.py),
[corrección AR](../../src/professor_feedback_residuals.py),
[evaluación](../../src/evaluate_professor_feedback.py) e
[informe](../../src/report_professor_feedback.py).
Los scripts de entrenamiento son la procedencia de los resultados guardados;
no es necesario ejecutarlos para leer o verificar las tablas publicadas.

Esta ampliación se diseñó cuando **2025 y 2026 ya habían sido inspeccionados**.
Sus registros de congelación evitan elegir después de calcular los nuevos
resultados, pero no convierten esos años en pruebas confirmatorias intactas.
La evaluación adicional del apartado 4.4 conserva su protocolo y su alcance
histórico propios; no se atribuye ese protocolo inicial a los modelos añadidos.

Los ZIP, el notebook y el HTML de la entrega original enlazados en este libro
conservan la versión que figura en sus manifiestos. Para esta ampliación deben
consultarse los capítulos resaltados y las evidencias específicas aquí enlazadas.

:::

:::{div}
:class: correccion-profesor

## 5.4 Referencias

:::

- Binance. [Datos públicos de mercado](https://data.binance.vision/) y [API](https://www.binance.com/en/binance-api).
- [timeseries-cv](https://pypi.org/project/timeseries-cv/), versión 0.1.5.
- Ramos, F. (2021). *Data Science na Modelação e Previsão de Séries Económico-financeiras: das Metodologias Clássicas ao Deep Learning*. Tesis doctoral, Instituto Universitário de Lisboa, ISCTE Business School.
- Scikit-learn. [LinearSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html).
