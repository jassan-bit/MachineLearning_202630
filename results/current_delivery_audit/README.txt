Suplemento de auditoría del entregable: 4 de octubre de 2026.
Datos: Binance Vision. Análisis: Jassan Arteta y Mateo Bernal.
Consultar DATA_LICENSE.md para licencia y atribución.
Quality: 144 archivos RAW y cinco procesados; rangos, nulos, duplicados, extremos.
EDA: representación vigente, análisis de desarrollo 2023–2024 y deriva descriptiva 2025.
Model: diagnósticos del SVR guardado, bootstrap temporal, curvas auxiliares y coeficientes.
Holdout: protocolo anterior a descarga, enero–agosto2026, 236 orígenes, sin refit/selección.
El uso previo externo de 2026 no está establecido por los artefactos auditados.
2025 sigue siendo retrospectivo. 20.000 muestras supervisadas no se alcanzan.
MCAR/MAR/MNAR no es identificable con esta interrupción única.
Reproducción: book/sections/15_reproducibilidad.md y scripts incluidos.
Los RAW originales se reconstruyen con src/download_three_years.py;
los datos de 2026 con src/evaluate_frozen_holdout.py download.
