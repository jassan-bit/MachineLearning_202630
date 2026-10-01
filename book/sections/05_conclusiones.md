> **Referencia histórica:** estudio anterior 2020–2025 con entradas diarias. Consulte el [estudio vigente de tres años](07_estudio_minuto.md).

# 5. Conclusiones

## Respuesta a la pregunta

| symbol | volatility_window | input_window | Persistence | SVR | SVR_minus_Persistence |
| --- | --- | --- | --- | --- | --- |
| BNBUSDT | 7 | 21 | 1.04053 | 2.87727 | 1.83675 |
| BNBUSDT | 14 | 28 | 0.62201 | 3.42416 | 2.80215 |
| BNBUSDT | 21 | 7 | 0.33270 | 3.27438 | 2.94168 |
| BNBUSDT | 28 | 7 | 0.14609 | 2.83684 | 2.69075 |
| BTCUSDT | 7 | 14 | 0.84373 | 4.79559 | 3.95186 |
| BTCUSDT | 14 | 14 | 0.37277 | 5.14443 | 4.77167 |
| BTCUSDT | 21 | 21 | 0.22854 | 5.21366 | 4.98512 |
| BTCUSDT | 28 | 28 | 0.22821 | 4.40628 | 4.17807 |
| ETHUSDT | 7 | 21 | 1.30690 | 2.41621 | 1.10930 |
| ETHUSDT | 14 | 21 | 0.47784 | 1.88671 | 1.40887 |
| ETHUSDT | 21 | 28 | 0.32266 | 2.14370 | 1.82104 |
| ETHUSDT | 28 | 28 | 0.26969 | 2.39408 | 2.12439 |
| XRPUSDT | 7 | 14 | 2.06372 | 9.78647 | 7.72275 |
| XRPUSDT | 14 | 14 | 0.72504 | 8.62323 | 7.89819 |
| XRPUSDT | 21 | 28 | 0.58286 | 9.06631 | 8.48345 |
| XRPUSDT | 28 | 28 | 0.40955 | 7.13051 | 6.72096 |

La ventana de entrada se seleccionó por validación para cada activo y ventana del objetivo. De las 16
comparaciones resultantes, SVR obtiene menor RMSE test que Persistence en 0. Los resultados completos
permiten examinar qué activo y definición de volatilidad favorecen cada modelo, sin reunir objetivos diferentes
en una única clasificación por RMSE. Las métricas corresponden a los modelos de los folds, no al reajuste de despliegue.

## Limitaciones

El Group K-Fold nativo necesita filtros cronológicos; esos filtros reducen las muestras.
Los folds dependientes y los pocos residuos limitan la interpretación de su variación y de BDS.
La volatilidad móvil comparte retornos entre salidas y puede contener retornos ya conocidos al emitir el pronóstico.
El historial de 2020–2022 no representa necesariamente las condiciones de 2024–2025.
Los días incompletos se excluyen con una política conservadora, y los activos se seleccionaron retrospectivamente.
Un SVR lineal puede predecir valores negativos; se reportan en las tablas sin recortarlos después de observar test.

## Continuidad

Una evaluación posterior puede ampliar los orígenes de test con una estrategia temporal más densa,
manteniendo estos resultados como referencia y sin reusar este test para afirmar una nueva evaluación independiente.
Los 16 modelos de API se reajustan con todos los datos elegibles de 2020–2025 una vez fijados los parámetros por validación.
Ese reajuste sirve a la inferencia y no tiene una nueva métrica fuera de muestra.
