"""Presentation of verified macro metrics, without changing model selection."""
import pandas as pd


def comparison_tables(macro):
    global_ = macro[macro.symbol == 'GLOBAL_MACRO'].set_index('model')
    rows = []
    for key, label in [('r2','R² ↑'),('rmse','RMSE ↓'),('mae','MAE ↓'),('mse','MSE ↓'),('mape','MAPE (%) ↓')]:
        baseline = float(global_.loc['Persistence',key])
        svr = float(global_.loc['SVR_optimized',key])
        improvement = f'+{svr-baseline:.4f} puntos de R²' if key == 'r2' else f'{100*(1-svr/baseline):.2f} % menos error'
        rows.append({'Métrica':label,'Persistencia':baseline,'SVR lineal':svr,'Mejora del SVR':improvement})
    detail = macro[macro.symbol != 'GLOBAL_MACRO'].copy()
    detail['model'] = detail.model.map({'Persistence':'Persistencia','SVR_optimized':'SVR lineal'})
    detail = detail.rename(columns={'symbol':'Criptomoneda','model':'Modelo','r2':'R² ↑','rmse':'RMSE ↓','mae':'MAE ↓','mse':'MSE ↓','mape':'MAPE (%) ↓'})
    interpretation = [
        'La comparación utiliza las mismas 358 fechas de prueba de 2025 y los mismos objetivos. Persistencia repite la última volatilidad observada durante los siete horizontes; el SVR lineal optimizado aprende una corrección relativa a esa referencia. ↑ indica que un valor mayor es mejor y ↓ que un valor menor es mejor.',
        'El R² macro aumenta de 0,5377 a 0,7007: una diferencia de 0,1630 puntos. El SVR explica mejor la variación de los objetivos, en promedio entre horizontes, ventanas y activos. Este R² no representa un 70,07 % de pronósticos correctos ni es el R² calculado sobre todas las series concatenadas.',
        'El RMSE disminuye aproximadamente un 20,01 %, el MAE un 16,73 % y el MSE un 35,14 %. La reducción conjunta indica menores errores absolutos y cuadrados en esta evaluación. El MSE penaliza más los errores grandes. El RMSE publicado promedia los RMSE de los horizontes y configuraciones, por lo que no coincide necesariamente con la raíz del MSE macro.',
        'El MAPE baja de 18,81 % a 15,64 %, aproximadamente un 16,85 % de reducción relativa. Describe el error relativo a la volatilidad real y puede amplificar los errores cuando esta es pequeña. Debe interpretarse junto con MAE y RMSE, no como porcentaje de aciertos.',
        'El SVR mejora las cinco métricas agregadas en las cuatro criptomonedas. BNB alcanza el mayor R² (0,8156); ETH conserva el menor (0,5932), aunque mejora frente a persistencia (0,3426). XRP mantiene el mayor RMSE absoluto (0,9047): las escalas de volatilidad difieren entre activos, por lo que este orden no equivale por sí solo a una comparación de dificultad.',
        'El SVR supera a persistencia en RMSE en las 16 combinaciones seleccionadas de activo y ventana objetivo. Esto no implica ganar en todos los días ni en cada horizonte. La evaluación es retrospectiva y los objetivos móviles se superponen; estas tablas no demuestran significancia estadística ni garantizan resultados futuros.',
        'RMSE y MAE están en puntos porcentuales de volatilidad no anualizada; MSE está en puntos porcentuales al cuadrado; MAPE está en porcentaje. Se seleccionó con validación de 2024 y se reajustó con etiquetas anteriores a 2025. Las mejoras aquí comparan el SVR optimizado con persistencia; la comparación con el SVR original combina cambios de características, validación y periodo de ajuste.'
    ]
    return pd.DataFrame(rows), detail, interpretation
