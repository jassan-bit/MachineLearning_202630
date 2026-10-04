"""Render the frozen evaluation without downloading, refitting or selecting."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/current_delivery_audit/holdout'


def main():
    metrics = pd.read_csv(OUT / 'metrics.csv')
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, symbol in zip(axes.flat, symbols):
        part = metrics[metrics.symbol.eq(symbol)].pivot(index='volatility_window', columns='model', values='rmse')
        x = np.arange(len(part))
        ax.bar(x - .18, part.Persistence, .36, label='Persistencia', color='#9ca3af')
        ax.bar(x + .18, part.SVR_optimized, .36, label='SVR lineal', color='#2563eb')
        ax.set_title(symbol.replace('USDT', ''))
        ax.set_xticks(x, part.index)
        ax.set_xlabel('Ventana del objetivo (días)')
        ax.set_ylabel('RMSE medio por horizonte (pp)')
        ax.grid(axis='y', alpha=.2)
    axes.flat[0].legend(frameon=False)
    fig.suptitle('Evaluación adicional 2026 · 236 orígenes · modelos guardados', fontsize=15)
    fig.tight_layout(rect=(0, .035, 1, .96))
    fig.text(.5, .01, 'Protocolo fijado antes de descargar enero–agosto de 2026; sin nuevo ajuste ni selección.', ha='center', fontsize=9)
    figure = ROOT / 'book/figures/current_holdout_metrics.png'
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    macro = pd.read_csv(OUT / 'macro_metrics.csv').set_index('model')
    improvement = 100 * (1 - macro.loc['SVR_optimized', 'rmse'] / macro.loc['Persistence', 'rmse'])
    print(f'Macro RMSE reduction: {improvement:.5f}%')
    pair = metrics.pivot(index=['symbol', 'volatility_window'], columns='model', values='rmse')
    print('Configurations with lower RMSE:', int((pair.SVR_optimized < pair.Persistence).sum()), '/', len(pair))


if __name__ == '__main__':
    main()
