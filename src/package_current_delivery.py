"""Package the supplementary audits, preserving the original delivery ZIP."""
from pathlib import Path
import csv
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/current_delivery_audit'
DEST = ROOT / 'delivery/Entregable1_auditoria_actualizada.zip'

ROWS = [
    ('1.3', 'Condiciones y atribución', 'Resuelto documentalmente', 'licence/evidence.json'),
    ('1.6', '20.000 ejemplos supervisados', 'No cumplido: límite del diseño diario', 'quality/methodology.json'),
    ('1.6.2', 'Rango de volatilidad objetivo', 'Calculado', 'quality/target_ranges.csv'),
    ('1.6.3', 'Tamaño efectivo', 'Estimación descriptiva con límites', 'quality/effective_sample_size.csv'),
    ('1.7.1a', 'Matrices y propagación de nulos', 'Calculado', 'quality/minute_missingness_matrix.csv'),
    ('1.7.1b', 'Mecanismo MCAR/MAR/MNAR', 'No identificable con una interrupción', 'quality/methodology.json'),
    ('1.7.2', 'Duplicados y casi duplicados', 'Auditado', 'quality/raw_file_audit.csv'),
    ('1.7.3', 'Outliers y sensibilidad', 'Calculado sin cambiar modelos', 'quality/outlier_sensitivity.csv'),
    ('1.7.4', 'Imposibles y exclusiones por archivo', 'Auditado', 'quality/raw_file_audit.csv'),
    ('2.2', 'Percentiles asimetría curtosis boxplots', 'Calculado en desarrollo', 'eda/univariate.csv.gz'),
    ('2.3', 'Asociaciones y redundancia', 'Calculado con objetivos futuros alineados', 'eda/future_associations.csv.gz'),
    ('2.4', 'PCA y extremos multivariados', 'Exploratorio ajustado solo en 2023', 'eda/pca_summary.csv'),
    ('2.5', 'Disponibilidad y fuga temporal', 'Verificado', 'eda/metadata.json'),
    ('2.6', 'STL ADF KPSS PACF estacionalidad deriva', 'Calculado con límites temporales', 'eda/stationarity.csv'),
    ('3.5', 'Intervalos temporales de métricas', 'Calculado con sensibilidad 35/56/70 días', 'model/metric_confidence_intervals.csv'),
    ('3.6', 'Normalidad heterocedasticidad autocorrelación', 'Calculado sobre 112 series actuales', 'model/residual_diagnostics.csv'),
    ('3.7', 'Curva de aprendizaje', '384 ajustes auxiliares 2023–2024', 'model/learning_curve.csv'),
    ('3.8', 'Coeficientes y ambos escaladores', 'Reconstruido y verificado', 'model/coefficient_prediction_verification.csv'),
    ('4', 'Contraste con persistencia', 'Bootstrap pareado con ajuste Holm', 'model/paired_loss_bootstrap.csv'),
    ('reserva2025', 'Reserva inicial intacta de 2025', 'No recuperable: periodo ya explorado', 'model/provenance.json'),
    ('adicional2026', 'Periodo adicional con protocolo congelado', '236 orígenes evaluados sin reentrenar', 'holdout/summary.json'),
]


def main():
    for _, _, _, evidence in ROWS:
        if not (OUT / evidence).is_file():
            raise ValueError(f'Missing evidence: {evidence}')
    with (OUT / 'resolution.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['requisito', 'descripcion', 'resolucion', 'evidencia'])
        writer.writerows(ROWS)
    (OUT / 'README.txt').write_text(
        'Suplemento de auditoría del entregable: 4 de octubre de 2026.\n'
        'Datos: Binance Vision. Análisis: Jassan Arteta y Mateo Bernal.\n'
        'Consultar DATA_LICENSE.md para licencia y atribución.\n'
        'Quality: 144 archivos RAW y cinco procesados; rangos, nulos, duplicados, extremos.\n'
        'EDA: representación vigente, análisis de desarrollo 2023–2024 y deriva descriptiva 2025.\n'
        'Model: diagnósticos del SVR guardado, bootstrap temporal, curvas auxiliares y coeficientes.\n'
        'Holdout: protocolo anterior a descarga, enero–agosto2026, 236 orígenes, sin refit/selección.\n'
        'El uso previo externo de 2026 no está establecido por los artefactos auditados.\n'
        '2025 sigue siendo retrospectivo. 20.000 muestras supervisadas no se alcanzan.\n'
        'MCAR/MAR/MNAR no es identificable con esta interrupción única.\n'
        'Reproducción: book/sections/15_reproducibilidad.md y scripts incluidos.\n'
        'Los RAW originales se reconstruyen con src/download_three_years.py;\n'
        'los datos de 2026 con src/evaluate_frozen_holdout.py download.\n', encoding='utf-8')
    sources = sorted(p for p in OUT.rglob('*') if p.is_file() and p.name != 'manifest.json')
    sources += sorted((ROOT / 'src').glob('audit_current_*.py'))
    sources += [ROOT / 'src/evaluate_frozen_holdout.py', ROOT / 'src/report_current_holdout.py',
                ROOT / 'src/verify_current_delivery.py',
                Path(__file__), ROOT / 'DATA_LICENSE.md']
    sources += sorted((ROOT / 'tests').glob('test_current_*_audit.py'))
    sources += sorted((ROOT / 'book/figures').glob('current_*.png'))
    sources += [ROOT / 'book/entregable1_master.md']
    sources += sorted((ROOT / 'book/sections').glob('1[1-5]_*.md'))
    manifest = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    sources.append(OUT / 'manifest.json')
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DEST, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, str(path.relative_to(ROOT)).replace('\\', '/'))
    with zipfile.ZipFile(DEST) as archive:
        if archive.testzip() is not None:
            raise ValueError('Corrupt supplemental package')
    print('Packaged', len(sources), 'files,', DEST.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
