"""Validate exported temporal cuts without allocating every native CV matrix."""
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
from tsxv import splitTrainVal as tsxv


METHODS = ('forwardChaining', 'kFold', 'groupKFold')
AUDIT_COLUMNS = ['method', 'native_fold', 'n_train', 'n_val', 'train_last',
                 'val_first', 'future_train_origins',
                 'train_labels_at_or_after_first_validation', 'source_sha256']
BOUNDARY = pd.Timestamp('2025-01-01', tz='UTC')


def _native_audit(out, development):
    audit = pd.read_csv(out / 'native_cv_audit.csv')
    if list(audit.columns) != AUDIT_COLUMNS or set(audit.method) != set(METHODS):
        raise ValueError('Auditoría nativa incompleta.')
    numeric = audit[AUDIT_COLUMNS[1:-1]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError('La auditoría nativa requiere índices y recuentos enteros.')
    if (numeric < 0).any() or (audit[['n_train', 'n_val']] < 1).any().any():
        raise ValueError('Recuentos nativos inválidos.')
    if ((audit.train_last + 7 >= len(development)).any()
            or (audit.val_first + 7 >= len(development)).any()
            or (audit.n_train > len(development)).any()
            or (audit.n_val > len(development)).any()
            or (audit.future_train_origins > audit.n_train).any()
            or (audit.train_labels_at_or_after_first_validation > audit.n_train).any()
            or (audit.train_labels_at_or_after_first_validation < audit.future_train_origins).any()):
        raise ValueError('La auditoría nativa contiene límites o recuentos inconsistentes.')
    for name in METHODS:
        rows = audit[audit.method.eq(name)]
        function = getattr(tsxv, 'split_train_val_' + name)
        digest = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
        if not rows.source_sha256.eq(digest).all():
            raise ValueError(f'El código nativo cambió: {name}.')
        count = 5 if name == 'groupKFold' else len(development) - 63
        if not np.array_equal(rows.native_fold.to_numpy(), np.arange(count)):
            raise ValueError(f'Folds nativos incompletos o desordenados: {name}.')
        if name != 'groupKFold' and not rows.n_val.eq(1).all():
            raise ValueError(f'Validación nativa incorrecta: {name}.')
        if not rows.future_train_origins.eq(0).equals(rows.train_last.le(rows.val_first)):
            raise ValueError(f'Conteo de orígenes futuros inconsistente: {name}.')
        if not rows.train_labels_at_or_after_first_validation.gt(0).equals(
                (rows.train_last + 7).ge(rows.val_first)):
            raise ValueError(f'Conteo de etiquetas nativas inconsistente: {name}.')
    forward = audit[audit.method.eq('forwardChaining')]
    sizes = np.arange(2, len(development) - 61)
    expected = np.column_stack([sizes, sizes + 26, sizes + 54])
    if not np.array_equal(forward[['n_train', 'train_last', 'val_first']], expected):
        raise ValueError('Los cortes Forward Chaining no corresponden al calendario original.')
    return audit


def saved_calendar(panel, out):
    """Return the original six purged folds, eligibility mask and native audit.

    The saved cuts are checked against the compact native Forward Chaining
    indices, the source hashes and all eligible dates in each validation block.
    No native splitter is executed and no fitted artifact is changed.
    """
    out = Path(out)
    dates = panel.index
    if not isinstance(dates, pd.DatetimeIndex) or str(dates.tz) != 'UTC':
        raise ValueError('El calendario requiere fechas UTC.')
    if not dates.is_unique or not dates.is_monotonic_increasing or len(dates) < 2:
        raise ValueError('El calendario requiere fechas únicas y ordenadas.')
    if not np.all(np.diff(dates.asi8) == pd.Timedelta(days=1).value):
        raise ValueError('El calendario requiere observaciones diarias consecutivas.')
    development = np.flatnonzero(dates < BOUNDARY)
    if len(development) < 64:
        raise ValueError('No hay suficientes fechas de development.')
    audit = _native_audit(out, development)
    calendar = pd.read_csv(out / 'calendar.csv')
    if (list(calendar.columns) != ['fold', 'split', 'origin', 'target_end']
            or set(calendar.fold) != set(range(1, 7))
            or set(calendar.split) != {'train', 'validation'}):
        raise ValueError('El calendario guardado no contiene los seis folds originales.')
    origins = pd.to_datetime(calendar.origin, utc=True)
    target_end = pd.to_datetime(calendar.target_end, utc=True)
    indices = dates.get_indexer(origins)
    if (indices < 0).any() or (indices + 7 >= len(dates)).any():
        raise ValueError('Fechas guardadas fuera del dataset.')
    if not np.array_equal(target_end.array.asi8, dates[indices + 7].asi8):
        raise ValueError('target_end no corresponde al horizonte de siete días.')
    if (target_end >= BOUNDARY).any():
        raise ValueError('Las etiquetas del calendario deben terminar antes de 2025.')
    finite = np.isfinite(panel.to_numpy()).all(axis=1)
    eligible = pd.Series(finite).rolling(36).sum().shift(-7).eq(36).to_numpy()
    forward = audit[audit.method.eq('forwardChaining')]
    folds = []
    for fold, month in enumerate([1, 3, 5, 7, 9, 11], 1):
        start = pd.Timestamp(f'2024-{month:02d}-01', tz='UTC')
        first = dates.get_indexer([start])[0]
        native = forward[forward.val_first.eq(first)]
        if first < 0 or len(native) != 1:
            raise ValueError(f'No hay un corte nativo para el fold {fold}.')
        train = np.arange(27, int(native.iloc[0].train_last) + 1)
        train = train[eligible[train] & (train + 7 < first)]
        validation = np.flatnonzero(
            eligible & (dates >= start) & (dates < start + pd.DateOffset(months=2))
            & (dates + pd.Timedelta(days=7) < BOUNDARY))
        if not len(train) or not len(validation) or train.max() + 7 >= validation.min():
            raise ValueError(f'Purga temporal incorrecta en el fold {fold}.')
        for role, expected_indices in [('train', train), ('validation', validation)]:
            selected = calendar.fold.eq(fold) & calendar.split.eq(role)
            if not np.array_equal(indices[selected], expected_indices):
                raise ValueError(f'Fechas de {role} modificadas en el fold {fold}.')
        folds.append((train, validation))
    return folds, eligible, audit
