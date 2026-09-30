"""Checklist documental: distingue evidencia ejecutada de límites y pendientes."""
from pathlib import Path
import csv
import json
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/tables'


def main():
    path=OUT/'entregable1_checklist.csv'
    with path.open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f);fields,rows=reader.fieldnames,list(reader)
    base_sections={1:'1.1',2:'1.2',3:'1.3',4:'1.3',5:'1.4',6:'1.4',7:'1.4',8:'1.4',
        9:'1.5',10:'1.5',11:'1.5',12:'1.6',13:'1.6.1',14:'1.6.1',15:'1.6.2',16:'1.6.2',17:'1.6.3',
        18:'1.7.1–1.7.3',19:'1.7.2',20:'1.7.2',21:'1.7.3',22:'1.7.3',23:'1.7.4',24:'1.7.4',
        25:'1.7.5',26:'1.7.5',27:'1.7.6',28:'1.7.6',29:'1.7.6',30:'1.7.7',31:'1.7.7',32:'1.8',33:'1.8',34:'1.8'}
    exceptions={
        '1.15':('NO APLICA','Regresión continua; no hay clases.'),
        '1.21':('PARCIALMENTE CUMPLIDO','Se describe evidencia y se reconoce que no identifica MCAR/MAR/MNAR. Añadir registros externos de adquisición si se pretende atribuir el mecanismo; no asignarlo sin evidencia.'),
        '1.22':('PARCIALMENTE CUMPLIDO','Comparación previa a diez huecos, descriptiva y dependiente; insuficiente para identificar el mecanismo. Mantener esta limitación o aportar evidencia externa.'),
        '1.33':('NO APLICA','Velas agregadas sin identificadores de personas; no se ejecuta anonimización innecesaria.'),
        '2.2.8':('NO APLICA','Se emplean histogramas, alternativa admitida a densidades por la guía.'),
        '2.2.11':('NO APLICA','No se presupone normalidad marginal para SVR; se justifica no aplicar pruebas iid a estas series.'),
        '2.2.17':('NO APLICA','No hay categorías raras que requieran tratamiento.'),
        '2.5.7':('PARCIALMENTE CUMPLIDO','La auditoría de partición descarta claves temporales y filas completas compartidas; no certifica casi-duplicados ignorando fecha entre DEVELOPMENT y TEST. No modificar el modelo examinando TEST.'),
        '2.6.27':('PARCIALMENTE CUMPLIDO','Se comparan periodos y se visualizan cambios; no se valida un modelo formal de regímenes. Ampliar si se afirma un régimen específico.'),
        '2.6.28':('NO CUMPLIDO','No hay estimación formal de puntos de cambio. Añadir método, supuestos, sensibilidad y localización con incertidumbre, únicamente en DEVELOPMENT, antes de afirmar cambios detectados.'),
        '2.6.29':('NO CUMPLIDO','No existe cronología de eventos contrastada con fuentes. Añadir fechas y referencias y distinguir coincidencia temporal de causalidad.'),
        '2.6.32':('PARCIALMENTE CUMPLIDO','Se distingue deriva marginal de cambio condicional; los gráficos no prueban concept drift. Añadir diagnóstico condicional si se pretende afirmarlo.'),
        '2.9.3':('NO APLICA','Se excluyen ventanas incompletas sin imputar; la decisión está justificada.'),
        '2.9.7':('NO APLICA','No hay variables categóricas en la matriz del SVR; modelos separados por activo.'),
        '2.9.9':('NO APLICA','Características cíclicas son un ejemplo de ingeniería, no una obligación de incorporarlas; la exclusión se documenta.'),
        '2.9.12':('NO APLICA','No se incorpora calendario al SVR fijado; no se seleccionan variables adicionales con TEST.'),
        '3.3':('NO APLICA','La guía permite Persistence o ingenuo estacional; se implementa Persistence.'),
        '3.16':('NO APLICA','No existen coordenadas ni vecindad geográfica.'),
        '4.7':('NO APLICA','El SVR no alcanza el desempeño alto que activa la sugerencia de cambiar a un dataset más desafiante.'),
        '5.1':('PARCIALMENTE CUMPLIDO','El Jupyter Book local está actualizado; falta confirmar que la versión final publicada incluya estas correcciones y que el profesor pueda acceder a ella.'),
        '5.5':('PARCIALMENTE CUMPLIDO','Se diagnostican errores y curva de aprendizaje; no se demuestra ausencia universal de sobreajuste. Mantener las limitaciones y no presentar validación usada para selección como evaluación independiente.'),
        '5.10':('PARCIALMENTE CUMPLIDO','Scripts, versiones, semillas, huellas y notebooks ejecutados disponibles. Falta una reconstrucción integral en entorno limpio y verificar distribución de datos y enlaces públicos.'),
    }
    for k in ['2.1.1','2.1.2','2.1.3','2.1.12','2.3.7','2.3.8','2.3.9','2.3.10']:
        exceptions[k]=('NO APLICA','Objetivo continuo, una única categoría original (symbol) y sin coordenadas; no se crean clases o variables artificiales para aplicar esta técnica.')
    for k in ['2.4.4','2.4.5','2.4.6','2.4.8']:
        exceptions[k]=('NO APLICA','Técnica alternativa o condicional: PCA e Isolation Forest cubren los diagnósticos elegidos; se documentan subpoblaciones conocidas por activo y periodo.')
    changes_done=(OUT/'change_events_metadata.json').exists()
    clean_path=OUT/'clean_reproduction_metadata.json'
    clean=json.loads(clean_path.read_text(encoding='utf-8')) if clean_path.exists() else {}
    clean_passed=clean.get('status')=='passed'
    if changes_done:
        for key in ['2.6.27','2.6.28','2.6.29']:
            exceptions[key]=('CUMPLIDO','Sección 2.6.11: cambio de media candidato, bootstrap por bloques, Holm, sensibilidad e incertidumbre condicional; tres eventos con fuentes primarias. No se afirma causalidad ni certeza de ruptura.')
    if clean_passed:
        exceptions['5.10']=('CUMPLIDO','Reejecución desde DEVELOPMENT en venv nuevo y carpeta aislada sin TEST: 17 etapas, incluidos los 170 ajustes; 13 tablas contrastadas con tolerancias declaradas. Lock de dependencias y metadatos disponibles. La publicación se verifica por separado.')
    access_path=OUT/'publication_access_audit.json'
    access=json.loads(access_path.read_text(encoding='utf-8')) if access_path.exists() else {}
    if access.get('http_status')==200:
        exceptions['5.1']=('PARCIALMENTE CUMPLIDO','El enlace público responde HTTP 200, pero las correcciones son locales y no se ha verificado su presencia en la versión publicada. Publicar y comprobar contenidos y descargas antes de entregar.')
    published = access.get('content_version_verified') is True and access.get('corrected_version_published') is True
    if published:
        exceptions['5.1']=('CUMPLIDO','Book corregido publicado; se verificaron contenido y descargas de la versión registrada en publication_access_audit.json. La disponibilidad futura del servidor no está garantizada.')
    for row in rows:
        key=row['requisito'].split(' ',1)[0]
        state='CUMPLIDO'
        note='Evidencia documental y resultados disponibles; cumplimiento del requisito, no garantía universal de generalización.'
        if key.startswith('1.'):
            reference='Sección '+base_sections[int(key.split('.')[1])]
            file='book/sections/01_base_datos.md'
        elif key.startswith('2.') and len(key.split('.'))>=3:
            reference='Sección '+'.'.join(key.split('.')[:2])
            file='book/sections/02_eda.md'
            if key.startswith('2.5.'):
                reference+='; 179 variables, 4.475 evaluaciones, disponibilidad por hora y fronteras estrictas'
            if key.startswith('2.6.'):
                reference+='; precio y objetivo horario, tramos continuos y límites de inferencia'
        elif key in ['2.7','2.8']:
            reference='Sección '+key
            file='book/sections/02_eda.md'
            state,note='NO APLICA','No hay coordenadas geográficas; justificación explícita.'
        elif key.startswith(('3.','4.')):
            reference='Sección 3; cinco folds, dos modelos, 50 diagnósticos residuales, bootstrap por bloques'
            file='book/sections/03_modelo_base.md'
        elif key.startswith('5.'):
            reference='Secciones 1.7–1.8, 2.9 y 3.10–3.11; dependencias, scripts y resultados locales'
            file='book/sections/03_modelo_base.md'
        else:
            reference='Protocolo común, objetivo, escala, información y fronteras'
            file='book/entregable1_master.md'
        if key in exceptions:
            state,note=exceptions[key]
        assert (ROOT/file).is_file()
        row.update(estado=state,evidencia=reference,archivo_resultado=file,observaciones=note)
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    counts=Counter(r['estado'] for r in rows)
    text='# 4. Auditoría de cierre y reproducibilidad\n\nEsta tabla coteja los requisitos con la versión local corregida. Distingue ejecución, justificación de no aplicabilidad y evidencia pendiente. No se asigna una nota ni se presenta como evaluación independiente del profesor. TEST permanece reservado.\n\n'
    text+='; '.join(f'**{k}: {v}**' for k,v in sorted(counts.items()))+'.\n\n'
    text+='## Prioridades antes de entregar\n\n1. Publicar la versión final corregida y verificar acceso al Book, datos y notebooks. La construcción local no publica cambios.\n2. Mantener explícitos los límites del mecanismo de faltantes, concept drift, localización de cambios y selección sobre los mismos folds. No usar TEST para resolver decisiones de desarrollo.\n\n'
    if not clean_passed:
        text+='La reproducción en entorno limpio todavía no está validada; consultar el registro de ejecución antes de declararla completada.\n\n'
    text+='## Reproducción local de estas correcciones\n\nDesde la raíz del proyecto, con Python 3.10 y las versiones de `requirements.txt`, ejecutar los comandos en este orden. Se requiere el snapshot local de DEVELOPMENT y las tablas de auditoría originales. Para repetir todo el entrenamiento del modelo, añadir primero `python src/18_base_model.py`; esto no evalúa TEST.\n\n```text\npython -m unittest discover -s tests -v\npython src/09_univariate_development.py\npython src/09_render_univariate_report.py\npython src/13_feature_diagnostics.py\npython src/13_render_feature_diagnostics.py\npython src/14_temporal_target.py\npython src/14_render_target_temporal.py\npython src/17_validate_pipeline.py\npython src/19_extended_diagnostics.py\npython src/18_render_base_report.py\npython src/20_sync_audit.py\n```\n\nLos notebooks permiten repetir sus cálculos mediante `RECALCULAR` o `REENTRENAR`; por defecto verifican y presentan resultados guardados. Los generadores del informe actualizan sus respectivas secciones, por lo que deben ejecutarse secuencialmente. Ejecutar el sincronizador del checklist al final evita mezclar estados de versiones anteriores.\n\nPara construir el sitio local, entrar en `book` y ejecutar `npm exec --yes --package=mystmd@1.11.0 -- myst build --html`. Esta orden fija MyST 1.11.0; se verificó con Node 24.19.0 y requiere Node/npm. Puede descargar el generador y la plantilla. No se afirma reconstrucción en entorno limpio ni publicación remota.\n\n'
    text+='## Auditoría completa\n\n[Checklist descargable](../../outputs/tables/entregable1_checklist.csv). Los identificadores son los del desglose de requisitos, no necesariamente los subtítulos del informe. Una técnica indicada como alternativa se marca NO APLICA cuando se emplea la alternativa admitida.\n\n| Requisito | Estado | Evidencia | Límite o acción requerida |\n|---|---|---|---|\n'
    text=text.replace('python src/20_sync_audit.py\n```','python src/22_change_events.py\npython src/23_render_change_events.py\npython src/20_sync_audit.py\n```')
    text=text.replace('No se afirma reconstrucción en entorno limpio ni publicación remota.', 'La publicación remota se comprueba por separado.')
    if clean_passed:
        position='## Auditoría completa'
        result='## Reproducción independiente ejecutada\n\nSe creó un entorno virtual nuevo, sin paquetes heredados, con Python 3.10.21 y `requirements.txt`. Se copiaron los scripts, protocolos y únicamente DEVELOPMENT a una carpeta nueva. TEST y el archivo maestro no se copiaron. Se ejecutaron 17 etapas, incluidas las pruebas, auditorías de calidad, EDA, 170 ajustes predictivos y diagnósticos nuevos. La descarga original y la partición inicial no se repitieron: la reproducción comienza en el snapshot cuyo SHA-256 se registra.\n\nSe contrastaron 13 tablas numéricas con los resultados originales usando `rtol=1e-5` y `atol=1e-8`; no se exige igualdad binaria entre plataformas o bibliotecas numéricas. [Registro de ejecución y diferencias](../../outputs/tables/clean_reproduction_metadata.json). [Dependencias transitivas fijadas para Windows/Python 3.10](../../requirements-lock-windows-py310.txt).\n\nPara repetir desde la raíz: crear un entorno con `python -m venv .venv-repro`, instalar `requirements-lock-windows-py310.txt` con el pip de ese entorno y ejecutar `.venv-repro/Scripts/python.exe src/24_clean_reproduction.py`. Cada ejecución crea una carpeta distinta bajo outputs y conserva los registros por etapa. No sobrescribe los resultados principales ni accede a TEST. El comando presupone Node disponible para las dos auditorías JavaScript.\n\n'
        text=text.replace(position,result+position,1)
        if clean.get('clean_notebooks'):
            text=text.replace('## Auditoría completa',f"Se ejecutaron además **{len(clean['clean_notebooks'])} notebooks** en la carpeta aislada, forzando el kernel del nuevo entorno y comprobando su ruta. Las celdas de la copia guardada no presentan errores. Para repetir esa comprobación tras el cálculo, ejecutar `.venv-repro/Scripts/python.exe src/25_validate_clean_notebooks.py`.\n\n## Auditoría completa",1)
    if access.get('http_status')==200:
        text=text.replace('## Auditoría completa','El [sitio publicado]('+access['url']+') respondió HTTP 200 en una comprobación de acceso; esto no verifica que contenga las correcciones locales. [Registro de acceso](../../outputs/tables/publication_access_audit.json).\n\n## Auditoría completa',1)
    for r in rows:
        text+='| '+' | '.join(str(r[c]).replace('|','\\|').replace('\n',' ') for c in ['requisito','estado','evidencia','observaciones'])+' |\n'
    if published:
        text=text.replace('1. Publicar la versión final corregida y verificar acceso al Book, datos y notebooks. La construcción local no publica cambios.', '1. Publicación verificada: consultar el registro de acceso y la versión comprobada. Conservar juntos el informe, el snapshot de DEVELOPMENT y los archivos de reproducción.')
        text=text.replace('respondió HTTP 200 en una comprobación de acceso; esto no verifica que contenga las correcciones locales.', 'respondió HTTP 200 y se verificaron contenido corregido y descargas; el registro identifica la versión comprobada.')
    internal=ROOT/'outputs/revision_interna'
    internal.mkdir(exist_ok=True)
    (internal/'04_auditoria.md').write_text(text,encoding='utf-8')
    print(dict(counts))


if __name__=='__main__':
    main()
