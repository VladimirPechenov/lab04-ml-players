"""Create the explanatory notebook; execution is a separate reproducible step."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
nb.cells = [
    md("""# ЛР №4. ML-классификация стилей игроков и сегментация
**Печенов В. В., ИВТ-465 · 17.11.2026**

Данные синтетические: это демонстрация метода, не исследование реальных пользователей.
Цель: выделить поведенческие сегменты и предсказать отсутствие возврата в дни 21–30.
Признаки доступны только до конца дня 20. Разбиение по игрокам: 60/20/20.
Archetype, churned и идентификаторы не попадают в матрицу признаков."""),
    code("""from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, display
from lab04.experiment import run
from lab04.features import player_features, churn_labels
ROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent
audit = run(ROOT)
df = pd.read_csv(ROOT / 'data/events.csv')
print('Игроков:', audit['players'], 'Сессий:', len(df), 'Churn:', round(audit['churn_rate'], 3))
display(df.head())"""),
    md("""## 1. Датасет и контроль качества
1500 игроков, три стиля: aggressive, explorer, social. Непрерывная скрытая вовлечённость
влияет на частоту и длительность сессий, а также вероятность будущего возврата.
Она не записывается в таблицу и не подаётся модели. У каждого игрока есть хотя бы одна
сессия в окне наблюдения; нулевые истории вне области этой модели.
Метка вычисляется из событий будущего окна, а не из признака last_day."""),
    code("""assert df.session_id.is_unique
assert df.notna().all().all()
assert df.day.between(0, 30).all()
X = player_features(df)
y = churn_labels(df, X.index)
assert X.shape == (1500, 16)
assert not {'archetype', 'churned', 'player_id', 'cluster'}.intersection(X.columns)
assert X.last_day.le(20).all()
display(df.groupby('archetype').agg(players=('player_id', 'nunique'), sessions=('session_id', 'count')))
display(X.describe().T.round(3))"""),
    md("""## 2. Признаки (16)
Базовые: число сессий; среднее/суммарное время; убийства и смерти за сессию;
сглаженное K/D; исследование; чаты; активные дни; последний день; сессии на активный день.
Добавлены: стандартное отклонение длительности, доля ночных сессий (00:00–05:59),
линейный тренд длительности по дням, нормированный тренд активности (дни 0–10 против 11–20),
recency = 20 - last_day. При одной сессии или одном уникальном дне тренд и std равны 0.
Коррелированные признаки допустимы, но разделяют между собой permutation importance."""),
    md("""## 3. Кластеры и 2D
KMeans обучается на пяти стандартизованных признаках стиля: kills_per_session,
deaths_per_session, kd_ratio, explored_mean, chats_per_session. Активность не включена,
чтобы разделять стиль игры, а не степень вовлечённости. Проверяем k=2..7 и выбираем
максимум silhouette. PCA применяется к тем же данным только для визуализации.
Названия даются по центрам кластеров; archetype используется только для внешней проверки ARI."""),
    code("""print('Выбрано k:', audit['k'], 'ARI:', round(audit['ari'], 4))
print('Доля дисперсии двух PCA-компонент:', sum(audit['pca_variance']))
display(pd.read_csv(ROOT / 'results/cluster_profiles.csv').round(3))
display(pd.read_csv(ROOT / 'results/cluster_vs_archetype.csv'))
display(Image(filename=str(ROOT / 'results/silhouette.png')))
display(Image(filename=str(ROOT / 'results/clusters_pca.png')))"""),
    md("""Интерпретация: «Бойцы» чаще убивают, «Исследователи» имеют наибольшее explored_mean,
«Социальные» чаще общаются. Эти сегменты можно использовать для боевых испытаний,
исследовательских заданий и кооперативных событий соответственно. Высокий ARI ожидаем
на синтетических архетипах; на реальных данных столь чистые границы не гарантированы."""),
    md("""## 4. Прогноз ухода: train / validation / test
Фиксируем seed=42; стратификация по метке на уровне игрока, 900/300/300 человек.
StandardScaler логистической регрессии обучается только на train в Pipeline.
Сравниваем Dummy(prior), LogisticRegression(C=1), RandomForest(300 деревьев,
min_samples_leaf=5). Выбор между LR/RF делается по validation ROC-AUC, затем один раз
оценивается test; параметры не подбираются по test. Модели оставлены обученными на train.
Top-20% — precision среди 60 тестовых игроков с самым высоким риском.
500 bootstrap-перевыборок test дают условный 95% интервал AUC выбранной модели."""),
    code("""metrics = pd.DataFrame(audit['metrics']).set_index('model')
display(metrics.round(4))
print('Выбрана по validation:', audit['selected'])
print('95% bootstrap CI:', audit['auc_ci95'])
selected_auc = metrics.loc[audit['selected'], 'test_auc']
assert selected_auc >= 0.70
display(Image(filename=str(ROOT / 'results/roc.png')))"""),
    md("""### Контроль утечки
Намеренно нарушаем границу времени: last_day из дней 0–30 подаётся отдельной LR.
Тестовые игроки те же. Метка «нет сессий в 21–30» почти дословно кодируется этим
признаком, поэтому AUC близко к 1. Это недопустимая модель для прогноза на день 20.
Честная модель использует last_day только из окна 0–20."""),
    code("""display(pd.DataFrame([
    {'experiment': 'Честная модель (0–20)', 'test_auc': float(selected_auc)},
    {'experiment': 'Утечка last_day (0–30)', 'test_auc': audit['leaky_auc']}
]))
# Независимое подтверждение метки по будущим событиям
assert y.equals(df.groupby('player_id').churned.first().loc[X.index].rename('churned'))"""),
    md("""## 5. Важность признаков
Permutation importance: уменьшение тестового ROC-AUC при перестановке одного признака,
15 повторов; полосы показывают стандартное отклонение. Объясняем уже выбранную модель,
не отбираем по этому графику признаки. Это статистическая ассоциация, не причинный эффект."""),
    code("""display(Image(filename=str(ROOT / 'results/feature_importance.png')))
display(pd.read_csv(ROOT / 'results/feature_importance.csv').sort_values('mean', ascending=False).head(8))
print('Версии:', audit['versions'])"""),
    md("""## Выводы и ограничения
Выполнены сегментация, интерпретация, прогноз ухода и контроль утечки.
Вероятности ухода полезны для ранжирования предложений удержания, но эффект акций
нужно проверять отдельным A/B-тестом. Данные созданы одной моделью генерации и одним seed;
высокие оценки не доказывают качество на реальной игре. Нужны внешняя временная проверка,
мониторинг сдвига распределений, калибровка вероятностей и оценка цены ошибок.

Повтор: `python -m lab04.experiment`; проверки: `python -m pytest`, `python -m ruff check .`.
Источники и полный отчёт: `docs/report.md`, `docs/report.pdf`."""),
]
nb.metadata["kernelspec"] = dict(display_name="Python 3", language="python", name="python3")
nbf.write(nb, ROOT / "notebooks/lab04.ipynb")
