"""Build a seven-page Russian report using the saved experiment results."""

import json
from html import escape
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/VladimirPechenov/lab04-ml-players"


def build():
    a = json.loads((ROOT / "results/metrics.json").read_text(encoding="utf-8"))
    profiles = pd.read_csv(ROOT / "results/cluster_profiles.csv")
    selected = next(r for r in a["metrics"] if r["model"] == a["selected"])
    regular = (
        "C:/Windows/Fonts/arial.ttf"
        if Path("C:/Windows/Fonts/arial.ttf").exists()
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    bold = (
        "C:/Windows/Fonts/arialbd.ttf"
        if Path("C:/Windows/Fonts/arialbd.ttf").exists()
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    )
    pdfmetrics.registerFont(TTFont("ArialR", regular))
    pdfmetrics.registerFont(TTFont("ArialB", bold))
    styles = {
        "body": ParagraphStyle("body", fontName="ArialR", fontSize=10.5, leading=15, spaceAfter=9),
        "title": ParagraphStyle("title", fontName="ArialB", fontSize=20, leading=27, spaceAfter=20),
        "h": ParagraphStyle("h", fontName="ArialB", fontSize=14, leading=20, spaceAfter=12),
        "small": ParagraphStyle("small", fontName="ArialR", fontSize=8.5, leading=12, spaceAfter=6),
        "center": ParagraphStyle("center", fontName="ArialR", fontSize=12, leading=19, alignment=TA_CENTER),
    }
    story, markdown = [], []

    def text(s, kind="body"):
        story.append(Paragraph(escape(s), styles[kind]))
        markdown.append(s + "\n")

    def heading(s):
        text(s, "h")
        markdown[-1] = "## " + markdown[-1]

    def picture(name, width=170 * mm):
        from PIL import Image as PILImage

        path = ROOT / "results" / name
        w, h = PILImage.open(path).size
        story.append(Image(str(path), width=width, height=width * h / w))
        markdown.append(f"![{name}](../results/{name})\n")

    def table(rows, widths):
        wrapped = [[Paragraph(escape(str(c)), styles["small"]) for c in row] for row in rows]
        t = Table(wrapped, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eff7")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#345779")),
                    ("LINEBELOW", (0, 1), (-1, -1), 0.2, colors.lightgrey),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 10))
        markdown.append("| " + " | ".join(map(str, rows[0])) + " |")
        markdown.append("|" + "---|" * len(rows[0]))
        for row in rows[1:]:
            markdown.append("| " + " | ".join(map(str, row)) + " |")
        markdown.append("")

    def newpage():
        story.append(PageBreak())

    text("ЛАБОРАТОРНАЯ РАБОТА № 4", "title")
    text("ML-классификация стилей игроков и сегментация", "title")
    text("Дисциплина «Игровой искусственный интеллект»", "center")
    story.append(Spacer(1, 20 * mm))
    text("Студент: Печенов В. В.", "center")
    text("Группа: ИВТ-465", "center")
    text("Стек: Python, pandas, scikit-learn, matplotlib, Jupyter", "center")
    text("Дата выполнения: 17.11.2026", "center")
    story.append(Spacer(1, 25 * mm))
    heading("1. Цель работы")
    text(
        "Построить воспроизводимый анализ поведения игроков: получить признаки из сессий, выделить "
        "интерпретируемые стили и оценить вероятность ухода без использования будущей информации."
    )
    text(
        "Работа выполнена по опубликованному шаблону преподавателя. Отдельные методические указания "
        "и официальный титульный лист не предоставлены. Этот документ является учебным протоколом "
        "объёмом 7 страниц; оформление по отсутствующему кафедральному шаблону не заявляется.",
        "small",
    )
    text(REPO, "small")
    newpage()
    heading("2. Постановка задачи и данные")
    text(
        "Требуется выполненный Jupyter-ноутбук; 10+ признаков на игрока; кластеризация с "
        "обоснованием k, интерпретацией и 2D; churn-классификатор с ROC-AUC не ниже 0,70; "
        "график важности признаков. Дополнительно выполнены baseline, precision в top-20%, "
        "контроль утечки, история коммитов по этапам, видео, тег и Release."
    )
    text(
        f"Использована разрешённая синтетика: {a['players']} игроков, {a['sessions']} сессий, "
        f"фиксированный seed={a['seed']}. Данные не являются телеметрией реального продукта. "
        "Три архетипа задают интенсивность боя, исследования и общения. Непрерывная скрытая "
        "вовлечённость влияет на наблюдаемую активность и вероятность будущего возврата с шумом."
    )
    text(
        "Генератор не хранит вовлечённость как столбец. Результирующая таблица содержит player_id, "
        "session_id, day, hour, length_s, kills, deaths, explored, chats, archetype и churned. "
        "Каждый игрок имеет хотя бы одну сессию в наблюдаемом окне; новые игроки без истории "
        "в данную модель не входят. Дубликатов session_id и пропусков нет."
    )
    table(
        [
            ["Элемент", "Определение"],
            ["Момент прогноза", "После завершения дня 20"],
            ["Признаки", "Только сессии дней 0–20 включительно"],
            ["Метка churn=1", "Нет ни одной сессии в дни 21–30 включительно"],
            ["Доля ухода", f"{a['churn_rate']:.2%} ({round(a['players'] * a['churn_rate'])} игроков)"],
            ["Метаданные", "archetype только для ARI; churned только для сверки меток"],
            ["Разбиение", "По игрокам: train 900 / validation 300 / test 300"],
        ],
        [45 * mm, 125 * mm],
    )
    text(
        "Сначала независимо формируются события окна наблюдения, затем будущие события. Метка "
        "для обучения заново выводится из наличия будущих сессий функцией churn_labels. "
        "Одинаковые player_id не пересекаются между выборками. Split сохранён в metrics.json."
    )
    text(
        "Результаты воспроизводятся командой python -m lab04.experiment; таблицы исходных событий, "
        "признаков и меток сохранены в data/events.csv, results/features.csv и results/labels.csv.",
        "small",
    )
    newpage()
    heading("3. Реализация: признаки и архитектура")
    table(
        [
            ["Признаки", "Расчёт по наблюдаемым сессиям"],
            ["n_sessions, avg_len, total_len", "Количество уникальных сессий; среднее и сумма length_s"],
            ["kills_per_session, deaths_per_session", "Средние kills и deaths"],
            ["kd_ratio", "(sum(kills)+1)/(sum(deaths)+1)"],
            ["explored_mean, chats_per_session", "Средние explored и chats"],
            ["active_days, last_day", "Число уникальных day; максимальный day <= 20"],
            ["sessions_per_active_day", "n_sessions / active_days"],
            ["len_std (собственный)", "Стандартное отклонение длительности, 0 при одной сессии"],
            ["night_share (собственный)", "Доля сессий, начавшихся в часы 0–5"],
            ["length_trend (собственный)", "Наклон регрессии length_s по day; 0 при одном дне"],
            ["activity_trend (собственный)", "(N(11–20)/10 - N(0–10)/11) / (N(0–20)/21)"],
            ["recency (собственный)", "20 - last_day"],
        ],
        [69 * mm, 101 * mm],
    )
    text(
        "Итого 16 числовых признаков. Идентификаторы, метка, архетип и номер кластера в churn-модель "
        "не передаются. Все значения конечны. Тренды не зависят от порядка строк. Временной фильтр "
        "применяется внутри player_features, что защищает вызывающий код от случайного использования будущих дней."
    )
    text(
        "Модули: make_dataset.py генерирует события; features.py агрегирует историю и вычисляет "
        "метки; models.py выполняет кластеризацию и оценку классификаторов; experiment.py сохраняет "
        "артефакты. Ноутбук связывает все пять этапов и показывает реальные выходы выполненных ячеек."
    )
    text(
        "Для сегментации выбраны пять признаков стиля: убийства, смерти, K/D, исследование и чаты. "
        "Так частота визитов не подменяет стиль вовлечённостью. Перед KMeans применяется StandardScaler. "
        "Для churn используются все 16 признаков; scaler LR обучается только на train внутри Pipeline."
    )
    newpage()
    heading("4. Демонстрация сегментации")
    text(
        f"Для k=2..7 выполнен KMeans с n_init=20 и random_state=42. Максимум silhouette "
        f"{max(a['silhouette'].values()):.3f} достигается при k={a['k']}. "
        f"ARI относительно скрытых архетипов: {a['ari']:.3f}. PCA сохраняет "
        f"{sum(a['pca_variance']):.1%} дисперсии в двух компонентах. PCA служит только для показа."
    )
    picture("silhouette.png", 145 * mm)
    picture("clusters_pca.png", 145 * mm)
    text(
        "Оба графика построены на наблюдаемом окне. ARI не используется при выборе k и не входит "
        "в признаки. Высокое совпадение объясняется чётко разделёнными синтетическими стилями.",
        "small",
    )
    newpage()
    heading("5. Интерпретация и прогноз ухода")
    table(
        [["Кластер / стиль", "Игроков", "Kills", "Explore", "Chats"]]
        + [
            [
                f"{int(r.cluster)} / {r['name']}",
                f"{int(r.players)} ({r.share:.1%})",
                f"{r.kills_per_session:.2f}",
                f"{r.explored_mean:.2f}",
                f"{r.chats_per_session:.2f}",
            ]
            for _, r in profiles.iterrows()
        ],
        [61 * mm, 36 * mm, 24 * mm, 25 * mm, 24 * mm],
    )
    text(
        "«Бойцам» подходят боевые испытания; «Исследователям» - новые области и секреты; "
        "«Социальным» - кооперативные события. Названия получены по средним профилям, "
        "не скопированы из archetype. Кластер - описание поведения, а не неизменный тип личности."
    )
    text(
        "DummyClassifier(strategy=prior) задаёт уровень случайного ранжирования. "
        "Baseline: LogisticRegression(C=1, max_iter=2000) со стандартизацией. "
        "Альтернатива: RandomForest с 300 деревьями, min_samples_leaf=5. "
        "Гиперпараметры зафиксированы заранее. Лучшая модель выбирается только по validation AUC."
    )
    table(
        [["Модель", "Val AUC", "Test AUC", "P@20%"]]
        + [
            [r["model"], f"{r['validation_auc']:.4f}", f"{r['test_auc']:.4f}", f"{r['precision_top20']:.4f}"]
            for r in a["metrics"]
        ],
        [65 * mm, 35 * mm, 35 * mm, 35 * mm],
    )
    text(
        f"Выбрана {a['selected']}; test ROC-AUC = {selected['test_auc']:.4f}, "
        f"95% bootstrap-интервал [{a['auc_ci95'][0]:.3f}; {a['auc_ci95'][1]:.3f}] "
        "(500 перевыборок тестовых игроков). Это интервал для фиксированной обученной модели, "
        "он не учитывает вариативность генератора и обучения."
    )
    picture("roc.png", 145 * mm)
    text(
        "Precision@top-20% измеряет долю ушедших среди 60 тестовых игроков с наибольшим риском. "
        "При равных рисках Dummy используется стабильный порядок индексов; его precision зависит от этого порядка.",
        "small",
    )
    newpage()
    heading("6. Важность признаков и контроль утечки")
    picture("feature_importance.png", 160 * mm)
    text(
        "Permutation importance измеряется как снижение test ROC-AUC после перестановки признака "
        "(15 повторов, полосы - стандартное отклонение). Признаки после анализа не отбирались, "
        "модель не переобучалась по test. Отрицательное значение возможно из-за случайности. "
        "Коррелированные avg_len, total_len и n_sessions делят между собой важность."
    )
    top = pd.read_csv(ROOT / "results/feature_importance.csv").sort_values("mean", ascending=False).iloc[0]
    text(
        f"Наибольшее среднее падение AUC у {top.feature}: {top['mean']:.4f}. "
        "Это связь с прогнозом модели, а не причинный эффект: увеличение длительности само по себе "
        "не доказывает снижение ухода. Синтетическая вовлечённость влияет и на поведение, и на возврат."
    )
    table(
        [
            ["Эксперимент на тех же test-игроках", "ROC-AUC"],
            ["Честная модель: признаки только 0–20", f"{selected['test_auc']:.4f}"],
            ["Намеренная утечка: last_day из 0–30", f"{a['leaky_auc']:.4f}"],
        ],
        [130 * mm, 40 * mm],
    )
    text(
        "В эксперименте с утечкой отдельная LR видит максимальный день за весь горизонт: "
        "отсутствие дня >=21 совпадает с определением ухода. Такая идеальная оценка не имеет "
        "практической ценности для прогноза в конце дня 20. Для честной модели максимум ограничен днём 20."
    )
    newpage()
    heading("7. Проверки, выводы и литература")
    text(
        "Проверены: независимость признаков от будущих сессий и метаданных; устойчивость к "
        "перемешиванию строк; корректность меток; конечность признаков для одной сессии; "
        "отсутствие пересечения игроков между train/validation/test; выбор k по silhouette; "
        "ROC-AUC >=0,70 на независимом синтетическом прогоне; корректность precision top-20%. "
        "Линтер и pytest запускаются также в GitHub Actions."
    )
    text(
        "Ноутбук выполнен целиком, сохранены execution_count и выводы всех ячеек. "
        "results/metrics.json содержит seed, параметры окон, версии, разбиение и метрики. "
        "CSV профилей, тестовых прогнозов и permutation importance позволяют независимо проверить таблицы."
    )
    text(
        f"Достигнуты {a['k']} интерпретируемых сегмента и test AUC {selected['test_auc']:.3f}; "
        "порог задания выполнен. По validation выбрана логистическая регрессия, "
        "сложный ансамбль преимущества не показал. Продемонстрирована опасность временной утечки."
    )
    text(
        "Ограничения: одна синтетическая популяция и один основной seed; простая модель вовлечённости; "
        "полностью известный будущий горизонт. Для реальной игры нужны временной holdout, "
        "обработка цензурирования, калибровка рисков и проверка стабильности сегментов. "
        "Эффект предложений удержания проверяется A/B-тестом; прогноз ухода сам по себе не доказывает пользу бонуса."
    )
    heading("Источники")
    for s in [
        "1. Задание и шаблон: github.com/kolesnikov-vstu-gameai/lab04-ml-players (дата обращения: 17.11.2026).",
        "2. Scikit-learn. Clustering, silhouette: scikit-learn.org/stable/modules/clustering.html.",
        "3. Scikit-learn. Data leakage: scikit-learn.org/stable/common_pitfalls.html.",
        "4. Scikit-learn. Permutation importance: scikit-learn.org/stable/modules/permutation_importance.html.",
    ]:
        text(s, "small")
    heading("Приложение. Материалы и повторение")
    text(REPO, "small")
    text(
        "Установка: python -m pip install -r requirements.txt. Расчёт: python -m lab04.experiment. "
        "Ноутбук: jupyter lab notebooks/lab04.ipynb. Проверки: python -m ruff check .; python -m pytest. "
        "Видео и Release связаны из README. Видео хранится вне Git.",
        "small",
    )
    text(
        "Версии основного прогона: Python "
        + a["python"]
        + "; "
        + "; ".join(f"{k} {v}" for k, v in a["versions"].items()),
        "small",
    )

    def footer(canvas, doc):
        canvas.setFont("ArialR", 8)
        canvas.setFillColor(colors.HexColor("#53657b"))
        canvas.drawString(20 * mm, 12 * mm, "Печенов В. В. · ИВТ-465 · ЛР №4")
        canvas.drawRightString(190 * mm, 12 * mm, str(doc.page))

    doc = SimpleDocTemplate(
        str(ROOT / "docs/report.pdf"),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=17 * mm,
        bottomMargin=20 * mm,
        title="ЛР №4. Печенов В. В. ИВТ-465",
        author="Печенов В. В.",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    (ROOT / "docs/report.md").write_text("\n".join(markdown), encoding="utf-8")


if __name__ == "__main__":
    build()
