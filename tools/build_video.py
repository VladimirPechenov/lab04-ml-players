"""Export a captioned video walkthrough of actual saved experiment outputs."""

import json
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT = (
    "C:/Windows/Fonts/arial.ttf"
    if Path("C:/Windows/Fonts/arial.ttf").exists()
    else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
)
BOLD = (
    "C:/Windows/Fonts/arialbd.ttf"
    if Path("C:/Windows/Fonts/arialbd.ttf").exists()
    else "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
)


def build():
    a = json.loads((ROOT / "results/metrics.json").read_text(encoding="utf-8"))
    row = next(r for r in a["metrics"] if r["model"] == a["selected"])
    scenes = [
        (
            12,
            "ЛР №4 · Стили игроков и прогноз ухода",
            None,
            [
                "Печенов В. В. · ИВТ-465",
                "1500 игроков · 24 980 сессий · seed=42",
                "Синтетическая телеметрия: демонстрация метода.",
                "Далее: признаки, кластеры, churn, утечка, важность.",
            ],
        ),
        (
            20,
            "Шаг 1 · Данные и временные окна",
            None,
            [
                "Точка прогноза: конец дня 20.",
                "История: дни 0–20. Будущее: дни 21–30.",
                "Уход = нет будущих сессий. Доля ухода: 44,1%.",
                "Скрытая вовлечённость влияет на активность и возврат.",
                "Archetype и churned не подаются модели.",
            ],
        ),
        (
            20,
            "Шаг 2 · 16 признаков на игрока",
            None,
            [
                "Сессии, среднее/суммарное время, активные дни.",
                "Убийства, смерти, K/D, исследование, чаты.",
                "Сессии в активный день, последний день, recency.",
                "Добавлены: std длительности, доля ночных сессий,",
                "тренд длительности и нормированный тренд активности.",
                "Все признаки считаются только по дням 0–20.",
            ],
        ),
        (
            20,
            "Шаг 3 · Выбор числа кластеров",
            "silhouette.png",
            [
                "KMeans на пяти признаках стиля.",
                "StandardScaler перед кластеризацией.",
                "Проверены k от 2 до 7.",
                "Максимум silhouette: 0,691 при k=3.",
                "Archetype не участвует в выборе k.",
            ],
        ),
        (
            20,
            "Шаг 3 · Сегменты игроков на PCA",
            "clusters_pca.png",
            [
                "Бойцы: 493 игрока, kills ≈ 9.",
                "Исследователи: 484, explored ≈ 0,85.",
                "Социальные: 523, chats ≈ 11.",
                "PCA 2D: 90,4% дисперсии; ARI=1,0.",
                "Чёткое разделение ожидаемо для синтетики.",
            ],
        ),
        (
            25,
            "Шаг 4 · Честная оценка ухода",
            "roc.png",
            [
                "По игрокам: train 900 / val 300 / test 300.",
                "Модель выбрана по validation: LR.",
                f"Test ROC-AUC: {row['test_auc']:.4f} ≥ 0,70.",
                "RF: 0,9121; Dummy: 0,5000.",
                "Precision top-20%: 0,9833 (59 из 60).",
                "95% bootstrap CI AUC: [0,882; 0,940].",
            ],
        ),
        (
            20,
            "Шаг 5 · Важность и утечка",
            "feature_importance.png",
            [
                "Permutation importance: 15 перестановок.",
                "Наибольшая важность: avg_len.",
                "Признаки после просмотра test не отбирались.",
                "Утечка last_day из дней 0–30: AUC=1,0.",
                "Это уже кодирует ответ и запрещено для прогноза.",
                "Честные признаки 0–20: AUC=0,9127.",
            ],
        ),
        (
            13,
            "Результат · Повторение и ограничения",
            None,
            [
                "Ноутбук выполнен целиком. CSV, графики и метрики сохранены.",
                "4 теста: границы времени, метки, split, качество и top-20%.",
                "Запуск расчёта: python -m lab04.experiment",
                "Оценки на синтетике не гарантируют качество в реальной игре.",
                "Нужны временная проверка и A/B-тест действий удержания.",
                "github.com/VladimirPechenov/lab04-ml-players",
            ],
        ),
    ]
    out = ROOT / "videos"
    out.mkdir(exist_ok=True)
    (ROOT / ".qa").mkdir(exist_ok=True)
    total = sum(s[0] for s in scenes)
    writer = imageio_ffmpeg.write_frames(
        str(out / "lab04_demo.mp4"),
        (1600, 900),
        fps=1,
        codec="libx264",
        macro_block_size=1,
        output_params=["-r", "15", "-crf", "20", "-movflags", "+faststart"],
    )
    writer.send(None)
    elapsed = 0
    chapters = []
    for duration, title, chart, lines in scenes:
        chapters.append({"start_s": elapsed, "duration_s": duration, "title": title})
        im = Image.new("RGB", (1600, 900), "#f3f6fb")
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, 1600, 116), fill="#183653")
        d.text((50, 34), title, font=ImageFont.truetype(BOLD, 39), fill="white")
        if chart:
            plot = Image.open(ROOT / "results" / chart).convert("RGB")
            plot.thumbnail((965, 660))
            im.paste(plot, (25 + (965 - plot.width) // 2, 145 + (660 - plot.height) // 2))
            x, y, width, fs = 1030, 170, 520, 28
        else:
            x, y, width, fs = 80, 175, 1440, 35
        f = ImageFont.truetype(FONT, fs)
        for line in lines:
            words, current, wrapped = line.split(), "", []
            for word in words:
                candidate = (current + " " + word).strip()
                if d.textlength(candidate, font=f) > width:
                    wrapped.append(current)
                    current = word
                else:
                    current = candidate
            wrapped.append(current)
            for part in wrapped:
                d.text((x, y), part, font=f, fill="#183653")
                y += fs + 10
            y += 17
        for local in range(duration):
            frame = im.copy()
            draw = ImageDraw.Draw(frame)
            second = elapsed + local
            draw.rectangle((0, 855, 1600, 899), fill="#e0e8f0")
            draw.rectangle((0, 850, int(1600 * (second + 1) / total), 855), fill="#278d96")
            draw.text(
                (40, 864),
                f"{second // 60:02d}:{second % 60:02d} / {total // 60:02d}:{total % 60:02d}"
                "    Печенов В. В. · ИВТ-465 · Видеопротокол реальных результатов",
                font=ImageFont.truetype(FONT, 21),
                fill="#183653",
            )
            writer.send(frame.tobytes())
        if chart == "roc.png":
            frame.save(ROOT / ".qa/video-preview.png")
        elapsed += duration
    writer.close()
    (out / "chapters.json").write_text(json.dumps(chapters, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Video: {total} seconds, 1600x900, captions, no audio")


if __name__ == "__main__":
    build()
