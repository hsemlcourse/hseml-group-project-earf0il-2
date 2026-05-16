# ML Project — Классификация цифр SVHN с шумом

**Студент:** Команда earf0il (HSE ML Group Project)

**Группа:** HSE ML

## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуск](#запуск)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Воспроизводимость](#воспроизводимость)
7. [Линтинг и тесты](#линтинг-и-тесты)
8. [Отчёт](#отчёт)


## Описание задачи

Задача — многоклассовая классификация изображений цифр (0–9) на основе модифицированной версии
датасета [SVHN (Street View House Numbers)](http://ufldl.stanford.edu/housenumbers/).
Изображения имеют размер 32×32×3, на каждом изображении присутствует ровно одна цифра
по центру кадра. К датасету искусственно добавлен шум типа «соль и перец» (salt-and-pepper),
что приближает задачу к реальным условиям съёмки в плохих условиях.

**Задача:** Многоклассовая классификация изображений (10 классов).

**Датасет:** Модифицированный SVHN — 50 000 train / 25 000 test, изображения 32×32×3,
10 сбалансированных классов (цифры 0–9). Источник оригинального SVHN:
http://ufldl.stanford.edu/housenumbers/. Модификация (центрирование одной цифры
и добавление шума) выполнена организаторами курса.

**Целевая метрика:** **Accuracy** — основная метрика (используется в Kaggle-рейтинге, классы
сбалансированы, поэтому точность корректно отражает качество). В качестве вспомогательной
рассматривается **macro F1-score** для оценки качества по каждому классу. При сбалансированных
классах эти метрики хорошо коррелируют, но F1 более устойчива к редким ошибкам на отдельных
цифрах (например, 1 vs 7).


## Структура репозитория
```
.
├── data
│   ├── processed                # Очищенные и обработанные данные
│   └── raw                      # Исходные файлы
├── models                       # Сохранённые модели
├── notebooks
│   ├── 01_eda.ipynb             # Exploratory Data Analysis
│   ├── 02_baseline.ipynb        # Baseline-модели (LogReg, KNN)
│   └── 03_experiments.ipynb     # Эксперименты (RF, XGBoost, CNN, ResNet, ансамбли)
├── presentation                 # Презентация для защиты
├── report
│   └── report.md                # Финальный отчёт
├── src
│   ├── __init__.py
│   ├── utils.py                 # Утилиты: фиксация seed, загрузка данных
│   ├── preprocessing.py         # Предобработка изображений и сплит
│   └── modeling.py              # Определение моделей, обучение, оценка
├── tests
│   └── test.py                  # Тесты пайплайна
├── .flake8                      # Конфигурация линтера
├── .pre-commit-config.yaml      # Хуки pre-commit
├── Makefile                     # Команды для линтинга и тестов
├── requirements.txt
└── README.md
```

## Запуск

```bash
# 1. Клонировать репозиторий
git clone <url>
cd hseml-group-project-earf0il

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 3a. (только macOS) для XGBoost требуется libomp:
#     brew install libomp

# 4. Запустить notebooks по порядку
jupyter notebook notebooks/01_eda.ipynb
```

## Данные

- `data/raw/` — исходные файлы датасета (большие бинарные файлы, не коммитятся в git)
- `data/processed/` — предобработанные данные (после denoise/crop)
- `data/data_train`, `data/data_test`, `data/meta` — pickled Python dict'ы:
  - `data_train`: `images` (50000, 32, 32, 3) float32, `labels` (50000,) uint8, `section` str
  - `data_test`:  `images` (25000, 32, 32, 3) float32, `section` str (без `labels` — Kaggle hold-out)
  - `meta`:       `label_names` — список из 10 строк-классов


## Результаты

| Модель                | Accuracy (val) | Macro F1 (val) | Примечание                    |
|-----------------------|----------------|----------------|-------------------------------|
| Baseline LogReg       | ~0.20          | ~0.20          | Плоские пиксели, без features |
| KNN (k=5)             | ~0.35          | ~0.34          | Плоские пиксели               |
| Random Forest         | ~0.45          | ~0.44          | Плоские + HOG-фичи            |
| XGBoost               | ~0.50          | ~0.50          | HOG-фичи, PCA                 |
| Simple CNN            | ~0.88          | ~0.88          | 3 conv-блока                  |
| Deeper CNN            | ~0.92          | ~0.92          | + BatchNorm, Dropout          |
| ResNet-like (final)   | **~0.94**      | **~0.94**      | + аугментация + denoise       |

Финальный результат на Kaggle: 0.93973 (public) / 0.94211 (private).


## Воспроизводимость

Во всех экспериментах используется фиксированный seed `SEED=42`, который применяется
для `random`, `numpy`, `tensorflow` и `sklearn` через [`src.utils.set_global_seed()`](src/utils.py:1).


## Линтинг и тесты

```bash
make lint          # запустить flake8
make test          # запустить pytest
make all           # lint + test

pre-commit install # установить pre-commit хуки
pre-commit run --all-files
```


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
