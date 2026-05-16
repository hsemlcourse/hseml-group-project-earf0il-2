# ML Project — Классификация цифр SVHN с шумом

**Студент:** Барсток Арсений Алексеевич

**Группа:** БИВ234

## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуск](#запуск)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Воспроизводимость](#воспроизводимость)
7. [Линтинг и тесты](#линтинг-и-тесты)
8. [Деплой](#деплой)
9. [Отчёт](#отчёт)


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
├── deploy                       # Production-сервис (см. раздел «Деплой»)
│   ├── api/                     # FastAPI: main.py, schemas.py, preprocess.py
│   ├── ui/app.py                # Streamlit-фронтенд
│   ├── tests/test_api.py        # Smoke-тесты API
│   ├── Dockerfile.api
│   ├── Dockerfile.ui
│   ├── docker-compose.yml
│   └── requirements.txt
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
| Baseline LogReg       | ~0.14          | ~0.13          | Плоские пиксели, без features |
| KNN (k=5)             | ~0.23          | ~0.16          | Плоские пиксели               |
| Random Forest         | ~0.71          | ~0.68          | Плоские + HOG-фичи            |
| XGBoost               | ~0.74          | ~0.71          | HOG-фичи, PCA
|LightGBM               | ~0.75          | ~0.72          | HOG-фичи, PCA                 |
| Simple CNN            | ~0.84          | ~0.82          | 3 conv-блока                  |
| Deeper CNN            | ~0.93          | ~0.93          | + BatchNorm, Dropout          |
| ResNet-like (final)   | ~0.93    | ~0.93      | + аугментация + denoise       |



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


## Деплой

Production-сервис состоит из двух контейнеров:

- **API** ([`deploy/api/main.py`](deploy/api/main.py:1)) — FastAPI, загружает [`models/resnet_like_final.keras`](models/resnet_like_final.keras), эндпоинты `/health`, `/info`, `POST /predict`.
- **UI** ([`deploy/ui/app.py`](deploy/ui/app.py:1)) — Streamlit: загрузка картинки, визуализация препроцессинга (raw → resize → median × 2 → crop 32×16), bar-chart распределения вероятностей.

**Запуск через docker-compose:**

```bash
docker compose -f deploy/docker-compose.yml up --build
# Streamlit UI:        http://localhost:8501
# FastAPI Swagger UI:  http://localhost:8000/docs
```

**Локальный запуск (без Docker):**

```bash
pip install -r deploy/requirements.txt
# В одном терминале — API:
uvicorn deploy.api.main:app --reload --port 8000
# В другом — UI:
API_URL=http://localhost:8000 streamlit run deploy/ui/app.py
```

**Пример запроса к API:**

```bash
curl -F "file=@some_digit.png" http://localhost:8000/predict
```

```json
{
  "label": "7",
  "label_index": 7,
  "confidence": 0.97,
  "probabilities": {"0": 0.001, "1": 0.002, "...": "...", "7": 0.97, "...": "..."},
  "inference_time_ms": 18.4
}
```

**Тесты API:**

```bash
pytest deploy/tests/test_api.py -v
```

Подробное описание архитектуры — в разделе «7. Деплой» отчёта ([`report/report.md`](report/report.md:121)).


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
