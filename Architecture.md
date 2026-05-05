Ниже — итоговая структура рабочего пространства, которая у тебя получается для ВКР по предиктивному мониторингу аномалий.

```text
ChatGPT / Codex
        ↓
GitHub repository
        ↓
GitHub Actions
        ↓
Kaggle API Token
        ↓
Kaggle Notebook
        ↓
Kaggle Dataset
        ↓
Kaggle Output
```

---

# 1. Общая логика рабочего пространства

У тебя будет один основной GitHub-репозиторий:

```text
vkr-anomaly-monitoring
```

Он хранит:

```text
код проекта;
notebook для Kaggle;
настройки Kaggle;
workflow для автоматического запуска;
configs;
README;
requirements.
```

Но он **не хранит**:

```text
Kaggle API tokens;
датасеты;
обработанные большие CSV/Parquet;
модели;
результаты экспериментов.
```

Данные и результаты находятся в Kaggle:

```text
/kaggle/input     — входные датасеты
/kaggle/working   — результаты выполнения
```

---

# 2. Итоговая структура репозитория

```text
vkr-anomaly-monitoring/
│
├── .github/
│   └── workflows/
│       └── run_kaggle.yml
│
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── aggregate_system_state.py
│   │   ├── create_anomaly_labels.py
│   │   └── create_predictive_dataset.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_classifier.py
│   │   └── train_regressor.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate_classifier.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── paths.py
│       └── io.py
│
├── kaggle_notebook/
│   ├── notebook.ipynb
│   ├── kernel-metadata.template.json
│   └── kernel-metadata.json
│
├── configs/
│   └── config.yaml
│
├── reports/
│   └── README.md
│
├── requirements.txt
├── README.md
└── .gitignore
```

```
vkr-anomaly-monitoring/
│
├── .github/                              # CI/CD: автоматизация через GitHub Actions
│   └── workflows/
│       └── run_kaggle.yml                # запуск Kaggle Notebook (выбор аккаунта, push, ожидание результата)
│
├── src/                                  # основной код проекта (логика ВКР)
│   ├── preprocessing/                    # подготовка данных
│   │   ├── __init__.py
│   │   ├── aggregate_system_state.py     # агрегирует метрики (latency, CPU, memory, network)
│   │   ├── create_anomaly_labels.py      # создаёт аномалии (latency/cpu/memory/system)
│   │   └── create_predictive_dataset.py  # формирует лаги, фичи и target (t+1)
│   │
│   ├── training/                         # обучение моделей
│   │   ├── __init__.py
│   │   ├── train_classifier.py           # модель для target_anomaly_t_plus_1 (predict_proba)
│   │   └── train_regressor.py            # прогноз численных метрик (например p95_latency)
│   │
│   ├── evaluation/                       # оценка моделей
│   │   ├── __init__.py
│   │   └── evaluate_classifier.py        # метрики, ROC, confusion matrix, feature importance
│   │
│   └── utils/                            # вспомогательные функции
│       ├── __init__.py
│       ├── paths.py                      # все пути (/kaggle/input, /kaggle/working)
│       └── io.py                         # чтение/сохранение файлов (csv, parquet, json)
│
├── kaggle_notebook/                      # всё, что отправляется на Kaggle
│   ├── notebook.ipynb                   # точка запуска pipeline (импорт src и выполнение)
│   ├── kernel-metadata.template.json    # шаблон настроек Kaggle (GPU, internet, datasets)
│   └── kernel-metadata.json             # генерируется автоматически (НЕ редактировать вручную)
│
├── configs/                              # конфигурации экспериментов
│   └── config.yaml                      # параметры: лаги, horizon, thresholds, split и т.д.
│
├── reports/                              # материалы для ВКР
│   └── README.md                        # ссылки на эксперименты, выводы, описание результатов
│
├── requirements.txt                      # зависимости (устанавливаются в Kaggle)
├── README.md                             # описание проекта и pipeline
└── .gitignore                            # исключает токены, данные и артефакты из Git
```

---

# 3. Папка `.github/workflows`

```text
.github/
└── workflows/
    └── run_kaggle.yml
```

Это папка для GitHub Actions.

Файл:

```text
run_kaggle.yml
```

отвечает за автоматический запуск Kaggle Notebook.

Он делает:

```text
1. забирает код из GitHub;
2. выбирает Kaggle-аккаунт: lil / racon / vic;
3. берёт нужный KGAT-токен из GitHub Secrets;
4. создаёт ~/.kaggle/access_token;
5. копирует src/, configs/, requirements.txt в kaggle_notebook/;
6. создаёт kernel-metadata.json из шаблона;
7. отправляет notebook на Kaggle;
8. ждёт результат выполнения notebook;
9. если Kaggle упал — GitHub Actions тоже падает.
```

Главная задача этого файла:

```text
связать GitHub и Kaggle
```

То есть GitHub становится центром управления, а Kaggle — средой выполнения.

---

# 4. GitHub Secrets и Variables

У тебя несколько Kaggle-аккаунтов.

Сейчас реальные secrets:

```text
KAGGLE_API_TOKEN_LIL
KAGGLE_API_TOKEN_RACON
KAGGLE_API_TOKEN_VIC
```

Это секретные токены формата:

```text
KGAT_...
```

Их нельзя:

```text
показывать;
коммитить;
вставлять в код;
публиковать в README;
отправлять в чат.
```

Для username лучше использовать Repository variables:

```text
KAGGLE_USERNAME_LIL
KAGGLE_USERNAME_RACON
KAGGLE_USERNAME_VIC
```

Получается:

```text
lil   → KAGGLE_USERNAME_LIL   + KAGGLE_API_TOKEN_LIL
racon → KAGGLE_USERNAME_RACON + KAGGLE_API_TOKEN_RACON
vic   → KAGGLE_USERNAME_VIC   + KAGGLE_API_TOKEN_VIC
```

В workflow ты выбираешь:

```text
kaggle_account = lil / racon / vic
```

И GitHub Actions подставляет нужный токен.

---

# 5. Папка `src`

```text
src/
```

Это основная папка с кодом ВКР.

Она не должна зависеть от Kaggle напрямую настолько, чтобы код нельзя было тестировать локально. Но пути под Kaggle можно хранить в `utils/paths.py`.

---

# 6. Папка `src/preprocessing`

```text
src/preprocessing/
├── __init__.py
├── aggregate_system_state.py
├── create_anomaly_labels.py
└── create_predictive_dataset.py
```

Эта папка отвечает за подготовку данных.

## 6.1. `aggregate_system_state.py`

Назначение:

```text
преобразовать исходные CSV из микросервисного датасета в агрегированное состояние системы.
```

Он должен делать:

```text
читать CSV;
находить latency columns;
считать p95_latency;
считать mean_latency;
считать max_latency;
считать std_latency;
находить CPU columns;
считать cpu_total, cpu_mean, cpu_max;
находить memory columns;
считать memory_total, memory_mean, memory_max;
находить network columns;
считать net_rx_total, net_tx_total;
сохранять агрегированные признаки.
```

Смысл для ВКР:

```text
из множества метрик микросервисов получить компактное состояние системы во времени.
```

---

## 6.2. `create_anomaly_labels.py`

Назначение:

```text
создать метки аномалий.
```

Он должен делать:

```text
считать thresholds только по train;
создать latency_anomaly;
создать cpu_anomaly;
создать memory_anomaly;
создать system_anomaly;
не использовать val/test для расчёта порогов.
```

Это критично для ВКР, потому что иначе будет утечка данных.

Правильно:

```text
train → считаем threshold
train/val/test → применяем threshold
```

Неправильно:

```text
весь dataset → считаем threshold
```

---

## 6.3. `create_predictive_dataset.py`

Назначение:

```text
создать датасет для предиктивного мониторинга.
```

Он должен делать:

```text
создать lag features;
создать rolling features;
создать diff features;
создать pct_change features;
создать target_anomaly_t_plus_1 через shift(-1);
создать target_p95_latency_t_plus_1 через shift(-1);
удалить NaN/Inf;
разделить train/val/test по времени без shuffle.
```

Главный target:

```text
target_anomaly_t_plus_1
```

То есть модель должна предсказывать:

```text
будет ли аномалия в следующий момент времени
```

а не просто определять текущую аномалию.

---

# 7. Папка `src/training`

```text
src/training/
├── __init__.py
├── train_classifier.py
└── train_regressor.py
```

Эта папка отвечает за обучение моделей.

## 7.1. `train_classifier.py`

Назначение:

```text
обучить модель классификации для предсказания будущей аномалии.
```

Основная цель:

```text
target_anomaly_t_plus_1
```

Модель должна уметь:

```text
fit;
predict;
predict_proba.
```

Сохраняемые результаты:

```text
model.joblib
metrics.json
predictions.parquet
predict_proba.parquet
```

---

## 7.2. `train_regressor.py`

Назначение:

```text
обучить регрессионную модель для прогноза значения метрики.
```

Например:

```text
target_p95_latency_t_plus_1
```

Это нужно, если в ВКР ты хочешь показать не только бинарное предсказание аномалии, но и прогноз будущей нагрузки/латентности.

---

# 8. Папка `src/evaluation`

```text
src/evaluation/
├── __init__.py
└── evaluate_classifier.py
```

Назначение:

```text
оценить качество модели.
```

Файл должен считать:

```text
accuracy;
precision;
recall;
f1;
roc_auc;
confusion matrix;
ROC curve;
feature importance.
```

И сохранять:

```text
metrics.json
confusion_matrix.png
roc_curve.png
feature_importance.csv
```

Для ВКР это важно, потому что тебе нужны таблицы и графики для анализа результатов.

---

# 9. Папка `src/utils`

```text
src/utils/
├── __init__.py
├── paths.py
└── io.py
```

Это вспомогательные функции.

## 9.1. `paths.py`

Назначение:

```text
централизованно задавать пути.
```

Например:

```text
/kaggle/input/microservices-bottleneck-detection-dataset
/kaggle/working/vkr_outputs
```

Идея:

```text
пути не должны быть раскиданы по всему проекту.
```

---

## 9.2. `io.py`

Назначение:

```text
чтение и сохранение файлов.
```

Например:

```text
безопасное чтение CSV;
сохранение parquet;
сохранение json;
создание папок;
логирование размеров DataFrame.
```

---

# 10. Папка `kaggle_notebook`

```text
kaggle_notebook/
├── notebook.ipynb
├── kernel-metadata.template.json
└── kernel-metadata.json
```

Это папка, которую GitHub Actions отправляет на Kaggle.

---

## 10.1. `notebook.ipynb`

Это главный Kaggle Notebook.

Он должен быть не местом, где лежит весь код, а **точкой запуска pipeline**.

Правильно:

```text
notebook.ipynb импортирует код из src
и запускает функции.
```

Неправильно:

```text
весь проект написан прямо в notebook.
```

Пример логики notebook:

```text
1. настроить пути;
2. вывести доступные /kaggle/input datasets;
3. установить requirements.txt;
4. импортировать preprocessing;
5. запустить preprocessing;
6. запустить training;
7. запустить evaluation;
8. сохранить результаты в /kaggle/working/vkr_outputs.
```

В notebook обязательно должен быть Python kernel metadata, иначе Kaggle выдаёт ошибку:

```text
No kernel name found in notebook
```

---

## 10.2. `kernel-metadata.template.json`

Это шаблон настроек Kaggle Notebook.

Он говорит Kaggle:

```text
как назвать notebook;
какой файл запускать;
включить ли интернет;
включить ли GPU;
какие datasets подключить.
```

Твой файл должен быть примерно такой:

```json
{
  "id": "__KAGGLE_USERNAME__/vkr-anomaly-monitoring",
  "title": "VKR Anomaly Monitoring",
  "code_file": "notebook.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [
    "gagansomashekar/microservices-bottleneck-detection-dataset"
  ]
}
```

Здесь:

```text
__KAGGLE_USERNAME__
```

заменяется в GitHub Actions на username выбранного Kaggle-аккаунта.

---

## 10.3. `kernel-metadata.json`

Это настоящий metadata-файл, который создаётся автоматически из шаблона.

Например:

```json
{
  "id": "racon_username/vkr-anomaly-monitoring",
  "title": "VKR Anomaly Monitoring",
  "code_file": "notebook.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [
    "gagansomashekar/microservices-bottleneck-detection-dataset"
  ]
}
```

Его можно не хранить в GitHub, потому что он генерируется автоматически.

Лучше добавить в `.gitignore`:

```gitignore
kaggle_notebook/kernel-metadata.json
```

---

# 11. Папка `configs`

```text
configs/
└── config.yaml
```

Здесь лежат настройки экспериментов.

Пример:

```yaml
dataset:
  dataset_root: "/kaggle/input/microservices-bottleneck-detection-dataset"
  input_subdir: "processed_dataset"

output:
  output_root: "/kaggle/working/vkr_outputs"

preprocessing:
  n_lags: 3
  forecast_horizon: 1
  rolling_window: 3
  anomaly_quantile: 0.95

target:
  classification_target: "target_anomaly_t_plus_1"
  regression_target: "target_p95_latency_t_plus_1"

split:
  train_frac: 0.7
  val_frac: 0.15
  test_frac: 0.15
  shuffle: false

training:
  random_state: 42
  save_model: true
  save_predictions: true
```

Смысл:

```text
если надо поменять horizon, n_lags или threshold — меняешь config, а не код.
```

---

# 12. Папка `reports`

```text
reports/
└── README.md
```

Эта папка нужна для материалов ВКР:

```text
описание экспериментов;
ссылки на Kaggle versions;
таблицы результатов;
выводы;
графики для вставки в текст ВКР.
```

Но большие файлы сюда лучше не класть.

Лучше хранить в `reports/README.md` ссылки:

```text
Experiment 1 — Kaggle version link
Experiment 2 — Kaggle version link
Final run — Kaggle version link
```

---

# 13. `requirements.txt`

```text
requirements.txt
```

Список зависимостей:

```text
numpy
pandas
scikit-learn
matplotlib
pyarrow
joblib
tqdm
pyyaml
```

Если появятся дополнительные модели:

```text
lightgbm
xgboost
```

Notebook на Kaggle сможет установить зависимости, потому что в metadata включено:

```json
"enable_internet": true
```

---

# 14. `README.md`

```text
README.md
```

Это главный файл описания проекта.

Он должен содержать:

```text
название ВКР;
цель проекта;
структуру репозитория;
описание pipeline;
как запустить GitHub Actions;
какие Kaggle accounts используются;
куда сохраняются результаты;
какой target используется.
```

Минимально:

```text
Цель:
разработка системы предиктивного мониторинга аномалий производительности в микросервисных архитектурах.

Основной target:
target_anomaly_t_plus_1

Данные:
Kaggle dataset gagansomashekar/microservices-bottleneck-detection-dataset

Результаты:
сохраняются в /kaggle/working/vkr_outputs
```

---

# 15. `.gitignore`

```text
.gitignore
```

Должен запрещать попадание секретов и больших файлов в GitHub:

```gitignore
__pycache__/
*.pyc
.ipynb_checkpoints/

.env
kaggle.json
access_token
.kaggle/
kaggle_notebook/kernel-metadata.json

outputs/
data/
*.pkl
*.joblib
*.parquet
*.csv

.vscode/
.idea/
.DS_Store
```

---

# 16. Что находится в Kaggle

После запуска notebook в Kaggle будет структура:

```text
/kaggle/input/
├── microservices-bottleneck-detection-dataset/
└── другие подключённые datasets
```

И результаты:

```text
/kaggle/working/vkr_outputs/
├── processed_dataset/
├── training_results/
│   ├── model.joblib
│   ├── metrics.json
│   ├── predictions.parquet
│   ├── predict_proba.parquet
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.csv
└── run_info.json
```

---

# 17. Логическое объединение датасетов

Если потом у тебя будет несколько Kaggle datasets:

```text
raccoonek/predictive-batch-001
raccoonek/predictive-batch-002
raccoonek/predictive-batch-003
```

ты добавишь их в:

```text
dataset_sources
```

Например:

```json
"dataset_sources": [
  "gagansomashekar/microservices-bottleneck-detection-dataset",
  "raccoonek/predictive-batch-001",
  "raccoonek/predictive-batch-002",
  "raccoonek/predictive-batch-003"
]
```

В Kaggle они появятся как отдельные папки:

```text
/kaggle/input/microservices-bottleneck-detection-dataset
/kaggle/input/predictive-batch-001
/kaggle/input/predictive-batch-002
/kaggle/input/predictive-batch-003
```

А код будет объединять их логически:

```text
не склеивать физически заранее;
а читать список источников и собирать общий manifest / общий DataFrame.
```

---

# 18. Как выглядит рабочий процесс

## 18.1. Ты ставишь задачу агенту

Например:

```text
Исправь src/preprocessing/create_predictive_dataset.py.
Нужно, чтобы target_anomaly_t_plus_1 создавался через shift(-1).
Не используй future leakage.
Создай Pull Request.
```

---

## 18.2. Агент создаёт PR

Он меняет файлы в GitHub:

```text
src/preprocessing/create_predictive_dataset.py
kaggle_notebook/notebook.ipynb
configs/config.yaml
```

---

## 18.3. Ты проверяешь diff

Смотришь:

```text
что изменено;
не сломана ли структура;
не появились ли токены;
не добавлены ли большие файлы.
```

---

## 18.4. Merge в main

После проверки:

```text
Merge pull request
```

---

## 18.5. Запускаешь GitHub Actions

```text
Actions
↓
Run Kaggle Notebook
↓
Run workflow
```

Выбираешь:

```text
kaggle_account = racon / lil / vic
notebook_slug = vkr-anomaly-monitoring
run_note = preprocessing test
```

---

## 18.6. GitHub Actions отправляет notebook на Kaggle

GitHub Actions:

```text
создаёт access_token;
генерирует metadata;
копирует src;
делает kaggle kernels push;
ждёт статус выполнения.
```

---

## 18.7. Kaggle выполняет notebook

Kaggle:

```text
создаёт новую version;
подключает datasets;
включает GPU;
включает internet;
запускает notebook;
сохраняет output.
```

---

## 18.8. Если Kaggle упал

Теперь GitHub Actions тоже должен стать:

```text
failed
```

Это правильно.

---

# 19. Коммиты в этой структуре

Коммиты надо делать после каждого логического блока.

Примеры:

```text
Initialize project structure
Add gitignore for secrets and generated files
Add project dependencies
Add Kaggle kernel metadata template with GPU and internet
Add initial Kaggle notebook
Add GitHub Actions workflow for Kaggle notebook runs
Fail workflow when Kaggle notebook run fails
Add Kaggle path utilities
Add default experiment configuration
Add initial preprocessing entry point
Connect Kaggle notebook to preprocessing entry point
Add system state aggregation logic
Add anomaly label generation
Implement predictive dataset creation
Add classifier training pipeline
Add classifier evaluation utilities
Save model metrics and predictions in Kaggle output
```

---

# 20. Самая важная итоговая структура

```text
GitHub:
- код;
- workflow;
- configs;
- notebook-запускатель;
- metadata template.

GitHub Secrets:
- KGAT-токены Kaggle.

GitHub Variables:
- usernames Kaggle.

Kaggle:
- datasets;
- notebook versions;
- logs;
- outputs;
- saved models;
- metrics.

ChatGPT / Codex:
- ставит задачи;
- правит файлы;
- создаёт PR;
- помогает анализировать ошибки.
```

Итог:

```text
репозиторий = центр разработки;
Kaggle = вычислительная среда;
GitHub Actions = мост между ними;
Codex = агент, который правит код;
Kaggle Output = место хранения результатов экспериментов.
```