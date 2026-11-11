# Лабораторная работа №2: Классификация с TensorFlow
Датасет: Satellite (Satellite Power-System Fault Diagnosis), Hugging Face: andrewcropper/ilp-datasets, папка satellite.

Запуск из корневой папки проекта:

python -m pip install -r requirements.txt

python download_data.py      # скачивает bk.pl и exs.pl в data/satellite/

python train.py              # опция: --use-prev (добавить предыдущий шаг)

Результаты (модель, метрики, графики, матрица ошибок, ошибочные шаги) сохраняются в artifacts/.
