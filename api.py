"""Модуль для загрузки случайного изображения из коллекции The Met."""

import csv
import json
import os
import random

import requests


def main() -> None:
    """Выбирает случайную картину из CSV и скачивает её через API."""
    paintings = []
    with open('MetObjects.csv', mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Classification'] == 'Paintings':
                paintings.append(row)

    target = random.choice(paintings)
    object_id = target['Object ID']

    # Разбиваем длинную строку, чтобы влезть в лимит 90 символов
    api_base = "https://collectionapi.metmuseum.org/public/collection/v1/objects/"
    api_url = f"{api_base}{object_id}"

    response = requests.get(api_url, timeout=10).json()

    img_url = response.get('primaryImage')
    if not img_url:
        print("Изображение не найдено, запустите скрипт снова.")
        return

    os.makedirs('paintings', exist_ok=True)

    with open(os.path.join('paintings', 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(response, f, indent=4)

    img_data = requests.get(img_url, timeout=10).content
    with open(os.path.join('paintings', 'original.jpg'), 'wb') as f:
        f.write(img_data)

    print("Данные успешно скачаны в папку paintings/")


if __name__ == "__main__":
    main()
