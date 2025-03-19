import os
import re
import time
import threading
import datetime
import logging
from typing import List, Dict, Any

import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import telebot
from telebot import types
from transformers import pipeline

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка API-токена для Telegram из переменной окружения
API_TELEGRAM_TOKEN = 'YOUR_API_TELEGRAM_TOKEN'

# Список кандидатов-эмоций (включая "сарказм")
CANDIDATE_LABELS = ["агрессия", "тревожность", "сарказм", "позитив", "нейтральное состояние"]

# Инициализация zero-shot классификатора
classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")

BASE_URL = "https://lenta.ru"
# Глобальный словарь для хранения новостного контекста по chat_id
user_context: Dict[int, List[Dict[str, Any]]] = {}


def format_news_headline(headline: str) -> str:
    """
    Форматирует заголовок новости, выделяя основное сообщение, время и дату, если они присутствуют.
    """
    pattern = r"^(.*?)(\d{2}:\d{2}),\s*([\d]{1,2}\s+\S+\s+\d{4})(.*)$"
    match = re.match(pattern, headline)
    if match:
        main_text = match.group(1).strip()
        time_str = match.group(2)
        date_str = match.group(3)
        return f"{main_text}, {time_str}, {date_str}"
    return headline


def get_first_sentence(url: str) -> str:
    """
    Извлекает первое предложение статьи по заданному URL.
    """
    try:
        response = requests.get(url)
        response.encoding = "utf-8"
        html = response.text
    except Exception as e:
        logging.error(f"Ошибка при получении статьи {url}: {e}")
        return ""

    soup = BeautifulSoup(html, 'html.parser')
    article_body = soup.find('div', class_='topic-body__content')
    text = ""
    if article_body:
        p = article_body.find('p')
        if p:
            text = p.get_text(strip=True)
    if not text:
        p = soup.find('p')
        if p:
            text = p.get_text(strip=True)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return sentences[0] if sentences else ""


def get_news_for_day(date_obj: datetime.date) -> List[Dict[str, Any]]:
    """
    Возвращает список новостей за указанный день.
    """
    url = f"{BASE_URL}/news/{date_obj.year}/{date_obj.month:02}/{date_obj.day:02}/"
    news_items = []
    try:
        response = requests.get(url)
        response.encoding = "utf-8"
        html = response.text
    except Exception as e:
        logging.error(f"Ошибка при получении {url}: {e}")
        return []

    soup = BeautifulSoup(html, 'html.parser')
    blacklisted_headlines = {
        "узнать больше", "подробнее", "пресс-релизы", "техподдержка", "спецпроекты",
        "условия использования", "политика конфиденциальности", "правила применения рекомендательных технологий",
        "условиями акции lenta.ru", "политикой конфиденциальности rambler id", "бывший ссср",
        "силовые структуры", "наука и техника", "интернет и СМИ", "путешествия",
        "среда обитания", "забота о себе"
    }
    blacklisted_url_parts = ["mailto:", "/parts/", "/specprojects/", "/info/", "help.rambler.ru", "/rubrics/"]

    for a in soup.find_all('a', href=True):
        raw_headline = a.get_text(strip=True)
        headline = format_news_headline(raw_headline)
        if not headline or len(headline) <= 10 or headline.lower() in blacklisted_headlines:
            continue

        href = a['href']
        if href.startswith('/'):
            full_url = BASE_URL + href
        elif href.startswith('http'):
            full_url = href
        else:
            full_url = BASE_URL + '/' + href

        if any(bad in full_url for bad in blacklisted_url_parts):
            continue

        news_items.append({'headline': headline, 'url': full_url})

    unique_news = {item['url']: item for item in news_items}.values()
    return list(unique_news)


def get_news_for_date_range(start_date: datetime.date, end_date: datetime.date) -> Dict[datetime.date, List[Dict[str, Any]]]:
    """
    Возвращает словарь, где ключ – дата, а значение – список новостей за этот день.
    """
    news_by_day = {}
    current_date = start_date
    delta = datetime.timedelta(days=1)
    while current_date <= end_date:
        news_by_day[current_date] = get_news_for_day(current_date)
        current_date += delta
    return news_by_day


def generate_emotion_graph(news_items: List[Dict[str, Any]]) -> str:
    """
    Генерирует график распределения эмоций на основе новостей и сохраняет его в файл.
    """
    emotion_counts = {label: 0 for label in CANDIDATE_LABELS}
    for news in news_items:
        emotion = news.get('predicted_emotion')
        if emotion in emotion_counts:
            emotion_counts[emotion] += 1

    plt.figure(figsize=(8, 6))
    plt.bar(emotion_counts.keys(), emotion_counts.values(), color='skyblue')
    plt.xlabel("Эмоция")
    plt.ylabel("Количество новостей")
    plt.title("Распределение эмоций новостей")
    filename = "emotion_graph.png"
    plt.savefig(filename)
    plt.close()
    return filename


def format_progress(percent: int) -> str:
    """
    Форматирует строку прогресса, используя 10 эмодзи.
    """
    total_blocks = 10
    filled_blocks = int((percent / 100) * total_blocks)
    empty_blocks = total_blocks - filled_blocks
    return "🔵" * filled_blocks + "⚪" * empty_blocks + f" {percent}%"


# Инициализация бота Telegram
bot = telebot.TeleBot(API_TOKEN, parse_mode="HTML")


@bot.message_handler(commands=['start'])
def send_welcome(message: types.Message) -> None:
    """
    Обрабатывает команду /start и отправляет приветственное сообщение.
    """
    chat_id = message.chat.id
    welcome_text = (
        "<b>Привет!</b> 👋\n\n"
        "Я — новостной бот, который анализирует эмоциональную окраску новостей с сайта "
        "<a href='https://lenta.ru'>Lenta.ru</a>.\n"
        "Введите дату или диапазон дат в формате <b>ДД-ММ-ГГГГ</b>.\n\n"
        "<b>Примеры:</b>\n"
        "Один день: <code>15-03-2025</code>\n"
        "Диапазон: <code>01-01-2025 31-01-2025</code>\n\n"
        "Если нужны подсказки, используйте команду <b>/help</b>."
    )
    bot.send_message(chat_id, welcome_text)


@bot.message_handler(commands=['help'])
def send_help(message: types.Message) -> None:
    """
    Обрабатывает команду /help и отправляет инструкции по использованию бота.
    """
    chat_id = message.chat.id
    help_text = (
        "<b>Инструкция по использованию:</b>\n\n"
        "1. Введите дату или диапазон дат в формате <b>ДД-ММ-ГГГГ</b>.\n"
        "   Например, <code>15-03-2025</code> или <code>01-01-2025 31-01-2025</code>.\n\n"
        "2. Бот соберёт новости с <a href='https://lenta.ru'>Lenta.ru</a> и проанализирует их эмоциональную окраску.\n\n"
        "3. Вы увидите график с распределением эмоций и сможете выбрать нужную категорию для просмотра новостей.\n\n"
        "Приятного использования! 😊"
    )
    bot.send_message(chat_id, help_text)


@bot.message_handler(func=lambda message: True)
def process_date_input(message: types.Message) -> None:
    """
    Обрабатывает ввод даты или диапазона дат, собирает новости, оценивает их эмоциональную окраску,
    отображает прогресс и отправляет результаты пользователю.
    """
    chat_id = message.chat.id
    text = message.text.strip()
    parts = text.split()
    aggregated_news: List[Dict[str, Any]] = []

    try:
        if len(parts) == 1:
            day, month, year = map(int, parts[0].split('-'))
            date_obj = datetime.date(year, month, day)
            aggregated_news = get_news_for_day(date_obj)
            bot.send_message(chat_id, f"Найдено <b>{len(aggregated_news)}</b> новостей за <b>{date_obj}</b>.")
        elif len(parts) == 2:
            day1, month1, year1 = map(int, parts[0].split('-'))
            day2, month2, year2 = map(int, parts[1].split('-'))
            start_date = datetime.date(year1, month1, day1)
            end_date = datetime.date(year2, month2, day2)
            if start_date > end_date:
                start_date, end_date = end_date, start_date
            news_range = get_news_for_date_range(start_date, end_date)
            for date_key, news_list in news_range.items():
                for news in news_list:
                    news['date'] = date_key
                    aggregated_news.append(news)
            bot.send_message(chat_id, f"Обработано новостей за период <b>{start_date}</b> - <b>{end_date}</b>.")
        else:
            bot.send_message(chat_id, "Неверный формат ввода. Пожалуйста, введите дату или диапазон дат в формате <b>ДД-ММ-ГГГГ</b>.")
            return
    except Exception as e:
        bot.send_message(chat_id, f"Ошибка обработки ввода: <b>{e}</b>")
        return

    if not aggregated_news:
        bot.send_message(chat_id, "Новостей не найдено. Попробуйте другую дату или диапазон.")
        return

    progress_msg = bot.send_message(chat_id, "Оценка эмоций: " + format_progress(0))
    progress = {"count": 0, "total": len(aggregated_news), "done": False}

    def update_progress() -> None:
        while not progress["done"]:
            percent = int((progress["count"] / progress["total"]) * 100) if progress["total"] > 0 else 100
            try:
                bot.edit_message_text(chat_id=chat_id, message_id=progress_msg.message_id,
                                      text="Оценка эмоций: " + format_progress(percent))
            except Exception:
                pass
            time.sleep(0.5)
        try:
            bot.edit_message_text(chat_id=chat_id, message_id=progress_msg.message_id,
                                  text="Оценка эмоций: " + format_progress(100))
        except Exception:
            pass

    progress_thread = threading.Thread(target=update_progress)
    progress_thread.start()

    for news in aggregated_news:
        first_sentence = get_first_sentence(news['url'])
        combined_text = news['headline']
        if first_sentence:
            combined_text += ". " + first_sentence
        news['combined_text'] = combined_text
        try:
            result = classifier(combined_text, candidate_labels=CANDIDATE_LABELS)
            news['predicted_emotion'] = result["labels"][0]
        except Exception as e:
            logging.error(f"Ошибка классификации текста: {e}")
            news['predicted_emotion'] = "не определено"
        progress["count"] += 1

    progress["done"] = True
    progress_thread.join()

    user_context[chat_id] = aggregated_news

    graph_file = generate_emotion_graph(aggregated_news)
    with open(graph_file, 'rb') as photo:
        bot.send_photo(chat_id, photo, caption="График распределения эмоций новостей")

    markup = types.InlineKeyboardMarkup()
    for label in CANDIDATE_LABELS:
        markup.add(types.InlineKeyboardButton(text=label.capitalize(), callback_data=f"emotion_{label}"))
    bot.send_message(chat_id, "Выберите категорию эмоций для просмотра новостей:", reply_markup=markup)


@bot.callback_query_handler(func=lambda call: call.data.startswith("emotion_"))
def callback_emotion(call: types.CallbackQuery) -> None:
    """
    Обрабатывает выбор пользователем категории эмоций и отправляет соответствующие новости.
    """
    chat_id = call.message.chat.id
    selected_emotion = call.data.split("_", 1)[1].lower()
    news_items = user_context.get(chat_id, [])
    filtered_news = [news for news in news_items if news.get('predicted_emotion', '').lower() == selected_emotion]

    if filtered_news:
        response_text = f"<b>Найдено {len(filtered_news)} новостей с эмоцией '{selected_emotion}':</b>\n\n"
        for news in filtered_news:
            date_info = f" (<i>{news['date']}</i>)" if "date" in news else ""
            response_text += f"• {news['headline']}{date_info}\nСсылка: <a href='{news['url']}'>Читать</a>\n\n"
    else:
        response_text = f"Новостей с эмоцией '{selected_emotion}' не найдено."
    bot.send_message(chat_id, response_text)


def main() -> None:
    """Основная функция для запуска бота."""
    logging.info("Бот запущен.")
    bot.polling(none_stop=True)


if __name__ == "__main__":
    main()
