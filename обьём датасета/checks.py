#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПРОВЕРКИ КАЧЕСТВА ДАННЫХ
Объединенный модуль для всех проверок датасета
"""

import pandas as pd


# ============================================================================
# ПРОВЕРКА СБАЛАНСИРОВАННОГО ДАТАСЕТА
# ============================================================================

def check_balanced_dataset(data_file='dataset_balanced.csv'):
    """Проверка сбалансированного датасета"""
    print("=" * 80)
    print("📊 ПРОВЕРКА СБАЛАНСИРОВАННОГО ДАТАСЕТА")
    print("=" * 80)
    
    df = pd.read_csv(data_file)
    print(f"\n✅ Размер датасета: {len(df):,} строк")
    
    # Анализ распределения
    category_counts = df['category'].value_counts()
    print("\n📂 Распределение по категориям:")
    for category, count in category_counts.items():
        print(f"  {category:<30} {count:>8,}")
    
    # Проверка качества данных
    print(f"\n📏 Качество данных:")
    print(f"  Минимальная длина текста: {df['text'].str.len().min()}")
    print(f"  Максимальная длина текста: {df['text'].str.len().max()}")
    print(f"  Средняя длина текста: {df['text'].str.len().mean():.1f}")
    
    # Проверка на нецензурные слова
    profanity_words = ['блядь', 'сука', 'пизда', 'хуй', 'ебать', 'говно', 'дерьмо']
    profanity_count = 0
    print(f"\n🔍 Проверка нецензурных слов:")
    for word in profanity_words:
        count = df[df['text'].str.lower().str.contains(word, na=False)].shape[0]
        if count > 0:
            profanity_count += count
            print(f"  Найдено \"{word}\": {count} раз")
    
    print(f"\n📊 Всего нецензурных слов: {profanity_count}")
    quality = "ОТЛИЧНО" if profanity_count == 0 else "ТРЕБУЕТ ДОРАБОТКИ"
    print(f"✅ Качество: {quality}")
    
    # Примеры по категориям
    print("\n📝 Примеры по категориям:")
    for category in category_counts.index[:3]:
        examples = df[df['category'] == category]['text'].head(2)
        print(f"\n  {category.upper()}:")
        for i, text in enumerate(examples, 1):
            print(f"    {i}. {text[:80]}...")
    
    return {
        'total_records': len(df),
        'categories': len(category_counts),
        'profanity_count': profanity_count,
        'quality': quality
    }


# ============================================================================
# ФИНАЛЬНАЯ ПРОВЕРКА ДАТАСЕТА
# ============================================================================

def final_check_dataset(data_file='dataset.csv'):
    """Финальная проверка датасета"""
    print("\n" + "=" * 80)
    print("🔍 ФИНАЛЬНАЯ ПРОВЕРКА ДАТАСЕТА")
    print("=" * 80)
    
    df = pd.read_csv(data_file)
    print(f"\n✅ Финальный размер датасета: {df.shape[0]:,} строк")
    
    # Проверка на нецензурные слова
    profanity_words = ['блядь', 'сука', 'пизда', 'хуй', 'ебать', 'говно', 'дерьмо', 
                      'бля', 'пиздец', 'хуйня']
    threat_words = ['убью', 'зарежу', 'застрелю', 'повешу', 'задушу', 'изнасилую', 
                    'изобью', 'угрожаю', 'уничтожу']
    
    profanity_count = 0
    threat_count = 0
    
    print("\n🔍 Проверка нецензурных слов:")
    for word in profanity_words:
        count = df[df['text'].str.lower().str.contains(word, na=False)].shape[0]
        if count > 0:
            profanity_count += count
            print(f"  Найдено \"{word}\": {count} раз")
    
    print("\n🔍 Проверка угроз:")
    for word in threat_words:
        count = df[df['text'].str.lower().str.contains(word, na=False)].shape[0]
        if count > 0:
            threat_count += count
            print(f"  Найдено \"{word}\": {count} раз")
    
    print(f"\n📊 РЕЗУЛЬТАТЫ ПРОВЕРКИ:")
    print(f"  Всего нецензурных слов: {profanity_count}")
    print(f"  Всего угроз: {threat_count}")
    
    quality = "ОТЛИЧНО" if profanity_count == 0 and threat_count == 0 else "ТРЕБУЕТ ДОРАБОТКИ"
    print(f"  ✅ Общее качество: {quality}")
    
    # Статистика по длине текстов
    df['text_length'] = df['text'].str.len()
    print(f"\n📏 Статистика текстов:")
    print(f"  Минимальная длина: {df['text_length'].min()} символов")
    print(f"  Максимальная длина: {df['text_length'].max()} символов")
    print(f"  Средняя длина: {df['text_length'].mean():.1f} символов")
    print(f"  Медианная длина: {df['text_length'].median():.1f} символов")
    
    return {
        'total_records': len(df),
        'profanity_count': profanity_count,
        'threat_count': threat_count,
        'quality': quality
    }


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def run_all_checks():
    """Запуск всех проверок"""
    print("🎯 ПРОВЕРКИ КАЧЕСТВА ДАННЫХ")
    print("=" * 80)
    
    # Проверка сбалансированного датасета
    try:
        check_balanced_dataset('dataset_balanced.csv')
    except FileNotFoundError:
        print("⚠️ Файл dataset_balanced.csv не найден")
    
    # Финальная проверка
    try:
        final_check_dataset('dataset.csv')
    except FileNotFoundError:
        print("⚠️ Файл dataset.csv не найден")
    
    print("\n" + "=" * 80)
    print("✅ ВСЕ ПРОВЕРКИ ЗАВЕРШЕНЫ!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_checks()

