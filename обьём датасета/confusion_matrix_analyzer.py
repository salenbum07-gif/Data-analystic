#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
АНАЛИЗ CONFUSION MATRIX ДЛЯ DATA ANALYST
Детальная матрица ошибок для мультиклассовой классификации

Функциональность:
- Построение confusion matrix из истинных и предсказанных меток
- Расчет метрик (Precision, Recall, F1-Score, Accuracy)
- Визуализация матрицы ошибок (heatmap)
- Анализ ошибок классификации
- Генерация детальных отчетов (Markdown, JSON, CSV)
- Создание графиков и визуализаций

Пример использования:
    from confusion_matrix_analyzer import ConfusionMatrixAnalyzer
    
    analyzer = ConfusionMatrixAnalyzer()
    matrix = analyzer.build_confusion_matrix(y_true, y_pred, classes)
    analyzer.print_detailed_report()
    analyzer.save_report('report.md')
    analyzer.create_all_visualizations()
    
Требования:
    - pandas
    - numpy
    - matplotlib (для визуализаций)
    - seaborn (для визуализаций)
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from collections import defaultdict, Counter

# Импорт системы категоризации ошибок
try:
    from error_categories import ErrorCategoryManager, create_default_manager
    HAS_ERROR_CATEGORIES = True
except ImportError:
    HAS_ERROR_CATEGORIES = False
    ErrorCategoryManager = None

# Импорт библиотек для визуализации (опционально)
try:
    import matplotlib
    matplotlib.use('Agg')  # Неинтерактивный бэкенд для сохранения без показа
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VISUALIZATION = True
    # Настройка для графиков
    plt.rcParams['font.family'] = 'DejaVu Sans'
    sns.set_style("whitegrid")
    sns.set_palette("husl")
except ImportError:
    HAS_VISUALIZATION = False
    plt = None
    sns = None

class ConfusionMatrixAnalyzer:
    """Класс для анализа матрицы ошибок (Confusion Matrix)"""
    
    def __init__(self, error_category_manager=None):
        self.confusion_matrix = None
        self.classes = None
        self.results = {}
        self.figures_dir = 'confusion_matrix_figures'
        if not os.path.exists(self.figures_dir):
            os.makedirs(self.figures_dir)
        
        # Инициализация менеджера категорий ошибок
        if HAS_ERROR_CATEGORIES:
            if error_category_manager is None:
                self.category_manager = create_default_manager()
            else:
                self.category_manager = error_category_manager
        else:
            self.category_manager = None
        
    def build_confusion_matrix(self, y_true, y_pred, classes=None):
        """
        Построение confusion matrix
        
        Args:
            y_true: список истинных меток
            y_pred: список предсказанных меток
            classes: список всех классов (опционально)
        
        Returns:
            confusion_matrix: 2D массив (pandas DataFrame)
        """
        # Определение классов
        if classes is None:
            classes = sorted(set(y_true) | set(y_pred))
        
        self.classes = classes
        
        # Проверка длины
        if len(y_true) != len(y_pred):
            raise ValueError(f"Длины не совпадают: y_true={len(y_true)}, y_pred={len(y_pred)}")
        
        # Создание матрицы
        n_classes = len(classes)
        matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        # Заполнение матрицы
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        for true_label, pred_label in zip(y_true, y_pred):
            true_idx = class_to_idx.get(true_label, -1)
            pred_idx = class_to_idx.get(pred_label, -1)
            if true_idx >= 0 and pred_idx >= 0:
                matrix[true_idx][pred_idx] += 1
        
        # Создание DataFrame
        self.confusion_matrix = pd.DataFrame(
            matrix,
            index=classes,
            columns=classes
        )
        
        return self.confusion_matrix
    
    def calculate_metrics_from_matrix(self):
        """
        Расчет метрик из confusion matrix
        
        Returns:
            dict с метриками для каждого класса
        """
        if self.confusion_matrix is None:
            raise ValueError("Сначала нужно построить confusion matrix")
        
        metrics = {}
        
        for i, true_class in enumerate(self.classes):
            tp = self.confusion_matrix.loc[true_class, true_class]  # True Positives
            
            # False Positives: все предсказания этого класса, кроме правильных
            fp = self.confusion_matrix.loc[:, true_class].sum() - tp
            
            # False Negatives: все истинные этого класса, кроме правильных
            fn = self.confusion_matrix.loc[true_class, :].sum() - tp
            
            # True Negatives: все остальные
            tn = self.confusion_matrix.values.sum() - tp - fp - fn
            
            # Метрики
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            
            metrics[true_class] = {
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'accuracy': round(accuracy, 4),
                'support': int(tp + fn)
            }
        
        return metrics
    
    def get_normalized_matrix(self):
        """
        Получение нормализованной матрицы (в процентах)
        
        Returns:
            normalized_matrix: DataFrame с процентами
        """
        if self.confusion_matrix is None:
            raise ValueError("Сначала нужно построить confusion matrix")
        
        normalized = self.confusion_matrix.copy()
        
        # Нормализация по строкам (истинным классам)
        row_sums = normalized.sum(axis=1)
        for cls in self.classes:
            if row_sums[cls] > 0:
                normalized.loc[cls] = (normalized.loc[cls] / row_sums[cls] * 100).round(2)
        
        return normalized
    
    def find_common_mistakes(self, top_n=10):
        """
        Поиск самых частых ошибок классификации
        
        Args:
            top_n: количество топ-ошибок
        
        Returns:
            list кортежей (true_class, pred_class, count)
        """
        if self.confusion_matrix is None:
            raise ValueError("Сначала нужно построить confusion matrix")
        
        mistakes = []
        
        for true_cls in self.classes:
            for pred_cls in self.classes:
                if true_cls != pred_cls:
                    count = self.confusion_matrix.loc[true_cls, pred_cls]
                    if count > 0:
                        mistakes.append((true_cls, pred_cls, int(count)))
        
        # Сортировка по количеству ошибок
        mistakes.sort(key=lambda x: x[2], reverse=True)
        
        return mistakes[:top_n]
    
    def interpret_errors(self):
        """
        Детальная интерпретация ошибок классификации
        
        Returns:
            dict с интерпретацией
        """
        if self.confusion_matrix is None:
            raise ValueError("Сначала нужно построить confusion matrix")
        
        metrics = self.calculate_metrics_from_matrix()
        mistakes = self.find_common_mistakes(top_n=50)  # Увеличиваем для полного анализа
        
        interpretation = {
            'problematic_classes': [],
            'symmetric_errors': [],
            'dominant_confusions': [],
            'low_performance_classes': [],
            'recommendations': [],
            'error_patterns': [],
            'class_stability': [],
            'confusion_clusters': [],
            'detailed_analysis': {}
        }
        
        # 1. Классы с низкой производительностью
        for cls, m in metrics.items():
            if m['f1'] < 0.5 or m['precision'] < 0.5 or m['recall'] < 0.5:
                interpretation['low_performance_classes'].append({
                    'class': cls,
                    'precision': m['precision'],
                    'recall': m['recall'],
                    'f1': m['f1'],
                    'issues': []
                })
                
                if m['precision'] < 0.5:
                    interpretation['low_performance_classes'][-1]['issues'].append(
                        f"Низкая точность ({m['precision']:.2%}) - много ложных срабатываний"
                    )
                if m['recall'] < 0.5:
                    interpretation['low_performance_classes'][-1]['issues'].append(
                        f"Низкая полнота ({m['recall']:.2%}) - много пропущенных случаев"
                    )
        
        # 2. Симметричные ошибки (классы путаются друг с другом)
        mistake_dict = {(t, p): c for t, p, c in mistakes}
        for true_cls, pred_cls, count in mistakes:
            reverse_count = mistake_dict.get((pred_cls, true_cls), 0)
            if reverse_count > 0:
                # Это симметричная ошибка
                if not any(s['class1'] == true_cls and s['class2'] == pred_cls 
                          for s in interpretation['symmetric_errors']):
                    interpretation['symmetric_errors'].append({
                        'class1': true_cls,
                        'class2': pred_cls,
                        'count1_to_2': count,
                        'count2_to_1': reverse_count,
                        'total_mistakes': count + reverse_count
                    })
        
        # 3. Доминирующие ошибки (более 20% от класса)
        for true_cls in self.classes:
            total_true = self.confusion_matrix.loc[true_cls, :].sum()
            for pred_cls in self.classes:
                if true_cls != pred_cls:
                    count = self.confusion_matrix.loc[true_cls, pred_cls]
                    percentage = (count / total_true * 100) if total_true > 0 else 0
                    if percentage >= 20:  # Более 20% ошибок в этом направлении
                        interpretation['dominant_confusions'].append({
                            'true_class': true_cls,
                            'predicted_class': pred_cls,
                            'count': int(count),
                            'percentage': round(percentage, 2),
                            'severity': 'критично' if percentage >= 50 else 'высокая' if percentage >= 30 else 'средняя'
                        })
        
        # 4. Проблемные классы (много ошибок во всех направлениях)
        for cls in self.classes:
            total_class = self.confusion_matrix.loc[cls, :].sum()
            correct = self.confusion_matrix.loc[cls, cls]
            error_rate = ((total_class - correct) / total_class * 100) if total_class > 0 else 0
            
            if error_rate >= 50:  # Более 50% ошибок
                interpretation['problematic_classes'].append({
                    'class': cls,
                    'error_rate': round(error_rate, 2),
                    'correct': int(correct),
                    'total': int(total_class),
                    'main_confusions': []
                })
                
                # Находим основные источники ошибок
                for pred_cls in self.classes:
                    if pred_cls != cls:
                        count = self.confusion_matrix.loc[cls, pred_cls]
                        if count > 0:
                            interpretation['problematic_classes'][-1]['main_confusions'].append({
                                'confused_with': pred_cls,
                                'count': int(count),
                                'percentage': round((count / total_class * 100), 2)
                            })
                
                # Сортировка по количеству
                interpretation['problematic_classes'][-1]['main_confusions'].sort(
                    key=lambda x: x['count'], reverse=True
                )
                interpretation['problematic_classes'][-1]['main_confusions'] = \
                    interpretation['problematic_classes'][-1]['main_confusions'][:3]
        
        # 5. Рекомендации
        recommendations = []
        
        # Рекомендации по симметричным ошибкам
        if interpretation['symmetric_errors']:
            recommendations.append({
                'type': 'симметричные_ошибки',
                'priority': 'высокий',
                'description': f"Обнаружено {len(interpretation['symmetric_errors'])} пар классов, которые путаются друг с другом",
                'action': 'Необходимо добавить больше различительных признаков между этими классами'
            })
        
        # Рекомендации по доминирующим ошибкам
        critical_confusions = [c for c in interpretation['dominant_confusions'] if c['severity'] == 'критично']
        if critical_confusions:
            recommendations.append({
                'type': 'критические_путаницы',
                'priority': 'критичный',
                'description': f"Обнаружено {len(critical_confusions)} критичных путаниц (>50% ошибок)",
                'action': 'Требуется пересмотр признаков или разделение классов'
            })
        
        # Рекомендации по проблемным классам
        if interpretation['low_performance_classes']:
            low_precision = [c for c in interpretation['low_performance_classes'] if c['precision'] < 0.5]
            low_recall = [c for c in interpretation['low_performance_classes'] if c['recall'] < 0.5]
            
            if low_precision:
                recommendations.append({
                    'type': 'низкая_точность',
                    'priority': 'высокий',
                    'description': f"{len(low_precision)} классов имеют низкую точность (<50%)",
                    'action': 'Уменьшить порог классификации или добавить больше отрицательных примеров'
                })
            
            if low_recall:
                recommendations.append({
                    'type': 'низкая_полнота',
                    'priority': 'высокий',
                    'description': f"{len(low_recall)} классов имеют низкую полноту (<50%)",
                    'action': 'Снизить порог классификации или увеличить вес этих классов'
                })
        
        interpretation['recommendations'] = recommendations
        
        # 6. ДЕТАЛЬНЫЙ АНАЛИЗ ОШИБОК ПО КАЖДОМУ КЛАССУ
        interpretation['detailed_analysis'] = self._detailed_class_analysis(metrics)
        
        # 7. АНАЛИЗ ПАТТЕРНОВ ОШИБОК
        interpretation['error_patterns'] = self._analyze_error_patterns()
        
        # 8. АНАЛИЗ СТАБИЛЬНОСТИ КЛАССОВ
        interpretation['class_stability'] = self._analyze_class_stability(metrics)
        
        # 9. КЛАСТЕРИЗАЦИЯ ПУТАНИЦ
        interpretation['confusion_clusters'] = self._find_confusion_clusters()
        
        # 10. СТАТИСТИЧЕСКИЙ АНАЛИЗ ОШИБОК
        interpretation['error_statistics'] = self._calculate_error_statistics()
        
        # 11. КАТЕГОРИЗАЦИЯ ОШИБОК ПО КАТЕГОРИЯМ
        if self.category_manager:
            interpretation['error_categorization'] = self._categorize_errors(metrics, interpretation)
        
        return interpretation
    
    def _categorize_errors(self, metrics, interpretation):
        """Категоризация ошибок по категориям"""
        categorization = {
            'by_category': {},  # Ошибки по категориям
            'by_class': {},  # Категории для каждого класса
            'category_summary': {}  # Сводка по категориям
        }
        
        detailed = interpretation.get('detailed_analysis', {})
        patterns = interpretation.get('error_patterns', {})
        stability = interpretation.get('class_stability', [])
        
        # Создаем словарь стабильности для быстрого доступа
        stability_dict = {s['class']: s for s in stability}
        
        # Категоризация по классам
        for cls in self.classes:
            class_metrics = metrics.get(cls, {})
            class_analysis = detailed.get(cls, {})
            
            # Добавляем данные о стабильности и паттернах
            class_analysis['stability_score'] = stability_dict.get(cls, {}).get('stability_score', 1.0)
            class_analysis['unique_error_classes'] = stability_dict.get(cls, {}).get('unique_error_classes', 0)
            
            # Проверяем концентрированные ошибки
            for ce in patterns.get('concentrated_errors', []):
                if ce['class'] == cls:
                    class_analysis['concentration'] = ce['concentration']
                    break
            
            if self.category_manager:
                categories = self.category_manager.categorize_error(
                    cls, class_metrics, class_analysis
                )
                categorization['by_class'][cls] = categories
                
                # Добавление в категории
                for cat_name in categories:
                    if cat_name not in categorization['by_category']:
                        categorization['by_category'][cat_name] = []
                    categorization['by_category'][cat_name].append({
                        'class': cls,
                        'metrics': class_metrics,
                        'analysis': class_analysis
                    })
        
        # Сводка по категориям
        if self.category_manager:
            for cat_name, category in self.category_manager.categories.items():
                classes_in_category = categorization['by_category'].get(cat_name, [])
                categorization['category_summary'][cat_name] = {
                    'name': cat_name,
                    'description': category.description,
                    'severity': category.severity.value,
                    'classes_count': len(classes_in_category),
                    'classes': [item['class'] for item in classes_in_category],
                    'recommendations': category.recommendations
                }
        
        return categorization
    
    def _detailed_class_analysis(self, metrics):
        """Детальный анализ каждого класса"""
        detailed = {}
        
        for cls in self.classes:
            m = metrics[cls]
            total_class = self.confusion_matrix.loc[cls, :].sum()
            correct = self.confusion_matrix.loc[cls, cls]
            errors = total_class - correct
            
            # Анализ распределения ошибок
            error_distribution = {}
            for pred_cls in self.classes:
                if pred_cls != cls:
                    count = self.confusion_matrix.loc[cls, pred_cls]
                    if count > 0:
                        error_distribution[pred_cls] = {
                            'count': int(count),
                            'percentage': round((count / total_class * 100), 2),
                            'percentage_of_errors': round((count / errors * 100), 2) if errors > 0 else 0
                        }
            
            # Определение типа проблем
            issues = []
            if m['precision'] < 0.3:
                issues.append('критически_низкая_точность')
            elif m['precision'] < 0.5:
                issues.append('низкая_точность')
            
            if m['recall'] < 0.3:
                issues.append('критически_низкая_полнота')
            elif m['recall'] < 0.5:
                issues.append('низкая_полнота')
            
            if m['f1'] < 0.3:
                issues.append('критически_низкий_f1')
            elif m['f1'] < 0.5:
                issues.append('низкий_f1')
            
            # Анализ направлений ошибок
            most_confused_with = sorted(
                error_distribution.items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )[:5]
            
            detailed[cls] = {
                'metrics': m,
                'total_samples': int(total_class),
                'correct_predictions': int(correct),
                'error_count': int(errors),
                'error_rate': round((errors / total_class * 100), 2) if total_class > 0 else 0,
                'error_distribution': error_distribution,
                'most_confused_with': [
                    {'class': k, 'count': v['count'], 'percentage': v['percentage']} 
                    for k, v in most_confused_with
                ],
                'issues': issues,
                'severity': self._calculate_severity(m, errors, total_class),
                'interpretation': self._generate_class_interpretation(cls, m, error_distribution, most_confused_with)
            }
        
        return detailed
    
    def _calculate_severity(self, metrics, errors, total):
        """Расчет серьезности проблем класса"""
        error_rate = (errors / total * 100) if total > 0 else 0
        
        if metrics['f1'] < 0.3 or error_rate > 70:
            return 'критическая'
        elif metrics['f1'] < 0.5 or error_rate > 50:
            return 'высокая'
        elif metrics['f1'] < 0.7 or error_rate > 30:
            return 'средняя'
        else:
            return 'низкая'
    
    def _generate_class_interpretation(self, cls, metrics, error_dist, most_confused):
        """Генерация текстовой интерпретации для класса"""
        interpretations = []
        
        # Анализ точности
        if metrics['precision'] < 0.5:
            interpretations.append(
                f"Класс '{cls}' имеет низкую точность ({metrics['precision']:.1%}). "
                f"Модель часто ошибочно классифицирует другие классы как '{cls}'. "
                f"Это означает, что модель слишком агрессивно предсказывает этот класс."
            )
        elif metrics['precision'] > 0.9:
            interpretations.append(
                f"Класс '{cls}' имеет высокую точность ({metrics['precision']:.1%}). "
                f"Когда модель предсказывает этот класс, она обычно права."
            )
        
        # Анализ полноты
        if metrics['recall'] < 0.5:
            interpretations.append(
                f"Класс '{cls}' имеет низкую полноту ({metrics['recall']:.1%}). "
                f"Модель пропускает много реальных примеров этого класса. "
                f"Это означает, что модель слишком консервативна в предсказании '{cls}'."
            )
        elif metrics['recall'] > 0.9:
            interpretations.append(
                f"Класс '{cls}' имеет высокую полноту ({metrics['recall']:.1%}). "
                f"Модель успешно находит большинство примеров этого класса."
            )
        
        # Анализ основных путаниц
        if most_confused:
            top_confusion = most_confused[0]
            interpretations.append(
                f"Основная путаница: '{cls}' чаще всего ошибочно классифицируется как "
                f"'{top_confusion[0]}' ({top_confusion[1]['count']} случаев, "
                f"{top_confusion[1]['percentage']:.1f}% от всех примеров класса). "
                f"Это указывает на семантическую близость или перекрывающиеся признаки между этими классами."
            )
        
        # Анализ F1
        if metrics['f1'] < 0.5:
            interpretations.append(
                f"Общая производительность класса '{cls}' низкая (F1={metrics['f1']:.1%}). "
                f"Требуется значительное улучшение как точности, так и полноты."
            )
        
        return " ".join(interpretations) if interpretations else f"Класс '{cls}' показывает приемлемую производительность."
    
    def _analyze_error_patterns(self):
        """Анализ паттернов ошибок"""
        patterns = {
            'one_way_confusions': [],  # Односторонние путаницы
            'bidirectional_confusions': [],  # Двусторонние путаницы
            'scattered_errors': [],  # Разбросанные ошибки
            'concentrated_errors': []  # Концентрированные ошибки
        }
        
        for true_cls in self.classes:
            total_true = self.confusion_matrix.loc[true_cls, :].sum()
            correct = self.confusion_matrix.loc[true_cls, true_cls]
            errors = total_true - correct
            
            if errors == 0:
                continue
            
            # Анализ распределения ошибок
            error_counts = []
            for pred_cls in self.classes:
                if pred_cls != true_cls:
                    count = self.confusion_matrix.loc[true_cls, pred_cls]
                    if count > 0:
                        error_counts.append((pred_cls, count, (count / errors * 100)))
            
            error_counts.sort(key=lambda x: x[1], reverse=True)
            
            # Определение типа паттерна
            if len(error_counts) == 0:
                continue
            
            top_error_pct = error_counts[0][2] if error_counts else 0
            
            # Концентрированные ошибки (>60% в одном направлении)
            if top_error_pct > 60:
                patterns['concentrated_errors'].append({
                    'class': true_cls,
                    'main_confusion': error_counts[0][0],
                    'concentration': round(top_error_pct, 1),
                    'total_errors': int(errors)
                })
            # Разбросанные ошибки (<30% в любом направлении)
            elif top_error_pct < 30:
                patterns['scattered_errors'].append({
                    'class': true_cls,
                    'error_distribution': len(error_counts),
                    'total_errors': int(errors),
                    'interpretation': 'Ошибки распределены между многими классами, что указывает на общую неопределенность модели для этого класса'
                })
            
            # Проверка двусторонних путаниц
            for pred_cls, count, pct in error_counts[:3]:
                reverse_count = self.confusion_matrix.loc[pred_cls, true_cls]
                if reverse_count > 0:
                    reverse_pct = (reverse_count / self.confusion_matrix.loc[pred_cls, :].sum() * 100) if self.confusion_matrix.loc[pred_cls, :].sum() > 0 else 0
                    if not any(b['class1'] == true_cls and b['class2'] == pred_cls for b in patterns['bidirectional_confusions']):
                        patterns['bidirectional_confusions'].append({
                            'class1': true_cls,
                            'class2': pred_cls,
                            'count1_to_2': int(count),
                            'count2_to_1': int(reverse_count),
                            'pct1_to_2': round(pct, 1),
                            'pct2_to_1': round(reverse_pct, 1),
                            'interpretation': f"Классы '{true_cls}' и '{pred_cls}' путаются друг с другом в обоих направлениях, что указывает на семантическую близость"
                        })
        
        return patterns
    
    def _analyze_class_stability(self, metrics):
        """Анализ стабильности классификации классов"""
        stability = []
        
        for cls in self.classes:
            m = metrics[cls]
            total = self.confusion_matrix.loc[cls, :].sum()
            
            # Количество различных классов, в которые ошибочно классифицируется
            unique_errors = sum(1 for pred_cls in self.classes 
                              if pred_cls != cls and self.confusion_matrix.loc[cls, pred_cls] > 0)
            
            # Стабильность = насколько предсказуемы ошибки
            error_entropy = 0
            errors = total - self.confusion_matrix.loc[cls, cls]
            if errors > 0:
                for pred_cls in self.classes:
                    if pred_cls != cls:
                        count = self.confusion_matrix.loc[cls, pred_cls]
                        if count > 0:
                            p = count / errors
                            error_entropy -= p * np.log2(p + 1e-10)
            
            # Нормализованная энтропия (0 = все ошибки в одном классе, 1 = равномерное распределение)
            max_entropy = np.log2(max(unique_errors, 1))
            normalized_entropy = (error_entropy / max_entropy) if max_entropy > 0 else 0
            
            stability.append({
                'class': cls,
                'stability_score': round(1 - normalized_entropy, 3),  # Выше = стабильнее
                'unique_error_classes': unique_errors,
                'error_entropy': round(error_entropy, 3),
                'interpretation': self._interpret_stability(cls, normalized_entropy, unique_errors, m)
            })
        
        return sorted(stability, key=lambda x: x['stability_score'])
    
    def _interpret_stability(self, cls, entropy, unique_errors, metrics):
        """Интерпретация стабильности класса"""
        if entropy < 0.3:
            return f"Класс '{cls}' имеет высокую стабильность ошибок - ошибки концентрируются в 1-2 классах. Это хорошо для исправления."
        elif entropy > 0.7:
            return f"Класс '{cls}' имеет низкую стабильность - ошибки распределены между {unique_errors} классами. Требуется общее улучшение модели."
        else:
            return f"Класс '{cls}' имеет среднюю стабильность ошибок."
    
    def _find_confusion_clusters(self):
        """Поиск кластеров путаниц (групп классов, которые часто путаются)"""
        clusters = []
        processed_pairs = set()
        
        # Создаем граф путаниц
        confusion_graph = {}
        for true_cls in self.classes:
            confusion_graph[true_cls] = []
            for pred_cls in self.classes:
                if true_cls != pred_cls:
                    count = self.confusion_matrix.loc[true_cls, pred_cls]
                    if count > 0:
                        total = self.confusion_matrix.loc[true_cls, :].sum()
                        pct = (count / total * 100) if total > 0 else 0
                        if pct > 10:  # Более 10% ошибок
                            confusion_graph[true_cls].append((pred_cls, count, pct))
        
        # Поиск кластеров (классы, которые путаются друг с другом)
        for cls1 in self.classes:
            for cls2 in self.classes:
                if cls1 >= cls2:  # Избегаем дублирования
                    continue
                
                pair_key = tuple(sorted([cls1, cls2]))
                if pair_key in processed_pairs:
                    continue
                
                # Проверяем двустороннюю путаницу
                count1_to_2 = self.confusion_matrix.loc[cls1, cls2]
                count2_to_1 = self.confusion_matrix.loc[cls2, cls1]
                
                if count1_to_2 > 0 and count2_to_1 > 0:
                    total1 = self.confusion_matrix.loc[cls1, :].sum()
                    total2 = self.confusion_matrix.loc[cls2, :].sum()
                    pct1_to_2 = (count1_to_2 / total1 * 100) if total1 > 0 else 0
                    pct2_to_1 = (count2_to_1 / total2 * 100) if total2 > 0 else 0
                    
                    if pct1_to_2 > 5 or pct2_to_1 > 5:  # Значимая путаница
                        clusters.append({
                            'classes': [cls1, cls2],
                            'count1_to_2': int(count1_to_2),
                            'count2_to_1': int(count2_to_1),
                            'pct1_to_2': round(pct1_to_2, 1),
                            'pct2_to_1': round(pct2_to_1, 1),
                            'total_confusions': int(count1_to_2 + count2_to_1),
                            'strength': 'сильная' if (pct1_to_2 > 20 and pct2_to_1 > 20) else 'средняя' if (pct1_to_2 > 10 or pct2_to_1 > 10) else 'слабая'
                        })
                        processed_pairs.add(pair_key)
        
        return sorted(clusters, key=lambda x: x['total_confusions'], reverse=True)
    
    def _calculate_error_statistics(self):
        """Статистический анализ ошибок"""
        total_samples = self.confusion_matrix.values.sum()
        total_correct = sum(self.confusion_matrix.loc[cls, cls] for cls in self.classes)
        total_errors = total_samples - total_correct
        
        # Распределение ошибок по классам
        error_by_class = {}
        for cls in self.classes:
            total_class = self.confusion_matrix.loc[cls, :].sum()
            correct = self.confusion_matrix.loc[cls, cls]
            errors = total_class - correct
            error_by_class[cls] = {
                'errors': int(errors),
                'error_rate': round((errors / total_class * 100), 2) if total_class > 0 else 0,
                'contribution_to_total_errors': round((errors / total_errors * 100), 2) if total_errors > 0 else 0
            }
        
        # Классы с наибольшим вкладом в общие ошибки
        top_error_contributors = sorted(
            error_by_class.items(),
            key=lambda x: x[1]['contribution_to_total_errors'],
            reverse=True
        )[:5]
        
        return {
            'total_samples': int(total_samples),
            'total_correct': int(total_correct),
            'total_errors': int(total_errors),
            'overall_error_rate': round((total_errors / total_samples * 100), 2) if total_samples > 0 else 0,
            'error_by_class': error_by_class,
            'top_error_contributors': [
                {'class': k, 'errors': v['errors'], 'contribution_pct': v['contribution_to_total_errors']}
                for k, v in top_error_contributors
            ],
            'average_error_rate_per_class': round(
                np.mean([v['error_rate'] for v in error_by_class.values()]), 2
            ),
            'error_rate_std': round(
                np.std([v['error_rate'] for v in error_by_class.values()]), 2
            )
        }
    
    def print_matrix(self, normalized=False):
        """Вывод матрицы в консоль"""
        if self.confusion_matrix is None:
            print("❌ Матрица не построена")
            return
        
        matrix_to_print = self.get_normalized_matrix() if normalized else self.confusion_matrix
        
        print("\n" + "=" * 100)
        title = "НОРМАЛИЗОВАННАЯ CONFUSION MATRIX (%)" if normalized else "CONFUSION MATRIX (абсолютные значения)"
        print(f"📊 {title}")
        print("=" * 100)
        print("\nСтроки = Истинные классы | Колонки = Предсказанные классы")
        print("-" * 100)
        
        # Заголовок
        true_pred = 'True\\Pred'
        header = f"{true_pred:<25}"
        for pred_cls in self.classes:
            # Сокращение длинных названий
            short_name = pred_cls[:15] + "..." if len(pred_cls) > 15 else pred_cls
            header += f"{short_name:>10}"
        print(header)
        print("-" * 100)
        
        # Строки матрицы
        for true_cls in self.classes:
            short_true = true_cls[:22] + "..." if len(true_cls) > 22 else true_cls
            row = f"{short_true:<25}"
            for pred_cls in self.classes:
                value = matrix_to_print.loc[true_cls, pred_cls]
                if normalized:
                    row += f"{value:>9.1f}%"
                else:
                    row += f"{int(value):>10}"
            print(row)
        
        print("=" * 100)
        
        # Диагональ (правильные предсказания)
        diagonal_sum = sum(self.confusion_matrix.loc[cls, cls] for cls in self.classes)
        total = self.confusion_matrix.values.sum()
        accuracy = (diagonal_sum / total * 100) if total > 0 else 0
        
        print(f"\n✅ Общая точность: {diagonal_sum:,} / {total:,} = {accuracy:.2f}%")
    
    def print_detailed_report(self):
        """Вывод детального отчета"""
        if self.confusion_matrix is None:
            print("❌ Матрица не построена")
            return
        
        metrics = self.calculate_metrics_from_matrix()
        
        print("\n" + "=" * 100)
        print("📊 ДЕТАЛЬНЫЙ ОТЧЕТ ПО CONFUSION MATRIX")
        print("=" * 100)
        
        # Вывод матрицы
        self.print_matrix(normalized=False)
        
        print("\n" + "=" * 100)
        print("📊 НОРМАЛИЗОВАННАЯ МАТРИЦА (в % от истинного класса)")
        print("=" * 100)
        self.print_matrix(normalized=True)
        
        # Метрики по классам
        print("\n" + "=" * 100)
        print("📋 МЕТРИКИ ПО КЛАССАМ")
        print("=" * 100)
        print(f"{'Класс':<30} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("-" * 100)
        
        for cls in self.classes:
            m = metrics[cls]
            print(f"{cls:<30} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                  f"{m['f1']:>10.4f} {m['support']:>10}")
        
        # Самые частые ошибки
        print("\n" + "=" * 100)
        print("⚠️ ТОП-10 САМЫХ ЧАСТЫХ ОШИБОК")
        print("=" * 100)
        print(f"{'Истинный класс':<30} {'Предсказанный класс':<30} {'Количество':>10} {'% от класса':>12}")
        print("-" * 100)
        
        mistakes = self.find_common_mistakes(top_n=10)
        for true_cls, pred_cls, count in mistakes:
            total_true = self.confusion_matrix.loc[true_cls, :].sum()
            percentage = (count / total_true * 100) if total_true > 0 else 0
            print(f"{true_cls:<30} {pred_cls:<30} {count:>10} {percentage:>11.2f}%")
        
        # Общие метрики
        print("\n" + "=" * 100)
        print("📊 ОБЩИЕ МЕТРИКИ")
        print("=" * 100)
        
        macro_precision = np.mean([m['precision'] for m in metrics.values()])
        macro_recall = np.mean([m['recall'] for m in metrics.values()])
        macro_f1 = np.mean([m['f1'] for m in metrics.values()])
        
        total_tp = sum(m['tp'] for m in metrics.values())
        total_fp = sum(m['fp'] for m in metrics.values())
        total_fn = sum(m['fn'] for m in metrics.values())
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        print(f"Macro Precision:  {macro_precision:.4f}")
        print(f"Macro Recall:     {macro_recall:.4f}")
        print(f"Macro F1:         {macro_f1:.4f}")
        print(f"\nMicro Precision:  {micro_precision:.4f}")
        print(f"Micro Recall:     {micro_recall:.4f}")
        print(f"Micro F1:         {micro_f1:.4f}")
        
        total_samples = self.confusion_matrix.values.sum()
        correct = sum(self.confusion_matrix.loc[cls, cls] for cls in self.classes)
        accuracy = (correct / total_samples * 100) if total_samples > 0 else 0
        print(f"\nОбщая точность (Accuracy): {correct:,} / {total_samples:,} = {accuracy:.2f}%")
        
        # Полная интерпретация ошибок
        self.print_full_interpretation()
        
        # Дополнительная информация (старый формат для совместимости)
        interpretation = self.interpret_errors()
        
        # Проблемные классы
        if interpretation['problematic_classes']:
            print("\n⚠️ ПРОБЛЕМНЫЕ КЛАССЫ (более 50% ошибок):")
            print("-" * 100)
            for pc in interpretation['problematic_classes']:
                print(f"\n📌 Класс: {pc['class']}")
                print(f"   Ошибок: {pc['error_rate']}% ({pc['total'] - pc['correct']:,} из {pc['total']:,})")
                print(f"   Правильных: {pc['correct']:,} ({100 - pc['error_rate']:.1f}%)")
                if pc['main_confusions']:
                    print(f"   Основные путаницы:")
                    for conf in pc['main_confusions']:
                        print(f"     → {conf['confused_with']}: {conf['count']:,} ({conf['percentage']}%)")
        
        # Классы с низкой производительностью
        if interpretation['low_performance_classes']:
            print("\n📉 КЛАССЫ С НИЗКОЙ ПРОИЗВОДИТЕЛЬНОСТЬЮ:")
            print("-" * 100)
            for lpc in interpretation['low_performance_classes']:
                print(f"\n📌 {lpc['class']}:")
                print(f"   Precision: {lpc['precision']:.2%} | Recall: {lpc['recall']:.2%} | F1: {lpc['f1']:.2%}")
                for issue in lpc['issues']:
                    print(f"   ⚠️ {issue}")
        
        # Симметричные ошибки
        if interpretation['symmetric_errors']:
            print("\n🔄 СИММЕТРИЧНЫЕ ОШИБКИ (классы путаются друг с другом):")
            print("-" * 100)
            for se in interpretation['symmetric_errors']:
                print(f"\n📌 {se['class1']} ↔ {se['class2']}:")
                print(f"   {se['class1']} → {se['class2']}: {se['count1_to_2']:,} ошибок")
                print(f"   {se['class2']} → {se['class1']}: {se['count2_to_1']:,} ошибок")
                print(f"   Всего ошибок: {se['total_mistakes']:,}")
                print(f"   💡 Интерпретация: Эти классы имеют перекрывающиеся признаки")
        
        # Доминирующие ошибки
        if interpretation['dominant_confusions']:
            print("\n📊 ДОМИНИРУЮЩИЕ ОШИБКИ (более 20% от класса):")
            print("-" * 100)
            # Группировка по степени серьезности
            by_severity = {'критично': [], 'высокая': [], 'средняя': []}
            for dc in interpretation['dominant_confusions']:
                by_severity[dc['severity']].append(dc)
            
            for severity, confusions in by_severity.items():
                if confusions:
                    severity_emoji = {'критично': '🔴', 'высокая': '🟠', 'средняя': '🟡'}
                    print(f"\n{severity_emoji.get(severity, '•')} {severity.upper()}:")
                    for dc in sorted(confusions, key=lambda x: x['percentage'], reverse=True):
                        print(f"   {dc['true_class']} → {dc['predicted_class']}: "
                              f"{dc['count']:,} ({dc['percentage']}% от класса)")
        
        # Рекомендации
        if interpretation['recommendations']:
            print("\n💡 РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ:")
            print("-" * 100)
            for i, rec in enumerate(interpretation['recommendations'], 1):
                priority_emoji = {'критичный': '🔴', 'высокий': '🟠', 'средний': '🟡', 'низкий': '🟢'}
                print(f"\n{i}. {priority_emoji.get(rec['priority'], '•')} [{rec['priority'].upper()}] {rec['type'].replace('_', ' ').title()}")
                print(f"   Описание: {rec['description']}")
                print(f"   Действие: {rec['action']}")
        
        print("\n" + "=" * 100)
    
    def print_full_interpretation(self):
        """Вывод полной интерпретации ошибок"""
        if self.confusion_matrix is None:
            print("❌ Матрица не построена")
            return
        
        interpretation = self.interpret_errors()
        
        print("\n" + "=" * 100)
        print("🔍 ПОЛНАЯ ИНТЕРПРЕТАЦИЯ ОШИБОК КЛАССИФИКАЦИИ")
        print("=" * 100)
        
        # Статистика ошибок
        stats = interpretation['error_statistics']
        print("\n📊 ОБЩАЯ СТАТИСТИКА ОШИБОК")
        print("-" * 100)
        print(f"Всего примеров: {stats['total_samples']:,}")
        print(f"Правильных предсказаний: {stats['total_correct']:,} ({100 - stats['overall_error_rate']:.2f}%)")
        print(f"Ошибок: {stats['total_errors']:,} ({stats['overall_error_rate']:.2f}%)")
        print(f"Средний процент ошибок по классам: {stats['average_error_rate_per_class']:.2f}% (σ={stats['error_rate_std']:.2f}%)")
        
        if stats['top_error_contributors']:
            print("\n🔴 ТОП-5 КЛАССОВ С НАИБОЛЬШИМ ВКЛАДОМ В ОШИБКИ:")
            for i, contrib in enumerate(stats['top_error_contributors'], 1):
                print(f"  {i}. {contrib['class']}: {contrib['errors']:,} ошибок ({contrib['contribution_pct']:.1f}% от всех ошибок)")
        
        # Детальный анализ по классам
        print("\n" + "=" * 100)
        print("📋 ДЕТАЛЬНЫЙ АНАЛИЗ ПО КАЖДОМУ КЛАССУ")
        print("=" * 100)
        
        detailed = interpretation['detailed_analysis']
        for cls in sorted(detailed.keys(), key=lambda x: detailed[x]['severity'] == 'критическая', reverse=True):
            analysis = detailed[cls]
            severity_emoji = {'критическая': '🔴', 'высокая': '🟠', 'средняя': '🟡', 'низкая': '🟢'}
            emoji = severity_emoji.get(analysis['severity'], '•')
            
            print(f"\n{emoji} КЛАСС: {cls}")
            print(f"   Серьезность проблем: {analysis['severity'].upper()}")
            print(f"   Всего примеров: {analysis['total_samples']:,}")
            print(f"   Правильных: {analysis['correct_predictions']:,} ({100 - analysis['error_rate']:.1f}%)")
            print(f"   Ошибок: {analysis['error_count']:,} ({analysis['error_rate']:.1f}%)")
            print(f"   Precision: {analysis['metrics']['precision']:.3f} | Recall: {analysis['metrics']['recall']:.3f} | F1: {analysis['metrics']['f1']:.3f}")
            
            if analysis['issues']:
                print(f"   Проблемы: {', '.join(analysis['issues'])}")
            
            if analysis['most_confused_with']:
                print(f"   Основные путаницы:")
                for conf in analysis['most_confused_with'][:3]:
                    print(f"     → {conf['class']}: {conf['count']:,} ({conf['percentage']:.1f}%)")
            
            print(f"   📝 Интерпретация: {analysis['interpretation']}")
        
        # Паттерны ошибок
        print("\n" + "=" * 100)
        print("🔀 ПАТТЕРНЫ ОШИБОК")
        print("=" * 100)
        
        patterns = interpretation['error_patterns']
        
        if patterns['concentrated_errors']:
            print("\n📌 КОНЦЕНТРИРОВАННЫЕ ОШИБКИ (>60% в одном направлении):")
            for ce in patterns['concentrated_errors']:
                print(f"   {ce['class']} → {ce['main_confusion']}: {ce['concentration']:.1f}% ошибок ({ce['total_errors']:,} ошибок)")
                print(f"     💡 Интерпретация: Ошибки этого класса в основном идут в одно направление, что упрощает исправление")
        
        if patterns['scattered_errors']:
            print("\n🌐 РАЗБРОСАННЫЕ ОШИБКИ (<30% в любом направлении):")
            for se in patterns['scattered_errors']:
                print(f"   {se['class']}: ошибки распределены между {se['error_distribution']} классами ({se['total_errors']:,} ошибок)")
                print(f"     💡 {se['interpretation']}")
        
        if patterns['bidirectional_confusions']:
            print("\n🔄 ДВУСТОРОННИЕ ПУТАНИЦЫ:")
            for bc in patterns['bidirectional_confusions'][:10]:
                print(f"   {bc['class1']} ↔ {bc['class2']}:")
                print(f"     {bc['class1']} → {bc['class2']}: {bc['count1_to_2']:,} ({bc['pct1_to_2']:.1f}%)")
                print(f"     {bc['class2']} → {bc['class1']}: {bc['count2_to_1']:,} ({bc['pct2_to_1']:.1f}%)")
                print(f"     💡 {bc['interpretation']}")
        
        # Стабильность классов
        print("\n" + "=" * 100)
        print("📊 СТАБИЛЬНОСТЬ КЛАССОВ")
        print("=" * 100)
        print("(Высокая стабильность = ошибки концентрируются в 1-2 классах, легко исправить)")
        print("-" * 100)
        
        for stability in interpretation['class_stability']:
            stability_emoji = '🟢' if stability['stability_score'] > 0.7 else '🟡' if stability['stability_score'] > 0.4 else '🔴'
            print(f"{stability_emoji} {stability['class']}: стабильность {stability['stability_score']:.3f}")
            print(f"   Ошибки распределены между {stability['unique_error_classes']} классами")
            print(f"   💡 {stability['interpretation']}")
        
        # Кластеры путаниц
        if interpretation['confusion_clusters']:
            print("\n" + "=" * 100)
            print("🔗 КЛАСТЕРЫ ПУТАНИЦ (классы, которые путаются друг с другом)")
            print("=" * 100)
            
            for i, cluster in enumerate(interpretation['confusion_clusters'][:10], 1):
                strength_emoji = {'сильная': '🔴', 'средняя': '🟠', 'слабая': '🟡'}
                emoji = strength_emoji.get(cluster['strength'], '•')
                print(f"\n{emoji} Кластер {i}: {cluster['classes'][0]} ↔ {cluster['classes'][1]}")
                print(f"   Сила связи: {cluster['strength']}")
                print(f"   {cluster['classes'][0]} → {cluster['classes'][1]}: {cluster['count1_to_2']:,} ({cluster['pct1_to_2']:.1f}%)")
                print(f"   {cluster['classes'][1]} → {cluster['classes'][0]}: {cluster['count2_to_1']:,} ({cluster['pct2_to_1']:.1f}%)")
                print(f"   Всего путаниц: {cluster['total_confusions']:,}")
        
        # Категоризация ошибок
        if 'error_categorization' in interpretation:
            print("\n" + "=" * 100)
            print("📂 КАТЕГОРИЗАЦИЯ ОШИБОК")
            print("=" * 100)
            
            categorization = interpretation['error_categorization']
            
            # Сводка по категориям
            if categorization.get('category_summary'):
                print("\n📊 СВОДКА ПО КАТЕГОРИЯМ:")
                print("-" * 100)
                for cat_name, summary in categorization['category_summary'].items():
                    severity_emoji = {
                        'критическая': '🔴', 'высокая': '🟠', 
                        'средняя': '🟡', 'низкая': '🟢', 'информационная': '🔵'
                    }
                    emoji = severity_emoji.get(summary['severity'], '•')
                    print(f"\n{emoji} {summary['name'].replace('_', ' ').title()}")
                    print(f"   Серьезность: {summary['severity'].upper()}")
                    print(f"   Описание: {summary['description']}")
                    print(f"   Классов в категории: {summary['classes_count']}")
                    if summary['classes']:
                        print(f"   Классы: {', '.join(summary['classes'][:5])}")
                        if len(summary['classes']) > 5:
                            print(f"   ... и еще {len(summary['classes']) - 5} классов")
            
            # Распределение по классам
            if categorization.get('by_class'):
                print("\n📋 РАСПРЕДЕЛЕНИЕ ПО КЛАССАМ:")
                print("-" * 100)
                for cls, categories in categorization['by_class'].items():
                    if categories:
                        print(f"\n  {cls}:")
                        for cat_name in categories:
                            cat = self.category_manager.get_category(cat_name) if self.category_manager else None
                            if cat:
                                print(f"    • {cat_name.replace('_', ' ').title()} ({cat.severity.value})")
        
        # Рекомендации
        if interpretation['recommendations']:
            print("\n" + "=" * 100)
            print("💡 РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ")
            print("=" * 100)
            
            for i, rec in enumerate(interpretation['recommendations'], 1):
                priority_emoji = {'критичный': '🔴', 'высокий': '🟠', 'средний': '🟡', 'низкий': '🟢'}
                print(f"\n{i}. {priority_emoji.get(rec['priority'], '•')} [{rec['priority'].upper()}] {rec['type'].replace('_', ' ').title()}")
                print(f"   Описание: {rec['description']}")
                print(f"   Действие: {rec['action']}")
        
        print("\n" + "=" * 100)
    
    def save_report(self, output_file='CONFUSION_MATRIX_REPORT.md', include_visualizations=True):
        """
        Сохранение отчета в файл
        
        Args:
            output_file: путь к выходному файлу
            include_visualizations: создавать ли визуализации и добавлять ссылки в отчет
        """
        if self.confusion_matrix is None:
            print("❌ Матрица не построена")
            return
        
        metrics = self.calculate_metrics_from_matrix()
        normalized_matrix = self.get_normalized_matrix()
        mistakes = self.find_common_mistakes(top_n=15)
        
        # Создание визуализаций если нужно
        vis_paths = None
        if include_visualizations:
            try:
                vis_paths = self.create_all_visualizations(show=False)
            except Exception as e:
                print(f"⚠️ Предупреждение: не удалось создать визуализации: {e}")
                vis_paths = None
        
        report_lines = [
            "# 📊 ОТЧЕТ ПО CONFUSION MATRIX",
            "",
            f"**Дата создания**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            ""
        ]
        
        # Добавление ссылок на визуализации
        if vis_paths:
            report_lines.extend([
                "## 🎨 Визуализации",
                "",
                "Графики и диаграммы сохранены в директории `confusion_matrix_figures/`:",
                "",
                f"- **Confusion Matrix (абсолютные значения)**: `{os.path.basename(vis_paths['confusion_matrix_absolute'])}`",
                f"- **Confusion Matrix (нормализованная)**: `{os.path.basename(vis_paths['confusion_matrix_normalized'])}`",
                f"- **Сравнение метрик**: `{os.path.basename(vis_paths['metrics_comparison'])}`",
                f"- **Анализ ошибок**: `{os.path.basename(vis_paths['error_analysis'])}`",
                "",
                "---",
                ""
            ])
        
        report_lines.extend([
            "## 📊 Confusion Matrix (абсолютные значения)",
            "",
            "**Строки = Истинные классы | Колонки = Предсказанные классы**",
            "",
            "| True / Pred | " + " | ".join(self.classes) + " |",
            "|" + "|".join(["---"] * (len(self.classes) + 1)) + "|"
        ])
        
        # Абсолютная матрица
        for true_cls in self.classes:
            row_values = [str(int(self.confusion_matrix.loc[true_cls, pred_cls])) for pred_cls in self.classes]
            report_lines.append(f"| **{true_cls}** | " + " | ".join(row_values) + " |")
        
        report_lines.extend([
            "",
            "## 📊 Нормализованная Confusion Matrix (%)",
            "",
            "**Процент от истинного класса (строки = 100%)**",
            "",
            "| True / Pred | " + " | ".join(self.classes) + " |",
            "|" + "|".join(["---"] * (len(self.classes) + 1)) + "|"
        ])
        
        # Нормализованная матрица
        for true_cls in self.classes:
            row_values = [f"{normalized_matrix.loc[true_cls, pred_cls]:.1f}%" for pred_cls in self.classes]
            report_lines.append(f"| **{true_cls}** | " + " | ".join(row_values) + " |")
        
        # Метрики
        report_lines.extend([
            "",
            "## 📋 Метрики по классам",
            "",
            "| Класс | Precision | Recall | F1-Score | Support |",
            "|-------|-----------|--------|----------|---------|"
        ])
        
        for cls in self.classes:
            m = metrics[cls]
            report_lines.append(
                f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | "
                f"{m['f1']:.4f} | {m['support']} |"
            )
        
        # Общие метрики
        macro_precision = np.mean([m['precision'] for m in metrics.values()])
        macro_recall = np.mean([m['recall'] for m in metrics.values()])
        macro_f1 = np.mean([m['f1'] for m in metrics.values()])
        
        total_tp = sum(m['tp'] for m in metrics.values())
        total_fp = sum(m['fp'] for m in metrics.values())
        total_fn = sum(m['fn'] for m in metrics.values())
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        total_samples = self.confusion_matrix.values.sum()
        correct = sum(self.confusion_matrix.loc[cls, cls] for cls in self.classes)
        accuracy = (correct / total_samples * 100) if total_samples > 0 else 0
        
        report_lines.extend([
            "",
            "## 📊 Общие метрики",
            "",
            "| Метрика | Значение |",
            "|---------|----------|",
            f"| **Macro Precision** | {macro_precision:.4f} |",
            f"| **Macro Recall** | {macro_recall:.4f} |",
            f"| **Macro F1** | {macro_f1:.4f} |",
            f"| **Micro Precision** | {micro_precision:.4f} |",
            f"| **Micro Recall** | {micro_recall:.4f} |",
            f"| **Micro F1** | {micro_f1:.4f} |",
            f"| **Общая точность (Accuracy)** | {accuracy:.2f}% ({correct:,}/{total_samples:,}) |",
            "",
            "## ⚠️ Самые частые ошибки",
            "",
            "| Истинный класс | Предсказанный класс | Количество | % от класса |",
            "|----------------|---------------------|------------|-------------|"
        ])
        
        for true_cls, pred_cls, count in mistakes:
            total_true = self.confusion_matrix.loc[true_cls, :].sum()
            percentage = (count / total_true * 100) if total_true > 0 else 0
            report_lines.append(f"| {true_cls} | {pred_cls} | {count} | {percentage:.2f}% |")
        
        # Интерпретация ошибок
        interpretation = self.interpret_errors()
        
        report_lines.extend([
            "",
            "## 🔍 ИНТЕРПРЕТАЦИЯ ОШИБОК",
            ""
        ])
        
        # Проблемные классы
        if interpretation['problematic_classes']:
            report_lines.extend([
                "### ⚠️ Проблемные классы (более 50% ошибок)",
                ""
            ])
            for pc in interpretation['problematic_classes']:
                report_lines.append(f"#### 📌 {pc['class']}")
                report_lines.append(f"- **Ошибок**: {pc['error_rate']}% ({pc['total'] - pc['correct']:,} из {pc['total']:,})")
                report_lines.append(f"- **Правильных**: {pc['correct']:,} ({100 - pc['error_rate']:.1f}%)")
                if pc['main_confusions']:
                    report_lines.append("- **Основные путаницы**:")
                    for conf in pc['main_confusions']:
                        report_lines.append(f"  - `{conf['confused_with']}`: {conf['count']:,} ({conf['percentage']}%)")
                report_lines.append("")
        
        # Классы с низкой производительностью
        if interpretation['low_performance_classes']:
            report_lines.extend([
                "### 📉 Классы с низкой производительностью",
                "",
                "| Класс | Precision | Recall | F1-Score | Проблемы |",
                "|-------|-----------|--------|----------|---------|"
            ])
            for lpc in interpretation['low_performance_classes']:
                issues_str = "; ".join(lpc['issues'])
                report_lines.append(
                    f"| {lpc['class']} | {lpc['precision']:.2%} | {lpc['recall']:.2%} | "
                    f"{lpc['f1']:.2%} | {issues_str} |"
                )
            report_lines.append("")
        
        # Симметричные ошибки
        if interpretation['symmetric_errors']:
            report_lines.extend([
                "### 🔄 Симметричные ошибки (классы путаются друг с другом)",
                "",
                "**Интерпретация**: Эти классы имеют перекрывающиеся признаки",
                "",
                "| Класс 1 | Класс 2 | 1→2 ошибок | 2→1 ошибок | Всего |",
                "|---------|---------|------------|------------|-------|"
            ])
            for se in interpretation['symmetric_errors']:
                report_lines.append(
                    f"| {se['class1']} | {se['class2']} | {se['count1_to_2']:,} | "
                    f"{se['count2_to_1']:,} | {se['total_mistakes']:,} |"
                )
            report_lines.append("")
        
        # Доминирующие ошибки
        if interpretation['dominant_confusions']:
            report_lines.extend([
                "### 📊 Доминирующие ошибки (более 20% от класса)",
                "",
                "| Истинный класс | Предсказанный класс | Количество | % от класса | Серьезность |",
                "|----------------|---------------------|------------|-------------|--------------|"
            ])
            for dc in sorted(interpretation['dominant_confusions'], key=lambda x: x['percentage'], reverse=True):
                report_lines.append(
                    f"| {dc['true_class']} | {dc['predicted_class']} | {dc['count']:,} | "
                    f"{dc['percentage']}% | {dc['severity']} |"
                )
            report_lines.append("")
        
        # Полная интерпретация ошибок
        report_lines.extend([
            "",
            "## 🔍 ПОЛНАЯ ИНТЕРПРЕТАЦИЯ ОШИБОК",
            ""
        ])
        
        # Статистика ошибок
        stats = interpretation['error_statistics']
        report_lines.extend([
            "### 📊 Общая статистика ошибок",
            "",
            f"- **Всего примеров**: {stats['total_samples']:,}",
            f"- **Правильных предсказаний**: {stats['total_correct']:,} ({100 - stats['overall_error_rate']:.2f}%)",
            f"- **Ошибок**: {stats['total_errors']:,} ({stats['overall_error_rate']:.2f}%)",
            f"- **Средний процент ошибок по классам**: {stats['average_error_rate_per_class']:.2f}% (σ={stats['error_rate_std']:.2f}%)",
            ""
        ])
        
        if stats['top_error_contributors']:
            report_lines.extend([
                "#### 🔴 Топ-5 классов с наибольшим вкладом в ошибки:",
                "",
                "| Класс | Количество ошибок | % от всех ошибок |",
                "|-------|-------------------|------------------|"
            ])
            for contrib in stats['top_error_contributors']:
                report_lines.append(f"| {contrib['class']} | {contrib['errors']:,} | {contrib['contribution_pct']:.1f}% |")
            report_lines.append("")
        
        # Детальный анализ по классам
        detailed = interpretation['detailed_analysis']
        report_lines.extend([
            "### 📋 Детальный анализ по каждому классу",
            ""
        ])
        
        for cls in sorted(detailed.keys(), key=lambda x: detailed[x]['severity'] == 'критическая', reverse=True):
            analysis = detailed[cls]
            severity_emoji = {'критическая': '🔴', 'высокая': '🟠', 'средняя': '🟡', 'низкая': '🟢'}
            emoji = severity_emoji.get(analysis['severity'], '•')
            
            report_lines.extend([
                f"#### {emoji} {cls}",
                "",
                f"- **Серьезность проблем**: {analysis['severity'].upper()}",
                f"- **Всего примеров**: {analysis['total_samples']:,}",
                f"- **Правильных**: {analysis['correct_predictions']:,} ({100 - analysis['error_rate']:.1f}%)",
                f"- **Ошибок**: {analysis['error_count']:,} ({analysis['error_rate']:.1f}%)",
                f"- **Precision**: {analysis['metrics']['precision']:.3f} | **Recall**: {analysis['metrics']['recall']:.3f} | **F1**: {analysis['metrics']['f1']:.3f}",
                ""
            ])
            
            if analysis['issues']:
                report_lines.append(f"- **Проблемы**: {', '.join(analysis['issues'])}")
                report_lines.append("")
            
            if analysis['most_confused_with']:
                report_lines.append("- **Основные путаницы**:")
                for conf in analysis['most_confused_with'][:3]:
                    report_lines.append(f"  - `{conf['class']}`: {conf['count']:,} ({conf['percentage']:.1f}%)")
                report_lines.append("")
            
            report_lines.extend([
                f"**Интерпретация**: {analysis['interpretation']}",
                ""
            ])
        
        # Паттерны ошибок
        patterns = interpretation['error_patterns']
        report_lines.extend([
            "### 🔀 Паттерны ошибок",
            ""
        ])
        
        if patterns['concentrated_errors']:
            report_lines.extend([
                "#### 📌 Концентрированные ошибки (>60% в одном направлении)",
                "",
                "| Класс | Основная путаница | Концентрация | Всего ошибок |",
                "|-------|-------------------|--------------|--------------|"
            ])
            for ce in patterns['concentrated_errors']:
                report_lines.append(
                    f"| {ce['class']} | {ce['main_confusion']} | {ce['concentration']:.1f}% | {ce['total_errors']:,} |"
                )
            report_lines.append("")
            report_lines.append("*Ошибки этих классов в основном идут в одно направление, что упрощает исправление*")
            report_lines.append("")
        
        if patterns['scattered_errors']:
            report_lines.extend([
                "#### 🌐 Разбросанные ошибки (<30% в любом направлении)",
                "",
                "| Класс | Количество классов с ошибками | Всего ошибок |",
                "|-------|-------------------------------|--------------|"
            ])
            for se in patterns['scattered_errors']:
                report_lines.append(
                    f"| {se['class']} | {se['error_distribution']} | {se['total_errors']:,} |"
                )
            report_lines.append("")
            report_lines.append("*Ошибки распределены между многими классами, что указывает на общую неопределенность модели*")
            report_lines.append("")
        
        if patterns['bidirectional_confusions']:
            report_lines.extend([
                "#### 🔄 Двусторонние путаницы",
                "",
                "| Класс 1 | Класс 2 | 1→2 ошибок | 2→1 ошибок | Интерпретация |",
                "|---------|---------|------------|------------|---------------|"
            ])
            for bc in patterns['bidirectional_confusions'][:10]:
                report_lines.append(
                    f"| {bc['class1']} | {bc['class2']} | {bc['count1_to_2']:,} ({bc['pct1_to_2']:.1f}%) | "
                    f"{bc['count2_to_1']:,} ({bc['pct2_to_1']:.1f}%) | {bc['interpretation']} |"
                )
            report_lines.append("")
        
        # Стабильность классов
        report_lines.extend([
            "### 📊 Стабильность классов",
            "",
            "*(Высокая стабильность = ошибки концентрируются в 1-2 классах, легко исправить)*",
            "",
            "| Класс | Стабильность | Классов с ошибками | Интерпретация |",
            "|-------|--------------|-------------------|---------------|"
        ])
        
        for stability in interpretation['class_stability']:
            report_lines.append(
                f"| {stability['class']} | {stability['stability_score']:.3f} | {stability['unique_error_classes']} | {stability['interpretation']} |"
            )
        report_lines.append("")
        
        # Кластеры путаниц
        if interpretation['confusion_clusters']:
            report_lines.extend([
                "### 🔗 Кластеры путаниц",
                "",
                "*(Классы, которые путаются друг с другом)*",
                "",
                "| Класс 1 | Класс 2 | 1→2 | 2→1 | Всего | Сила связи |",
                "|---------|---------|-----|-----|-------|------------|"
            ])
            for cluster in interpretation['confusion_clusters'][:15]:
                report_lines.append(
                    f"| {cluster['classes'][0]} | {cluster['classes'][1]} | "
                    f"{cluster['count1_to_2']:,} ({cluster['pct1_to_2']:.1f}%) | "
                    f"{cluster['count2_to_1']:,} ({cluster['pct2_to_1']:.1f}%) | "
                    f"{cluster['total_confusions']:,} | {cluster['strength']} |"
                )
            report_lines.append("")
        
        # Категоризация ошибок
        if 'error_categorization' in interpretation:
            categorization = interpretation['error_categorization']
            
            report_lines.extend([
                "",
                "## 📂 КАТЕГОРИЗАЦИЯ ОШИБОК",
                ""
            ])
            
            # Сводка по категориям
            if categorization.get('category_summary'):
                report_lines.extend([
                    "### 📊 Сводка по категориям",
                    "",
                    "| Категория | Серьезность | Классов | Описание |",
                    "|-----------|-------------|---------|----------|"
                ])
                
                for cat_name, summary in categorization['category_summary'].items():
                    classes_str = ', '.join(summary['classes'][:3])
                    if len(summary['classes']) > 3:
                        classes_str += f" (+{len(summary['classes']) - 3})"
                    report_lines.append(
                        f"| {summary['name'].replace('_', ' ').title()} | {summary['severity']} | "
                        f"{summary['classes_count']} | {summary['description']} |"
                    )
                report_lines.append("")
                
                # Детали по категориям
                for cat_name, summary in categorization['category_summary'].items():
                    if summary['classes_count'] > 0:
                        report_lines.extend([
                            f"#### {summary['name'].replace('_', ' ').title()}",
                            "",
                            f"**Серьезность**: {summary['severity']}",
                            f"**Классов в категории**: {summary['classes_count']}",
                            f"**Классы**: {', '.join(summary['classes'])}",
                            "",
                            "**Рекомендации**:",
                            ""
                        ])
                        for rec in summary['recommendations']:
                            report_lines.append(f"- {rec}")
            report_lines.append("")
        
        # Рекомендации
        if interpretation['recommendations']:
            report_lines.extend([
                "### 💡 Рекомендации по улучшению",
                ""
            ])
            for i, rec in enumerate(interpretation['recommendations'], 1):
                report_lines.extend([
                    f"#### {i}. [{rec['priority'].upper()}] {rec['type'].replace('_', ' ').title()}",
                    "",
                    f"**Описание**: {rec['description']}",
                    "",
                    f"**Действие**: {rec['action']}",
                    ""
                ])
        
        report_lines.extend([
            "",
            f"**Всего примеров**: {total_samples:,}",
            f"**Всего классов**: {len(self.classes)}",
            ""
        ])
        
        # Добавление информации о визуализациях в конец
        if vis_paths:
            report_lines.extend([
                "",
                "---",
                "",
                "## 📁 Файлы визуализаций",
                "",
                "Все графики сохранены в высоком разрешении (300 DPI) и доступны в директории с отчетами."
        ])
        
        # Сохранение
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ Отчет сохранен: {output_file}")
        
        # Сохранение категорий ошибок если есть
        if self.category_manager:
            categories_file = output_file.replace('.md', '_categories.json')
            self.category_manager.save_categories(categories_file)
            print(f"✅ Категории ошибок сохранены: {categories_file}")
        
        # Сохранение JSON
        json_data = {
            'confusion_matrix': self.confusion_matrix.to_dict(),
            'normalized_matrix': normalized_matrix.to_dict(),
            'metrics': metrics,
            'common_mistakes': [{'true_class': t, 'pred_class': p, 'count': c} for t, p, c in mistakes],
            'interpretation': interpretation,
            'overall_metrics': {
                'macro_precision': float(macro_precision),
                'macro_recall': float(macro_recall),
                'macro_f1': float(macro_f1),
                'micro_precision': float(micro_precision),
                'micro_recall': float(micro_recall),
                'micro_f1': float(micro_f1),
                'accuracy': float(accuracy),
                'total_samples': int(total_samples),
                'correct_predictions': int(correct)
            }
        }
        
        json_file = output_file.replace('.md', '.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ JSON данные сохранены: {json_file}")
        
        # Сохранение CSV матрицы
        csv_file = output_file.replace('.md', '.csv')
        self.confusion_matrix.to_csv(csv_file)
        print(f"✅ CSV матрица сохранена: {csv_file}")
    
    def save_html_report(self, output_file='CONFUSION_MATRIX_REPORT.html', include_visualizations=True):
        """
        Сохранение полного HTML-отчета с интерпретацией ошибок для веб-просмотра
        
        Args:
            output_file: путь к выходному HTML файлу
            include_visualizations: создавать ли визуализации и добавлять ссылки
        """
        if self.confusion_matrix is None:
            print("❌ Матрица не построена")
            return
        
        metrics = self.calculate_metrics_from_matrix()
        normalized_matrix = self.get_normalized_matrix()
        mistakes = self.find_common_mistakes(top_n=20)
        interpretation = self.interpret_errors()
        
        # Создание визуализаций если нужно
        vis_paths = None
        if include_visualizations:
            try:
                vis_paths = self.create_all_visualizations(show=False)
            except Exception as e:
                print(f"⚠️ Предупреждение: не удалось создать визуализации: {e}")
                vis_paths = None
        
        # Расчет общих метрик
        macro_precision = np.mean([m['precision'] for m in metrics.values()])
        macro_recall = np.mean([m['recall'] for m in metrics.values()])
        macro_f1 = np.mean([m['f1'] for m in metrics.values()])
        
        total_tp = sum(m['tp'] for m in metrics.values())
        total_fp = sum(m['fp'] for m in metrics.values())
        total_fn = sum(m['fn'] for m in metrics.values())
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        total_samples = self.confusion_matrix.values.sum()
        correct = sum(self.confusion_matrix.loc[cls, cls] for cls in self.classes)
        accuracy = (correct / total_samples * 100) if total_samples > 0 else 0
        
        # Генерация HTML
        html_content = self._generate_html_content(
            metrics, normalized_matrix, mistakes, interpretation,
            macro_precision, macro_recall, macro_f1,
            micro_precision, micro_recall, micro_f1,
            accuracy, total_samples, correct, vis_paths
        )
        
        # Сохранение HTML
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML отчет сохранен: {output_file}")
        print(f"📂 Откройте файл в браузере для просмотра полной интерпретации ошибок")
        
        return output_file
    
    def _generate_html_content(self, metrics, normalized_matrix, mistakes, interpretation,
                               macro_precision, macro_recall, macro_f1,
                               micro_precision, micro_recall, micro_f1,
                               accuracy, total_samples, correct, vis_paths):
        """Генерация HTML контента"""
        
        # CSS стили
        css_styles = """
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                padding: 40px;
            }
            h1 {
                color: #667eea;
                text-align: center;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
                font-size: 1.1em;
            }
            h2 {
                color: #764ba2;
                margin-top: 40px;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 3px solid #667eea;
                font-size: 1.8em;
            }
            h3 {
                color: #555;
                margin-top: 30px;
                margin-bottom: 15px;
                font-size: 1.4em;
            }
            h4 {
                color: #666;
                margin-top: 20px;
                margin-bottom: 10px;
                font-size: 1.2em;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }
            .stat-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            .stat-card h3 {
                color: white;
                margin: 0 0 10px 0;
                font-size: 1.1em;
            }
            .stat-value {
                font-size: 2.5em;
                font-weight: bold;
                margin: 10px 0;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            th {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                text-align: left;
                font-weight: 600;
            }
            td {
                padding: 12px 15px;
                border-bottom: 1px solid #eee;
            }
            tr:hover {
                background: #f8f9fa;
            }
            .severity-critical { background-color: #ffebee; border-left: 4px solid #f44336; }
            .severity-high { background-color: #fff3e0; border-left: 4px solid #ff9800; }
            .severity-medium { background-color: #fff9c4; border-left: 4px solid #ffc107; }
            .severity-low { background-color: #e8f5e9; border-left: 4px solid #4caf50; }
            .badge {
                display: inline-block;
                padding: 5px 12px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: 600;
                margin: 2px;
            }
            .badge-critical { background: #f44336; color: white; }
            .badge-high { background: #ff9800; color: white; }
            .badge-medium { background: #ffc107; color: white; }
            .badge-low { background: #4caf50; color: white; }
            .interpretation-box {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin: 15px 0;
                border-left: 4px solid #667eea;
            }
            .pattern-card {
                background: white;
                padding: 20px;
                margin: 15px 0;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                border-left: 4px solid #667eea;
            }
            .cluster-item {
                background: #f8f9fa;
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
                border-left: 4px solid #764ba2;
            }
            .recommendation {
                background: #e3f2fd;
                padding: 20px;
                margin: 15px 0;
                border-radius: 8px;
                border-left: 4px solid #2196f3;
            }
            .metric-bar {
                height: 25px;
                background: #e0e0e0;
                border-radius: 12px;
                margin: 5px 0;
                overflow: hidden;
            }
            .metric-fill {
                height: 100%;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: 600;
                font-size: 0.9em;
            }
            .nav-menu {
                position: sticky;
                top: 20px;
                background: white;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 30px;
            }
            .nav-menu a {
                display: inline-block;
                padding: 8px 15px;
                margin: 5px;
                background: #667eea;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                transition: background 0.3s;
            }
            .nav-menu a:hover {
                background: #764ba2;
            }
            .section {
                scroll-margin-top: 100px;
            }
            @media print {
                body { background: white; }
                .container { box-shadow: none; }
                .nav-menu { display: none; }
            }
        </style>
        """
        
        # JavaScript для интерактивности
        js_script = """
        <script>
            function scrollToSection(id) {
                document.getElementById(id).scrollIntoView({ behavior: 'smooth' });
            }
            function toggleSection(id) {
                const element = document.getElementById(id);
                if (element.style.display === 'none') {
                    element.style.display = 'block';
                } else {
                    element.style.display = 'none';
                }
            }
        </script>
        """
        
        # Навигационное меню
        nav_menu = """
        <div class="nav-menu">
            <a href="#overview">Обзор</a>
            <a href="#matrix">Матрица</a>
            <a href="#metrics">Метрики</a>
            <a href="#detailed">Детальный анализ</a>
            <a href="#patterns">Паттерны</a>
            <a href="#stability">Стабильность</a>
            <a href="#clusters">Кластеры</a>
            <a href="#categorization">Категории</a>
            <a href="#recommendations">Рекомендации</a>
        </div>
        """
        
        # Начало HTML
        html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Полная интерпретация ошибок - Confusion Matrix</title>
    {css_styles}
</head>
<body>
    <div class="container">
        <h1>📊 Полная интерпретация ошибок классификации</h1>
        <div class="subtitle">Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        
        {nav_menu}
        
        <!-- ОБЗОР -->
        <section id="overview" class="section">
            <h2>📈 Общая статистика</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Всего примеров</h3>
                    <div class="stat-value">{total_samples:,}</div>
                </div>
                <div class="stat-card">
                    <h3>Правильных</h3>
                    <div class="stat-value">{correct:,}</div>
                    <div>({100-accuracy:.2f}%)</div>
                </div>
                <div class="stat-card">
                    <h3>Ошибок</h3>
                    <div class="stat-value">{total_samples-correct:,}</div>
                    <div>({(total_samples-correct)/total_samples*100:.2f}%)</div>
                </div>
                <div class="stat-card">
                    <h3>Общая точность</h3>
                    <div class="stat-value">{accuracy:.2f}%</div>
                </div>
            </div>
        </section>
        
        <!-- CONFUSION MATRIX -->
        <section id="matrix" class="section">
            <h2>📊 Confusion Matrix</h2>
            {self._generate_matrix_html_table()}
            <h3>Нормализованная матрица (%)</h3>
            {self._generate_normalized_matrix_html_table(normalized_matrix)}
        </section>
        
        <!-- МЕТРИКИ -->
        <section id="metrics" class="section">
            <h2>📋 Метрики по классам</h2>
            {self._generate_metrics_html_table(metrics)}
            
            <h3>Общие метрики</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Macro Precision</h3>
                    <div class="stat-value">{macro_precision:.4f}</div>
                </div>
                <div class="stat-card">
                    <h3>Macro Recall</h3>
                    <div class="stat-value">{macro_recall:.4f}</div>
                </div>
                <div class="stat-card">
                    <h3>Macro F1</h3>
                    <div class="stat-value">{macro_f1:.4f}</div>
                </div>
                <div class="stat-card">
                    <h3>Micro F1</h3>
                    <div class="stat-value">{micro_f1:.4f}</div>
                </div>
            </div>
        </section>
        
        <!-- ДЕТАЛЬНЫЙ АНАЛИЗ -->
        <section id="detailed" class="section">
            <h2>🔍 Детальный анализ по каждому классу</h2>
            {self._generate_detailed_analysis_html(interpretation['detailed_analysis'])}
        </section>
        
        <!-- ПАТТЕРНЫ ОШИБОК -->
        <section id="patterns" class="section">
            <h2>🔀 Паттерны ошибок</h2>
            {self._generate_patterns_html(interpretation['error_patterns'])}
        </section>
        
        <!-- СТАБИЛЬНОСТЬ -->
        <section id="stability" class="section">
            <h2>📊 Стабильность классов</h2>
            {self._generate_stability_html(interpretation['class_stability'])}
        </section>
        
        <!-- КЛАСТЕРЫ -->
        <section id="clusters" class="section">
            <h2>🔗 Кластеры путаниц</h2>
            {self._generate_clusters_html(interpretation['confusion_clusters'])}
        </section>
        
        <!-- КАТЕГОРИЗАЦИЯ ОШИБОК -->
        {self._generate_categorization_html(interpretation.get('error_categorization', {})) if 'error_categorization' in interpretation else ''}
        
        <!-- РЕКОМЕНДАЦИИ -->
        <section id="recommendations" class="section">
            <h2>💡 Рекомендации по улучшению</h2>
            {self._generate_recommendations_html(interpretation['recommendations'])}
        </section>
        
        <!-- ВИЗУАЛИЗАЦИИ -->
        {self._generate_visualizations_section(vis_paths) if vis_paths else ''}
        
    </div>
    {js_script}
</body>
</html>"""
        
        return html
    
    def _generate_matrix_html_table(self):
        """Генерация HTML таблицы для confusion matrix"""
        html = '<table><thead><tr><th>True / Pred</th>'
        for cls in self.classes:
            html += f'<th>{cls}</th>'
        html += '</tr></thead><tbody>'
        
        for true_cls in self.classes:
            html += f'<tr><th>{true_cls}</th>'
            for pred_cls in self.classes:
                value = int(self.confusion_matrix.loc[true_cls, pred_cls])
                # Подсветка диагонали
                if true_cls == pred_cls:
                    html += f'<td style="background-color: #c8e6c9; font-weight: bold;">{value:,}</td>'
                else:
                    html += f'<td>{value:,}</td>'
            html += '</tr>'
        
        html += '</tbody></table>'
        return html
    
    def _generate_normalized_matrix_html_table(self, normalized_matrix):
        """Генерация HTML таблицы для нормализованной матрицы"""
        html = '<table><thead><tr><th>True / Pred</th>'
        for cls in self.classes:
            html += f'<th>{cls}</th>'
        html += '</tr></thead><tbody>'
        
        for true_cls in self.classes:
            html += f'<tr><th>{true_cls}</th>'
            for pred_cls in self.classes:
                value = normalized_matrix.loc[true_cls, pred_cls]
                # Цветовая индикация
                if true_cls == pred_cls:
                    color = '#c8e6c9' if value > 50 else '#fff9c4' if value > 30 else '#ffccbc'
                    html += f'<td style="background-color: {color}; font-weight: bold;">{value:.1f}%</td>'
                else:
                    color = '#ffccbc' if value > 20 else '#fff9c4' if value > 10 else '#f5f5f5'
                    html += f'<td style="background-color: {color};">{value:.1f}%</td>'
            html += '</tr>'
        
        html += '</tbody></table>'
        return html
    
    def _generate_metrics_html_table(self, metrics):
        """Генерация HTML таблицы метрик"""
        html = '<table><thead><tr><th>Класс</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead><tbody>'
        
        for cls in self.classes:
            m = metrics[cls]
            # Определение класса серьезности
            if m['f1'] < 0.5:
                severity_class = 'severity-critical'
            elif m['f1'] < 0.7:
                severity_class = 'severity-high'
            elif m['f1'] < 0.85:
                severity_class = 'severity-medium'
            else:
                severity_class = 'severity-low'
            
            html += f'<tr class="{severity_class}">'
            html += f'<td><strong>{cls}</strong></td>'
            html += f'<td>{m["precision"]:.4f}</td>'
            html += f'<td>{m["recall"]:.4f}</td>'
            html += f'<td><strong>{m["f1"]:.4f}</strong></td>'
            html += f'<td>{m["support"]:,}</td>'
            html += '</tr>'
        
        html += '</tbody></table>'
        return html
    
    def _generate_detailed_analysis_html(self, detailed_analysis):
        """Генерация HTML для детального анализа"""
        html = ''
        
        for cls in sorted(detailed_analysis.keys(), 
                         key=lambda x: detailed_analysis[x]['severity'] == 'критическая', 
                         reverse=True):
            analysis = detailed_analysis[cls]
            severity_class = {
                'критическая': 'severity-critical',
                'высокая': 'severity-high',
                'средняя': 'severity-medium',
                'низкая': 'severity-low'
            }.get(analysis['severity'], '')
            
            html += f'<div class="pattern-card {severity_class}">'
            html += f'<h4>{cls}</h4>'
            html += f'<p><strong>Серьезность:</strong> <span class="badge badge-{analysis["severity"]}">{analysis["severity"].upper()}</span></p>'
            html += f'<p><strong>Всего примеров:</strong> {analysis["total_samples"]:,} | '
            html += f'<strong>Правильных:</strong> {analysis["correct_predictions"]:,} ({100-analysis["error_rate"]:.1f}%) | '
            html += f'<strong>Ошибок:</strong> {analysis["error_count"]:,} ({analysis["error_rate"]:.1f}%)</p>'
            
            html += '<div class="metric-bar"><div class="metric-fill" style="width: ' + str(analysis['metrics']['precision']*100) + '%">Precision: ' + f'{analysis["metrics"]["precision"]:.3f}' + '</div></div>'
            html += '<div class="metric-bar"><div class="metric-fill" style="width: ' + str(analysis['metrics']['recall']*100) + '%">Recall: ' + f'{analysis["metrics"]["recall"]:.3f}' + '</div></div>'
            html += '<div class="metric-bar"><div class="metric-fill" style="width: ' + str(analysis['metrics']['f1']*100) + '%">F1: ' + f'{analysis["metrics"]["f1"]:.3f}' + '</div></div>'
            
            if analysis['most_confused_with']:
                html += '<p><strong>Основные путаницы:</strong></p><ul>'
                for conf in analysis['most_confused_with'][:3]:
                    html += f'<li><code>{conf["class"]}</code>: {conf["count"]:,} ({conf["percentage"]:.1f}%)</li>'
                html += '</ul>'
            
            html += f'<div class="interpretation-box"><strong>📝 Интерпретация:</strong><br>{analysis["interpretation"]}</div>'
            html += '</div>'
        
        return html
    
    def _generate_patterns_html(self, patterns):
        """Генерация HTML для паттернов ошибок"""
        html = ''
        
        if patterns['concentrated_errors']:
            html += '<h3>📌 Концентрированные ошибки (>60% в одном направлении)</h3>'
            for ce in patterns['concentrated_errors']:
                html += f'<div class="pattern-card"><p><strong>{ce["class"]}</strong> → <strong>{ce["main_confusion"]}</strong></p>'
                html += f'<p>Концентрация: {ce["concentration"]:.1f}% | Всего ошибок: {ce["total_errors"]:,}</p>'
                html += '<p><em>Ошибки этого класса в основном идут в одно направление, что упрощает исправление</em></p></div>'
        
        if patterns['scattered_errors']:
            html += '<h3>🌐 Разбросанные ошибки (<30% в любом направлении)</h3>'
            for se in patterns['scattered_errors']:
                html += f'<div class="pattern-card"><p><strong>{se["class"]}</strong></p>'
                html += f'<p>Ошибки распределены между {se["error_distribution"]} классами | Всего ошибок: {se["total_errors"]:,}</p>'
                html += f'<p><em>{se["interpretation"]}</em></p></div>'
        
        if patterns['bidirectional_confusions']:
            html += '<h3>🔄 Двусторонние путаницы</h3>'
            html += '<table><thead><tr><th>Класс 1</th><th>Класс 2</th><th>1→2</th><th>2→1</th><th>Интерпретация</th></tr></thead><tbody>'
            for bc in patterns['bidirectional_confusions'][:10]:
                html += f'<tr><td>{bc["class1"]}</td><td>{bc["class2"]}</td>'
                html += f'<td>{bc["count1_to_2"]:,} ({bc["pct1_to_2"]:.1f}%)</td>'
                html += f'<td>{bc["count2_to_1"]:,} ({bc["pct2_to_1"]:.1f}%)</td>'
                html += f'<td>{bc["interpretation"]}</td></tr>'
            html += '</tbody></table>'
        
        return html
    
    def _generate_stability_html(self, stability):
        """Генерация HTML для стабильности классов"""
        html = '<table><thead><tr><th>Класс</th><th>Стабильность</th><th>Классов с ошибками</th><th>Интерпретация</th></tr></thead><tbody>'
        
        for s in stability:
            stability_color = '#4caf50' if s['stability_score'] > 0.7 else '#ff9800' if s['stability_score'] > 0.4 else '#f44336'
            html += f'<tr><td><strong>{s["class"]}</strong></td>'
            html += f'<td><span style="color: {stability_color}; font-weight: bold;">{s["stability_score"]:.3f}</span></td>'
            html += f'<td>{s["unique_error_classes"]}</td>'
            html += f'<td>{s["interpretation"]}</td></tr>'
        
        html += '</tbody></table>'
        return html
    
    def _generate_clusters_html(self, clusters):
        """Генерация HTML для кластеров путаниц"""
        if not clusters:
            return '<p>Кластеры путаниц не обнаружены.</p>'
        
        html = ''
        for i, cluster in enumerate(clusters[:15], 1):
            strength_color = {'сильная': '#f44336', 'средняя': '#ff9800', 'слабая': '#ffc107'}.get(cluster['strength'], '#666')
            html += f'<div class="cluster-item">'
            html += f'<h4>Кластер {i}: {cluster["classes"][0]} ↔ {cluster["classes"][1]}</h4>'
            html += f'<p><strong>Сила связи:</strong> <span style="color: {strength_color}; font-weight: bold;">{cluster["strength"]}</span></p>'
            html += f'<p>{cluster["classes"][0]} → {cluster["classes"][1]}: {cluster["count1_to_2"]:,} ({cluster["pct1_to_2"]:.1f}%)</p>'
            html += f'<p>{cluster["classes"][1]} → {cluster["classes"][0]}: {cluster["count2_to_1"]:,} ({cluster["pct2_to_1"]:.1f}%)</p>'
            html += f'<p><strong>Всего путаниц:</strong> {cluster["total_confusions"]:,}</p>'
            html += '</div>'
        
        return html
    
    def _generate_recommendations_html(self, recommendations):
        """Генерация HTML для рекомендаций"""
        if not recommendations:
            return '<p>Рекомендации отсутствуют.</p>'
        
        html = ''
        for i, rec in enumerate(recommendations, 1):
            priority_color = {'критичный': '#f44336', 'высокий': '#ff9800', 'средний': '#ffc107', 'низкий': '#4caf50'}.get(rec['priority'], '#666')
            html += f'<div class="recommendation">'
            html += f'<h4>{i}. [{rec["priority"].upper()}] {rec["type"].replace("_", " ").title()}</h4>'
            html += f'<p><strong>Описание:</strong> {rec["description"]}</p>'
            html += f'<p><strong>Действие:</strong> {rec["action"]}</p>'
            html += '</div>'
        
        return html
    
    def _generate_categorization_html(self, categorization):
        """Генерация HTML для категоризации ошибок"""
        if not categorization or not categorization.get('category_summary'):
            return ''
        
        html = '<section id="categorization" class="section"><h2>📂 Категоризация ошибок</h2>'
        
        # Сводка по категориям
        html += '<h3>📊 Сводка по категориям</h3>'
        html += '<table><thead><tr><th>Категория</th><th>Серьезность</th><th>Классов</th><th>Описание</th></tr></thead><tbody>'
        
        for cat_name, summary in categorization['category_summary'].items():
            severity_color = {
                'критическая': '#f44336', 'высокая': '#ff9800',
                'средняя': '#ffc107', 'низкая': '#4caf50', 'информационная': '#2196f3'
            }.get(summary['severity'], '#666')
            
            html += f'<tr>'
            html += f'<td><strong>{summary["name"].replace("_", " ").title()}</strong></td>'
            html += f'<td><span style="color: {severity_color}; font-weight: bold;">{summary["severity"]}</span></td>'
            html += f'<td>{summary["classes_count"]}</td>'
            html += f'<td>{summary["description"]}</td>'
            html += f'</tr>'
        
        html += '</tbody></table>'
        
        # Детали по категориям
        html += '<h3>📋 Детали по категориям</h3>'
        for cat_name, summary in categorization['category_summary'].items():
            if summary['classes_count'] > 0:
                severity_color = {
                    'критическая': '#f44336', 'высокая': '#ff9800',
                    'средняя': '#ffc107', 'низкая': '#4caf50', 'информационная': '#2196f3'
                }.get(summary['severity'], '#666')
                
                html += f'<div class="pattern-card" style="border-left-color: {severity_color};">'
                html += f'<h4>{summary["name"].replace("_", " ").title()}</h4>'
                html += f'<p><strong>Серьезность:</strong> <span style="color: {severity_color}; font-weight: bold;">{summary["severity"]}</span></p>'
                html += f'<p><strong>Описание:</strong> {summary["description"]}</p>'
                html += f'<p><strong>Классов в категории:</strong> {summary["classes_count"]}</p>'
                html += f'<p><strong>Классы:</strong> {", ".join(summary["classes"])}</p>'
                
                if summary['recommendations']:
                    html += '<p><strong>Рекомендации:</strong></p><ul>'
                    for rec in summary['recommendations']:
                        html += f'<li>{rec}</li>'
                    html += '</ul>'
                
                html += '</div>'
        
        html += '</section>'
        return html
    
    def _generate_visualizations_section(self, vis_paths):
        """Генерация секции с визуализациями"""
        if not vis_paths:
            return ''
        
        html = '<section id="visualizations" class="section"><h2>🎨 Визуализации</h2>'
        html += '<p>Графики сохранены в директории <code>confusion_matrix_figures/</code>:</p><ul>'
        for name, path in vis_paths.items():
            filename = os.path.basename(path)
            html += f'<li><strong>{name.replace("_", " ").title()}:</strong> <code>{filename}</code></li>'
        html += '</ul></section>'
        return html
    
    def plot_confusion_matrix(self, normalized=False, figsize=None, save_path=None, show=True):
        """
        Визуализация confusion matrix
        
        Args:
            normalized: если True, показывает нормализованную матрицу
            figsize: размер фигуры (ширина, высота)
            save_path: путь для сохранения графика
            show: показывать ли график
        """
        if not HAS_VISUALIZATION:
            raise ImportError("Для визуализации необходимо установить matplotlib и seaborn: pip install matplotlib seaborn")
        
        if self.confusion_matrix is None:
            raise ValueError("Сначала нужно построить confusion matrix")
        
        # Определение данных для визуализации
        if normalized:
            matrix_data = self.get_normalized_matrix()
            title = "Нормализованная Confusion Matrix (%)"
            fmt = '.1f'
            annot_kws = {'fontsize': 8}
        else:
            matrix_data = self.confusion_matrix
            title = "Confusion Matrix (абсолютные значения)"
            fmt = 'd'
            annot_kws = {'fontsize': 9}
        
        # Определение размера фигуры
        n_classes = len(self.classes)
        if figsize is None:
            base_size = max(10, n_classes * 0.8)
            figsize = (base_size, base_size * 0.9)
        
        # Создание фигуры
        fig, ax = plt.subplots(figsize=figsize)
        
        # Создание heatmap
        sns.heatmap(
            matrix_data,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            cbar_kws={'label': 'Процент' if normalized else 'Количество'},
            square=True,
            linewidths=0.5,
            linecolor='gray',
            ax=ax,
            annot_kws=annot_kws,
            vmin=0
        )
        
        # Настройка заголовков и меток
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Предсказанный класс', fontsize=12, fontweight='bold')
        ax.set_ylabel('Истинный класс', fontsize=12, fontweight='bold')
        
        # Поворот меток для лучшей читаемости
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Добавление информации о точности
        diagonal_sum = sum(self.confusion_matrix.loc[cls, cls] for cls in self.classes)
        total = self.confusion_matrix.values.sum()
        accuracy = (diagonal_sum / total * 100) if total > 0 else 0
        
        # Добавление текста с точностью
        fig.text(0.5, 0.02, f'Общая точность: {accuracy:.2f}% ({diagonal_sum:,}/{total:,})', 
                ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        
        # Сохранение
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ График сохранен: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_metrics_comparison(self, save_path=None, show=True):
        """
        Визуализация метрик по классам
        
        Args:
            save_path: путь для сохранения графика
            show: показывать ли график
        """
        if not HAS_VISUALIZATION:
            raise ImportError("Для визуализации необходимо установить matplotlib и seaborn: pip install matplotlib seaborn")
        
        if self.confusion_matrix is None:
            raise ValueError("Сначала нужно построить confusion matrix")
        
        metrics = self.calculate_metrics_from_matrix()
        
        # Подготовка данных
        classes_list = list(metrics.keys())
        precision = [metrics[cls]['precision'] for cls in classes_list]
        recall = [metrics[cls]['recall'] for cls in classes_list]
        f1 = [metrics[cls]['f1'] for cls in classes_list]
        
        # Создание фигуры
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # График 1: Метрики по классам (столбчатая диаграмма)
        x = np.arange(len(classes_list))
        width = 0.25
        
        ax1.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax1.bar(x, recall, width, label='Recall', alpha=0.8)
        ax1.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax1.set_xlabel('Классы', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Значение метрики', fontsize=12, fontweight='bold')
        ax1.set_title('Метрики по классам', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes_list, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 1.1])
        
        # Добавление значений на столбцы
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            ax1.text(i - width, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=7)
            ax1.text(i, r + 0.02, f'{r:.2f}', ha='center', va='bottom', fontsize=7)
            ax1.text(i + width, f + 0.02, f'{f:.2f}', ha='center', va='bottom', fontsize=7)
        
        # График 2: Heatmap метрик
        metrics_df = pd.DataFrame({
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }, index=classes_list)
        
        sns.heatmap(
            metrics_df.T,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Значение метрики'},
            ax=ax2,
            linewidths=0.5,
            linecolor='gray'
        )
        
        ax2.set_title('Heatmap метрик по классам', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Классы', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Метрики', fontsize=12, fontweight='bold')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Сохранение
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ График сохранен: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_error_analysis(self, top_n=15, save_path=None, show=True):
        """
        Визуализация анализа ошибок
        
        Args:
            top_n: количество топ-ошибок для отображения
            save_path: путь для сохранения графика
            show: показывать ли график
        """
        if not HAS_VISUALIZATION:
            raise ImportError("Для визуализации необходимо установить matplotlib и seaborn: pip install matplotlib seaborn")
        
        if self.confusion_matrix is None:
            raise ValueError("Сначала нужно построить confusion matrix")
        
        mistakes = self.find_common_mistakes(top_n=top_n)
        interpretation = self.interpret_errors()
        
        # Создание фигуры с несколькими подграфиками
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # График 1: Топ ошибок
        ax1 = fig.add_subplot(gs[0, :])
        if mistakes:
            true_classes = [f"{t} → {p}" for t, p, _ in mistakes]
            counts = [c for _, _, c in mistakes]
            
            colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(counts)))
            bars = ax1.barh(true_classes, counts, color=colors)
            ax1.set_xlabel('Количество ошибок', fontsize=11, fontweight='bold')
            ax1.set_title(f'Топ-{top_n} самых частых ошибок классификации', 
                         fontsize=12, fontweight='bold')
            ax1.grid(axis='x', alpha=0.3)
            
            # Добавление значений на столбцы
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax1.text(count + max(counts) * 0.01, i, f'{count:,}', 
                        va='center', fontsize=9)
        
        # График 2: Проблемные классы (error rate)
        ax2 = fig.add_subplot(gs[1, 0])
        if interpretation['problematic_classes']:
            problem_classes = [pc['class'] for pc in interpretation['problematic_classes']]
            error_rates = [pc['error_rate'] for pc in interpretation['problematic_classes']]
            
            colors = plt.cm.OrRd(np.linspace(0.5, 0.9, len(error_rates)))
            bars = ax2.barh(problem_classes, error_rates, color=colors)
            ax2.set_xlabel('Процент ошибок (%)', fontsize=11, fontweight='bold')
            ax2.set_title('Проблемные классы (>50% ошибок)', fontsize=12, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
            ax2.set_xlim([0, 100])
            
            for bar, rate in zip(bars, error_rates):
                ax2.text(rate + 1, bar.get_y() + bar.get_height()/2, 
                        f'{rate:.1f}%', va='center', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'Нет проблемных классов', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=12, style='italic')
            ax2.set_title('Проблемные классы', fontsize=12, fontweight='bold')
        
        # График 3: Симметричные ошибки
        ax3 = fig.add_subplot(gs[1, 1])
        if interpretation['symmetric_errors']:
            symmetric_labels = [f"{se['class1']} ↔ {se['class2']}" 
                              for se in interpretation['symmetric_errors']]
            total_mistakes = [se['total_mistakes'] for se in interpretation['symmetric_errors']]
            
            colors = plt.cm.Purples(np.linspace(0.4, 0.9, len(total_mistakes)))
            bars = ax3.barh(symmetric_labels, total_mistakes, color=colors)
            ax3.set_xlabel('Всего ошибок', fontsize=11, fontweight='bold')
            ax3.set_title('Симметричные ошибки', fontsize=12, fontweight='bold')
            ax3.grid(axis='x', alpha=0.3)
            
            for bar, count in zip(bars, total_mistakes):
                ax3.text(count + max(total_mistakes) * 0.01, bar.get_y() + bar.get_height()/2,
                        f'{count:,}', va='center', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'Нет симметричных ошибок', 
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=12, style='italic')
            ax3.set_title('Симметричные ошибки', fontsize=12, fontweight='bold')
        
        # График 4: Классы с низкой производительностью
        ax4 = fig.add_subplot(gs[2, :])
        if interpretation['low_performance_classes']:
            low_perf_classes = [lpc['class'] for lpc in interpretation['low_performance_classes']]
            low_precision = [lpc['precision'] for lpc in interpretation['low_performance_classes']]
            low_recall = [lpc['recall'] for lpc in interpretation['low_performance_classes']]
            low_f1 = [lpc['f1'] for lpc in interpretation['low_performance_classes']]
            
            x = np.arange(len(low_perf_classes))
            width = 0.25
            
            ax4.bar(x - width, low_precision, width, label='Precision', alpha=0.8)
            ax4.bar(x, low_recall, width, label='Recall', alpha=0.8)
            ax4.bar(x + width, low_f1, width, label='F1-Score', alpha=0.8)
            
            ax4.set_xlabel('Классы', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Значение метрики', fontsize=11, fontweight='bold')
            ax4.set_title('Классы с низкой производительностью (<50% по метрикам)', 
                         fontsize=12, fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(low_perf_classes, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(axis='y', alpha=0.3)
            ax4.set_ylim([0, 1])
            ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Порог 50%')
        else:
            ax4.text(0.5, 0.5, 'Нет классов с низкой производительностью', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=12, style='italic')
            ax4.set_title('Классы с низкой производительностью', fontsize=12, fontweight='bold')
        
        plt.suptitle('Анализ ошибок классификации', fontsize=16, fontweight='bold', y=0.995)
        
        # Сохранение
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ График сохранен: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def create_all_visualizations(self, save_dir=None, show=False):
        """
        Создание всех визуализаций
        
        Args:
            save_dir: директория для сохранения графиков
            show: показывать ли графики
        """
        if self.confusion_matrix is None:
            raise ValueError("Сначала нужно построить confusion matrix")
        
        if save_dir is None:
            save_dir = self.figures_dir
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Confusion Matrix (абсолютные значения)
        path1 = os.path.join(save_dir, f'confusion_matrix_absolute_{timestamp}.png')
        self.plot_confusion_matrix(normalized=False, save_path=path1, show=show)
        
        # 2. Confusion Matrix (нормализованная)
        path2 = os.path.join(save_dir, f'confusion_matrix_normalized_{timestamp}.png')
        self.plot_confusion_matrix(normalized=True, save_path=path2, show=show)
        
        # 3. Метрики по классам
        path3 = os.path.join(save_dir, f'metrics_comparison_{timestamp}.png')
        self.plot_metrics_comparison(save_path=path3, show=show)
        
        # 4. Анализ ошибок
        path4 = os.path.join(save_dir, f'error_analysis_{timestamp}.png')
        self.plot_error_analysis(save_path=path4, show=show)
        
        print(f"\n✅ Все визуализации сохранены в директорию: {save_dir}")
        
        return {
            'confusion_matrix_absolute': path1,
            'confusion_matrix_normalized': path2,
            'metrics_comparison': path3,
            'error_analysis': path4
        }


def load_predictions_from_file(predictions_file, true_labels_file=None):
    """Загрузка предсказаний из файла"""
    if predictions_file.endswith('.csv'):
        pred_df = pd.read_csv(predictions_file)
        if 'prediction' in pred_df.columns:
            y_pred = pred_df['prediction'].tolist()
        elif 'category' in pred_df.columns:
            y_pred = pred_df['category'].tolist()
        else:
            y_pred = pred_df.iloc[:, -1].tolist()
    elif predictions_file.endswith('.json'):
        with open(predictions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                y_pred = [item.get('prediction', item.get('category', '')) for item in data]
            else:
                y_pred = data.get('predictions', [])
    else:
        raise ValueError(f"Неподдерживаемый формат: {predictions_file}")
    
    if true_labels_file:
        if true_labels_file.endswith('.csv'):
            true_df = pd.read_csv(true_labels_file)
            if 'category' in true_df.columns:
                y_true = true_df['category'].tolist()
            else:
                y_true = true_df.iloc[:, -1].tolist()
        elif true_labels_file.endswith('.json'):
            with open(true_labels_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    y_true = [item.get('category', '') for item in data]
                else:
                    y_true = data.get('labels', [])
    else:
        df = pd.read_csv('dataset_balanced.csv')
        y_true = df['category'].tolist()
        y_true = y_true[:len(y_pred)]
    
    return y_true, y_pred


def create_demo_analysis():
    """Демонстрационный анализ Confusion Matrix"""
    print("🎯 ДЕМОНСТРАЦИОННЫЙ АНАЛИЗ CONFUSION MATRIX")
    print("=" * 100)
    
    print("\n📥 Загрузка данных...")
    df = pd.read_csv('dataset_balanced.csv')
    print(f"✅ Загружено {len(df):,} записей")
    
    y_true = df['category'].tolist()
    classes = sorted(df['category'].unique())
    
    print(f"✅ Классов: {len(classes)}")
    
    print("\n🔮 Создание baseline предсказаний...")
    keywords_map = {
        'получение_посылки': ['получить', 'посылка', 'заказ', 'выдача', 'забрать'],
        'проблемы_с_кодом': ['код', 'смс', 'сообщение'],
        'связь_с_оператором': ['оператор', 'связаться', 'соединить'],
        'статус_заказа': ['статус', 'отследить', 'где'],
        'проблемы_доставки': ['доставка', 'курьер', 'адрес'],
        'возврат_обмен': ['вернуть', 'обмен', 'замена'],
        'технические_проблемы': ['не работает', 'ошибка', 'проблема'],
        'жалобы': ['жалоба', 'плохо', 'недоволен']
    }
    
    y_pred = []
    for text in df['text'].tolist():
        text_lower = str(text).lower()
        predicted = 'другое'
        
        for category, keywords in keywords_map.items():
            if any(keyword in text_lower for keyword in keywords):
                predicted = category
                break
        
        y_pred.append(predicted)
    
    print(f"✅ Создано {len(y_pred)} предсказаний")
    
    print("\n📊 Построение Confusion Matrix...")
    analyzer = ConfusionMatrixAnalyzer()
    matrix = analyzer.build_confusion_matrix(y_true, y_pred, classes)
    
    print(f"✅ Матрица построена: {len(classes)}x{len(classes)}")
    
    analyzer.print_detailed_report()
    analyzer.save_report('CONFUSION_MATRIX_REPORT.md', include_visualizations=True)
    
    # Создание HTML отчета для веб-просмотра
    print("\n🌐 Создание HTML отчета для веб-просмотра...")
    analyzer.save_html_report('CONFUSION_MATRIX_REPORT.html', include_visualizations=True)
    
    print("\n✅ Демонстрационный анализ завершен!")
    return analyzer


def analyze_from_files(predictions_file, true_labels_file=None):
    """Анализ из файлов"""
    print("🎯 АНАЛИЗ CONFUSION MATRIX ИЗ ФАЙЛОВ")
    print("=" * 100)
    
    print("\n📥 Загрузка данных...")
    y_true, y_pred = load_predictions_from_file(predictions_file, true_labels_file)
    
    print(f"✅ Загружено {len(y_pred)} предсказаний")
    print(f"✅ Загружено {len(y_true)} истинных меток")
    
    classes = sorted(set(y_true) | set(y_pred))
    print(f"✅ Классов: {len(classes)}")
    
    print("\n📊 Построение Confusion Matrix...")
    analyzer = ConfusionMatrixAnalyzer()
    matrix = analyzer.build_confusion_matrix(y_true, y_pred, classes)
    
    analyzer.print_detailed_report()
    analyzer.save_report('CONFUSION_MATRIX_REPORT.md', include_visualizations=True)
    
    # Создание HTML отчета для веб-просмотра
    print("\n🌐 Создание HTML отчета для веб-просмотра...")
    analyzer.save_html_report('CONFUSION_MATRIX_REPORT.html', include_visualizations=True)
    
    print("\n✅ Анализ завершен!")
    return analyzer


def main():
    """Главная функция"""
    import sys
    
    print("=" * 100)
    print("📊 CONFUSION MATRIX АНАЛИЗАТОР ДЛЯ DATA ANALYST")
    print("=" * 100)
    print(f"📅 Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    
    if len(sys.argv) > 1:
        predictions_file = sys.argv[1]
        true_labels_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        if not os.path.exists(predictions_file):
            print(f"❌ Файл не найден: {predictions_file}")
            return
        
        try:
            analyzer = analyze_from_files(predictions_file, true_labels_file)
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
    else:
        try:
            analyzer = create_demo_analysis()
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

