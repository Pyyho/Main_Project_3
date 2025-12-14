"""
Модуль для визуализации и генерации отчетов.
"""

from typing import Dict, Any, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os


def create_histograms(
    df: pd.DataFrame, 
    numeric_cols: List[str], 
    max_columns: int = 10,
    save_dir: str = "."
) -> None:
    """
    Создает гистограммы для числовых колонок.
    
    Параметры:
    ----------
    df : pd.DataFrame
        Входной DataFrame
    numeric_cols : List[str]
        Список числовых колонок
    max_columns : int, default=10
        Максимальное количество колонок для визуализации
    save_dir : str, default="."
        Директория для сохранения изображений
    """
    # Ограничиваем количество колонок
    cols_to_plot = numeric_cols[:max_columns]
    
    if not cols_to_plot:
        return
    
    n_cols = min(2, len(cols_to_plot))
    n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if len(cols_to_plot) > 1 else [axes]
    
    for i, col in enumerate(cols_to_plot):
        ax = axes[i]
        df[col].hist(ax=ax, bins=30, edgecolor='black')
        ax.set_title(f'Распределение {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Частота')
    
    # Скрываем лишние оси
    for i in range(len(cols_to_plot), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    hist_path = Path(save_dir) / "histograms.png"
    plt.savefig(hist_path, dpi=100)
    plt.close()
    
    return str(hist_path)


def generate_markdown_report(
    report_data: Dict[str, Any],
    save_dir: str = "."
) -> str:
    """
    Генерирует markdown-отчет на основе данных.
    
    Параметры:
    ----------
    report_data : Dict[str, Any]
        Данные для отчета
    save_dir : str, default="."
        Директория для сохранения отчета
        
    Возвращает:
    -----------
    str: Путь к сохраненному отчету
    """
    report_path = Path(save_dir) / "report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # Заголовок
        f.write(f"# {report_data['title']}\n\n")
        
        # Параметры отчета
        params = report_data['report_params']
        f.write("## Параметры анализа\n")
        f.write(f"- Максимум колонок для гистограмм: {params['max_hist_columns']}\n")
        f.write(f"- Топ-K категорий: {params['top_k_categories']}\n")
        f.write(f"- Порог проблемных пропусков: {params['min_missing_share']:.1%}\n\n")
        
        # Базовая статистика
        basic = report_data['basic_stats']
        f.write("## Общая информация\n")
        f.write(f"- Количество строк: {basic['num_rows']:,}\n")
        f.write(f"- Количество колонок: {basic['num_columns']}\n")
        f.write(f"- Использование памяти: {basic['memory_usage_mb']:.2f} MB\n\n")
        
        # Типы данных
        f.write("## Типы данных\n")
        for col, dtype in basic['dtypes'].items():
            f.write(f"- `{col}`: {dtype}\n")
        f.write("\n")
        
        # Пропуски
        missing = report_data['missing_stats']
        f.write("## Пропуски в данных\n")
        
        if missing['cols_with_missing']:
            f.write("### Колонки с пропусками:\n")
            for col in missing['cols_with_missing']:
                count = missing['missing_counts'][col]
                share = missing['missing_shares'][col]
                f.write(f"- `{col}`: {count:,} пропусков ({share:.1%})\n")
        else:
            f.write("Пропусков нет.\n")
        
        f.write(f"\nВсего пропусков: {missing['total_missing']:,} ")
        f.write(f"({missing['total_missing_share']:.1%} от всех ячеек)\n\n")
        
        # Качество данных
        quality = report_data['quality_flags']
        f.write("## Качество данных\n")
        f.write(f"**Интегральный показатель качества: {quality['quality_score']:.1f}/100**\n\n")
        
        f.write("### Проблемы обнаружены:\n")
        
        # Высокие пропуски
        if quality['has_high_missing_share']:
            f.write(f"1. **Высокие пропуски** (> {quality['high_missing_threshold']:.0%}):\n")
            for col in quality['high_missing_cols']:
                share = missing['missing_shares'][col]
                f.write(f"   - `{col}`: {share:.1%}\n")
        
        # Константные колонки
        if quality['has_constant_columns']:
            f.write(f"2. **Константные колонки**:\n")
            for col in quality['constant_columns']:
                f.write(f"   - `{col}`\n")
        
        # Высококардинальные категории
        if quality['has_high_cardinality_categoricals']:
            f.write(f"3. **Высококардинальные категориальные признаки** ")
            f.write(f"(> {quality['high_cardinality_threshold']:.0%} уникальных значений):\n")
            for col_info in quality['high_cardinality_cols']:
                f.write(f"   - `{col_info['column']}`: ")
                f.write(f"{col_info['unique_count']:,} уникальных ")
                f.write(f"({col_info['unique_ratio']:.1%})\n")
        
        # Дубликаты ID
        if quality['has_suspicious_id_duplicates']:
            f.write(f"4. **Дубликаты в ID-колонках**:\n")
            for id_info in quality['suspicious_id_duplicates']:
                f.write(f"   - `{id_info['column']}`: ")
                f.write(f"{id_info['duplicate_count']:,} дубликатов ")
                f.write(f"({id_info['duplicate_share']:.1%})\n")
        
        # Много нулей
        if quality['has_many_zero_values']:
            f.write(f"5. **Много нулей в числовых колонках** ")
            f.write(f"(> {quality['zero_threshold']:.0%} нулей):\n")
            for zero_info in quality['many_zeros_cols']:
                f.write(f"   - `{zero_info['column']}`: ")
                f.write(f"{zero_info['zero_count']:,} нулей ")
                f.write(f"({zero_info['zero_share']:.1%})\n")
        
        if not any([
            quality['has_high_missing_share'],
            quality['has_constant_columns'],
            quality['has_high_cardinality_categoricals'],
            quality['has_suspicious_id_duplicates'],
            quality['has_many_zero_values']
        ]):
            f.write("Серьезных проблем не обнаружено.\n")
        
        f.write("\n### Штрафы за качество:\n")
        for problem, penalty in quality['quality_penalties'].items():
            f.write(f"- {problem}: -{penalty} баллов\n")
        f.write("\n")
        
        # Числовая статистика
        numeric = report_data['numeric_stats']
        if numeric['numeric_cols']:
            f.write("## Числовая статистика\n")
            for col in numeric['numeric_cols']:
                stats = numeric['stats'][col]
                f.write(f"### `{col}`\n")
                f.write(f"- Среднее: {stats['mean']:.2f}\n")
                f.write(f"- Стандартное отклонение: {stats['std']:.2f}\n")
                f.write(f"- Минимум: {stats['min']:.2f}\n")
                f.write(f"- Максимум: {stats['max']:.2f}\n")
                f.write(f"- Медиана: {stats['median']:.2f}\n")
                f.write(f"- 25-й перцентиль: {stats['q25']:.2f}\n")
                f.write(f"- 75-й перцентиль: {stats['q75']:.2f}\n")
                f.write(f"- Нулей: {stats['zeros_count']:,} ({stats['zeros_share']:.1%})\n")
                f.write("\n")
        
        # Категориальная статистика
        categorical = report_data['categorical_stats']
        if categorical['categorical_cols']:
            f.write("## Категориальная статистика\n")
            for col in categorical['categorical_cols']:
                stats = categorical['stats'][col]
                f.write(f"### `{col}`\n")
                f.write(f"- Уникальных значений: {stats['unique_count']:,}\n")
                f.write(f"- Мода: {stats['mode']}\n")
                f.write(f"- Частота моды: {stats['mode_count']:,}\n")
                
                f.write("- Топ значения:\n")
                for value, count in stats['top_values'].items():
                    share = count / report_data['basic_stats']['num_rows']
                    f.write(f"  - `{value}`: {count:,} ({share:.1%})\n")
                f.write("\n")
        
        # Визуализации
        f.write("## Визуализации\n")
        f.write("Гистограммы числовых признаков сохранены в `histograms.png`\n")
    
    return str(report_path)