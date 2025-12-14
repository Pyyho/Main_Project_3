"""
Тесты для модуля core.
"""

import pandas as pd
import numpy as np
import pytest
from eda_cli.core import (
    compute_basic_stats,
    compute_missing_stats,
    compute_numeric_stats,
    compute_categorical_stats,
    compute_quality_flags,
    generate_report_data
)


def test_compute_basic_stats():
    """Тест для базовой статистики."""
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['a', 'b', 'c']
    })
    
    stats = compute_basic_stats(df)
    
    assert stats['num_rows'] == 3
    assert stats['num_columns'] == 2
    assert 'A' in stats['columns']
    assert 'B' in stats['columns']
    assert stats['dtypes']['A'] == 'int64'
    assert stats['dtypes']['B'] == 'object'


def test_compute_missing_stats():
    """Тест для статистики пропусков."""
    df = pd.DataFrame({
        'A': [1, np.nan, 3],
        'B': ['a', 'b', np.nan],
        'C': [1, 2, 3]
    })
    
    stats = compute_missing_stats(df)
    
    assert stats['missing_counts']['A'] == 1
    assert stats['missing_counts']['B'] == 1
    assert stats['missing_counts']['C'] == 0
    assert 'A' in stats['cols_with_missing']
    assert 'B' in stats['cols_with_missing']
    assert 'C' not in stats['cols_with_missing']
    assert stats['total_missing'] == 2


def test_compute_numeric_stats():
    """Тест для числовой статистики."""
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [0, 0, 1, 2, 3],
        'C': ['a', 'b', 'c', 'd', 'e']  # Не числовая
    })
    
    stats = compute_numeric_stats(df)
    
    assert 'A' in stats['numeric_cols']
    assert 'B' in stats['numeric_cols']
    assert 'C' not in stats['numeric_cols']
    assert stats['stats']['A']['mean'] == 3.0
    assert stats['stats']['B']['zeros_count'] == 2
    assert stats['stats']['B']['zeros_share'] == 0.4


def test_compute_categorical_stats():
    """Тест для категориальной статистики."""
    df = pd.DataFrame({
        'A': ['cat', 'dog', 'cat', 'dog', 'bird'],
        'B': [1, 2, 3, 4, 5]
    })
    
    stats = compute_categorical_stats(df, top_k=2)
    
    assert 'A' in stats['categorical_cols']
    assert 'B' not in stats['categorical_cols']
    assert stats['stats']['A']['unique_count'] == 3
    assert 'cat' in stats['stats']['A']['top_values']
    assert 'dog' in stats['stats']['A']['top_values']


def test_compute_quality_flags_constant_columns():
    """Тест для новой эвристики: константные колонки."""
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'constant_col': [1, 1, 1, 1, 1],  # Константная колонка
        'normal_col': [1, 2, 3, 4, 5]
    })
    
    flags = compute_quality_flags(df)
    
    # Проверяем флаг константных колонок
    assert flags['has_constant_columns'] == True
    assert 'constant_col' in flags['constant_columns']
    assert len(flags['constant_columns']) == 1
    
    # Качество должно быть ниже из-за константной колонки
    assert flags['quality_score'] < 100
    assert 'constant_cols' in flags['quality_penalties']


def test_compute_quality_flags_high_cardinality():
    """Тест для новой эвристики: высококардинальные категориальные признаки."""
    # Создаем датасет с высококардинальной категориальной колонкой
    df = pd.DataFrame({
        'id': range(100),
        'high_card_col': [f'value_{i}' for i in range(100)],  # 100% уникальных значений
        'low_card_col': ['A', 'B'] * 50  # 2 уникальных значения
    })
    
    flags = compute_quality_flags(df)
    
    # Проверяем флаг высококардинальных категорий
    assert flags['has_high_cardinality_categoricals'] == True
    assert len(flags['high_cardinality_cols']) == 1
    assert flags['high_cardinality_cols'][0]['column'] == 'high_card_col'
    assert flags['high_cardinality_cols'][0]['unique_ratio'] > 0.5
    
    # low_card_col не должна быть в списке
    high_card_cols = [col['column'] for col in flags['high_cardinality_cols']]
    assert 'low_card_col' not in high_card_cols


def test_compute_quality_flags_id_duplicates():
    """Тест для новой эвристики: дубликаты ID."""
    df = pd.DataFrame({
        'user_id': [1, 2, 2, 3, 4],  # Дубликат user_id=2
        'transaction_id': [100, 101, 102, 103, 104],  # Уникальные
        'value': [10, 20, 30, 40, 50]
    })
    
    flags = compute_quality_flags(df)
    
    # Проверяем флаг дубликатов ID
    assert flags['has_suspicious_id_duplicates'] == True
    assert len(flags['suspicious_id_duplicates']) == 1
    assert flags['suspicious_id_duplicates'][0]['column'] == 'user_id'
    assert flags['suspicious_id_duplicates'][0]['duplicate_count'] == 1
    assert flags['suspicious_id_duplicates'][0]['duplicate_share'] == 0.2


def test_compute_quality_flags_many_zeros():
    """Тест для новой эвристики: много нулей в числовых колонках."""
    df = pd.DataFrame({
        'col_with_many_zeros': [0, 0, 0, 0, 1, 2, 3, 4, 5, 6],  # 40% нулей
        'col_with_few_zeros': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],   # 10% нулей
        'categorical_col': ['A', 'B'] * 5
    })
    
    flags = compute_quality_flags(df)
    
    # Проверяем флаг многих нулей
    assert flags['has_many_zero_values'] == True
    assert len(flags['many_zeros_cols']) == 1
    assert flags['many_zeros_cols'][0]['column'] == 'col_with_many_zeros'
    assert flags['many_zeros_cols'][0]['zero_share'] > 0.4
    
    # col_with_few_zeros не должна быть в списке
    many_zeros_cols = [col['column'] for col in flags['many_zeros_cols']]
    assert 'col_with_few_zeros' not in many_zeros_cols


def test_generate_report_data_with_params():
    """Тест генерации данных отчета с параметрами."""
    df = pd.DataFrame({
        'A': [1, 2, 3, np.nan, 5],
        'B': ['cat', 'dog', 'cat', 'dog', 'bird'],
        'C': [0, 0, 0, 1, 2]
    })
    
    report_data = generate_report_data(
        df=df,
        max_hist_columns=5,
        top_k_categories=3,
        min_missing_share=0.2,
        title="Тестовый отчет"
    )
    
    assert report_data['title'] == "Тестовый отчет"
    assert report_data['report_params']['max_hist_columns'] == 5
    assert report_data['report_params']['top_k_categories'] == 3
    assert report_data['report_params']['min_missing_share'] == 0.2
    
    # Проверяем, что параметры влияют на вычисления
    cat_stats = report_data['categorical_stats']
    assert 'B' in cat_stats['stats']
    # Должно быть только топ-3 значения
    assert len(cat_stats['stats']['B']['top_values']) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])