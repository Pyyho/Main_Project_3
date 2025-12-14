"""
Модуль с базовой EDA-логикой.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


def read_data(filepath: str) -> pd.DataFrame:
    """Читает CSV файл в DataFrame."""
    return pd.read_csv(filepath)


def compute_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Вычисляет базовую статистику по DataFrame."""
    stats = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
    }
    return stats


def compute_missing_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Вычисляет статистику по пропускам."""
    missing_counts = df.isnull().sum()
    missing_shares = missing_counts / len(df)
    
    # Колонки с пропусками
    cols_with_missing = missing_counts[missing_counts > 0].index.tolist()
    
    return {
        "missing_counts": missing_counts.to_dict(),
        "missing_shares": missing_shares.to_dict(),
        "cols_with_missing": cols_with_missing,
        "total_missing": missing_counts.sum(),
        "total_missing_share": missing_counts.sum() / (len(df) * len(df.columns)),
    }


def compute_numeric_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Вычисляет статистику по числовым колонкам."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return {"numeric_cols": [], "stats": {}}
    
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
            "q25": float(df[col].quantile(0.25)),
            "q75": float(df[col].quantile(0.75)),
            "zeros_count": int((df[col] == 0).sum()),
            "zeros_share": float((df[col] == 0).sum() / len(df)),
        }
    
    return {
        "numeric_cols": numeric_cols,
        "stats": stats,
    }


def compute_categorical_stats(df: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
    """Вычисляет статистику по категориальным колонкам."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    if not cat_cols:
        return {"categorical_cols": [], "stats": {}}
    
    stats = {}
    for col in cat_cols:
        value_counts = df[col].value_counts(dropna=False)
        top_values = value_counts.head(top_k).to_dict()
        
        stats[col] = {
            "unique_count": int(df[col].nunique()),
            "mode": str(df[col].mode().iloc[0]) if not df[col].mode().empty else None,
            "mode_count": int(value_counts.iloc[0]) if not value_counts.empty else 0,
            "top_values": top_values,
        }
    
    return {
        "categorical_cols": cat_cols,
        "stats": stats,
    }


def compute_quality_flags(df: pd.DataFrame, min_missing_share: float = 0.3) -> Dict[str, Any]:
    """
    Вычисляет флаги качества данных.
    
    Параметры:
    ----------
    df : pd.DataFrame
        Входной DataFrame
    min_missing_share : float, default=0.3
        Порог доли пропусков для флага has_high_missing_share
        
    Возвращает:
    -----------
    Dict с флагами качества и интегральным показателем
    """
    # Базовые флаги
    missing_stats = compute_missing_stats(df)
    numeric_stats = compute_numeric_stats(df)
    cat_stats = compute_categorical_stats(df)
    
    flags = {}
    
    # 1. Флаг высоких пропусков в колонках
    high_missing_cols = [
        col for col, share in missing_stats["missing_shares"].items()
        if share > min_missing_share
    ]
    flags["has_high_missing_share"] = len(high_missing_cols) > 0
    flags["high_missing_cols"] = high_missing_cols
    flags["high_missing_threshold"] = min_missing_share
    
    # 2. Флаг константных колонок (новая эвристика)
    constant_cols = []
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            constant_cols.append(col)
    
    flags["has_constant_columns"] = len(constant_cols) > 0
    flags["constant_columns"] = constant_cols
    
    # 3. Флаг высококардинальных категориальных признаков (новая эвристика)
    high_cardinality_cols = []
    high_cardinality_threshold = 0.5  # > 50% уникальных значений считается высококардинальным
    
    for col in cat_stats["categorical_cols"]:
        unique_ratio = cat_stats["stats"][col]["unique_count"] / len(df)
        if unique_ratio > high_cardinality_threshold:
            high_cardinality_cols.append({
                "column": col,
                "unique_count": cat_stats["stats"][col]["unique_count"],
                "unique_ratio": unique_ratio
            })
    
    flags["has_high_cardinality_categoricals"] = len(high_cardinality_cols) > 0
    flags["high_cardinality_cols"] = high_cardinality_cols
    flags["high_cardinality_threshold"] = high_cardinality_threshold
    
    # 4. Флаг дублирования ID (новая эвристика)
    # Ищем колонки, которые могут быть идентификаторами
    id_candidates = [col for col in df.columns if "id" in col.lower() or "ID" in col]
    suspicious_id_duplicates = []
    
    for col in id_candidates:
        duplicate_count = df[col].duplicated().sum()
        if duplicate_count > 0:
            suspicious_id_duplicates.append({
                "column": col,
                "duplicate_count": int(duplicate_count),
                "duplicate_share": float(duplicate_count / len(df))
            })
    
    flags["has_suspicious_id_duplicates"] = len(suspicious_id_duplicates) > 0
    flags["suspicious_id_duplicates"] = suspicious_id_duplicates
    
    # 5. Флаг многих нулей в числовых колонках (новая эвристика)
    many_zeros_cols = []
    zero_threshold = 0.4  # > 40% нулей считается проблемой
    
    for col in numeric_stats["numeric_cols"]:
        zero_share = numeric_stats["stats"][col]["zeros_share"]
        if zero_share > zero_threshold:
            many_zeros_cols.append({
                "column": col,
                "zero_count": numeric_stats["stats"][col]["zeros_count"],
                "zero_share": zero_share
            })
    
    flags["has_many_zero_values"] = len(many_zeros_cols) > 0
    flags["many_zeros_cols"] = many_zeros_cols
    flags["zero_threshold"] = zero_threshold
    
    # Интегральный показатель качества (обновленный с учетом новых флагов)
    quality_score = 100
    
    # Штрафы за разные проблемы
    penalties = {
        "high_missing": len(high_missing_cols) * 15,
        "constant_cols": len(constant_cols) * 10,
        "high_cardinality": len(high_cardinality_cols) * 8,
        "id_duplicates": len(suspicious_id_duplicates) * 12,
        "many_zeros": len(many_zeros_cols) * 7,
    }
    
    total_penalty = sum(penalties.values())
    quality_score = max(0, quality_score - total_penalty)
    
    flags["quality_score"] = quality_score
    flags["quality_penalties"] = penalties
    
    return flags


def generate_report_data(
    df: pd.DataFrame,
    max_hist_columns: int = 10,
    top_k_categories: int = 5,
    min_missing_share: float = 0.3,
    title: str = "Анализ данных"
) -> Dict[str, Any]:
    """
    Генерирует все данные для отчета.
    
    Параметры:
    ----------
    df : pd.DataFrame
        Входной DataFrame
    max_hist_columns : int, default=10
        Максимальное количество колонок для гистограмм
    top_k_categories : int, default=5
        Количество топ-значений для категориальных признаков
    min_missing_share : float, default=0.3
        Порог доли пропусков
    title : str, default="Анализ данных"
        Заголовок отчета
        
    Возвращает:
    -----------
    Dict со всеми данными для отчета
    """
    return {
        "title": title,
        "basic_stats": compute_basic_stats(df),
        "missing_stats": compute_missing_stats(df),
        "numeric_stats": compute_numeric_stats(df),
        "categorical_stats": compute_categorical_stats(df, top_k=top_k_categories),
        "quality_flags": compute_quality_flags(df, min_missing_share=min_missing_share),
        "report_params": {
            "max_hist_columns": max_hist_columns,
            "top_k_categories": top_k_categories,
            "min_missing_share": min_missing_share,
            "title": title,
        }
    }