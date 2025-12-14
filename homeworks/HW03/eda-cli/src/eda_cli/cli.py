"""
CLI-интерфейс для EDA-утилиты.
"""

import click
import pandas as pd
from pathlib import Path
from .core import read_data, compute_basic_stats, compute_missing_stats, compute_numeric_stats, generate_report_data
from .viz import create_histograms, generate_markdown_report


@click.group()
def cli():
    """Консольная утилита для разведочного анализа данных (EDA)."""
    pass


@cli.command()
@click.argument('filepath', type=click.Path(exists=True))
def overview(filepath: str):
    """
    Выводит общую информацию о датасете.
    
    FILEPATH: Путь к CSV файлу
    """
    df = read_data(filepath)
    
    basic = compute_basic_stats(df)
    missing = compute_missing_stats(df)
    numeric = compute_numeric_stats(df)
    
    click.echo("=" * 50)
    click.echo(f"Обзор датасета: {Path(filepath).name}")
    click.echo("=" * 50)
    
    click.echo(f"\n📊 Базовая статистика:")
    click.echo(f"  • Строк: {basic['num_rows']:,}")
    click.echo(f"  • Колонок: {basic['num_columns']}")
    click.echo(f"  • Память: {basic['memory_usage_mb']:.1f} MB")
    
    click.echo(f"\n📋 Типы данных:")
    for col, dtype in basic['dtypes'].items():
        click.echo(f"  • {col}: {dtype}")
    
    click.echo(f"\n❌ Пропуски:")
    if missing['cols_with_missing']:
        click.echo(f"  • Всего пропусков: {missing['total_missing']:,}")
        click.echo(f"  • Колонки с пропусками: {len(missing['cols_with_missing'])}")
        for col in missing['cols_with_missing'][:5]:  # Показываем первые 5
            share = missing['missing_shares'][col]
            click.echo(f"    - {col}: {share:.1%}")
        if len(missing['cols_with_missing']) > 5:
            click.echo(f"    ... и ещё {len(missing['cols_with_missing']) - 5}")
    else:
        click.echo("  • Пропусков нет")
    
    click.echo(f"\n🔢 Числовые колонки: {len(numeric['numeric_cols'])}")
    for col in numeric['numeric_cols'][:3]:  # Показываем первые 3
        stats = numeric['stats'][col]
        click.echo(f"  • {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
    if len(numeric['numeric_cols']) > 3:
        click.echo(f"  ... и ещё {len(numeric['numeric_cols']) - 3}")


@cli.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.option('--out-dir', '-o', default='reports',
              help='Директория для сохранения отчета (по умолчанию: reports)')
@click.option('--max-hist-columns', default=10, type=int,
              help='Максимальное количество колонок для гистограмм (по умолчанию: 10)')
@click.option('--top-k-categories', default=5, type=int,
              help='Количество топ-значений для категориальных признаков (по умолчанию: 5)')
@click.option('--title', default='Анализ данных',
              help='Заголовок отчета (по умолчанию: "Анализ данных")')
@click.option('--min-missing-share', default=0.3, type=float,
              help='Порог доли пропусков для флага проблемных колонок (по умолчанию: 0.3)')
def report(filepath: str, out_dir: str, max_hist_columns: int, 
           top_k_categories: int, title: str, min_missing_share: float):
    """
    Генерирует полный отчет по датасету.
    
    FILEPATH: Путь к CSV файлу
    """
    # Создаем директорию для отчета
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)
    
    # Читаем данные
    df = read_data(filepath)
    
    # Генерируем данные для отчета с учетом параметров
    report_data = generate_report_data(
        df=df,
        max_hist_columns=max_hist_columns,
        top_k_categories=top_k_categories,
        min_missing_share=min_missing_share,
        title=title
    )
    
    # Создаем гистограммы
    if report_data['numeric_stats']['numeric_cols']:
        hist_path = create_histograms(
            df=df,
            numeric_cols=report_data['numeric_stats']['numeric_cols'],
            max_columns=max_hist_columns,
            save_dir=out_dir
        )
        click.echo(f"📊 Гистограммы сохранены: {hist_path}")
    
    # Генерируем markdown-отчет
    md_path = generate_markdown_report(report_data, save_dir=out_dir)
    
    click.echo(f"📄 Отчет сохранен: {md_path}")
    click.echo(f"📁 Директория: {out_path.absolute()}")
    
    # Выводим краткую сводку
    quality = report_data['quality_flags']
    click.echo(f"\n🎯 Качество данных: {quality['quality_score']:.1f}/100")
    
    if quality['quality_score'] < 70:
        click.echo("⚠️  Внимание: обнаружены проблемы с качеством данных")
        for problem, penalty in quality['quality_penalties'].items():
            if penalty > 0:
                click.echo(f"   • {problem}: -{penalty} баллов")
    else:
        click.echo("✅ Качество данных удовлетворительное")


if __name__ == '__main__':
    cli()