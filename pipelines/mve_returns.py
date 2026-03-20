import polars as pl
from signals.utils import get_assets_data
import datetime as dt
from pathlib import Path
import numpy as np
from pipelines.configs import configs
import seaborn as sns
import matplotlib.pyplot as plt

def save_returns_plot(data: pl.DataFrame, signal_name: str):
    title = {
        'momentum': 'Momentum',
        'reversal': 'Reversal',
        'bab': 'BAB'
    }[signal_name]

    sns.lineplot(data, x='date', y='return')
    plt.title(title)
    plt.xlabel(None)
    plt.ylabel('Cumulative Log Return (%)')
    plt.tight_layout()
    plt.savefig(f"results/returns/{signal_name}.png")
    plt.clf()

start = dt.date(2000, 1, 1)
end = dt.date(2024, 12, 31)

assets = (
    get_assets_data(start, end)
    .select('date', 'barrid', 'forward_return')
)

for config in configs:
    signal_expr = config['signal_expr']
    gamma = config['gamma']
    constraints = config['constraints']
    signal_name = signal_expr.__name__

    weights = pl.read_parquet(f"weights/{signal_name}/{gamma}/*.parquet")

    portfolio_returns = (
        weights
        .join(
            other=assets,
            on=['date', 'barrid'],
            how='left'
        )
        .group_by('date')
        .agg(
            pl.col('forward_return').mul(pl.col('weight')).sum().alias('return')
        )
        .sort('date')
    )

    # Diagnostic
    volatility = portfolio_returns['return'].std() * np.sqrt(252)
    print(f"{signal_name}:")
    print(f"    Volatility (%): {volatility:2%}")

    cumulative_returns = (
        portfolio_returns
        .sort('date')
        .with_columns(
            pl.col('return')
            .log1p()
            .cum_sum()
            .mul(100)
        )
    )
    save_returns_plot(
        data=cumulative_returns,
        signal_name=signal_name
    )

    # Save returns
    file_path = Path(f"data/mve_returns/{signal_name}.parquet")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    portfolio_returns.write_parquet(file_path)

