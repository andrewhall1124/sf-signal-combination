import polars as pl
from signals.utils import get_assets_data
import datetime as dt
from pathlib import Path
import numpy as np
from pipelines.configs import configs

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
            pl.col('forward_return').mul(pl.col('weight')).sum()
        )
        .sort('date')
    )

    # Diagnostic
    volatility = portfolio_returns['forward_return'].std() * np.sqrt(252)
    print(f"{signal_name}:")
    print(f"    Volatility (%): {volatility:2%}")

    # Save returns
    file_path = Path(f"data/mve_returns/f{signal_name}.parquet")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    portfolio_returns.write_parquet(file_path)

