import datetime as dt
from signals.expr import momentum, reversal, bab
from signals.mve_returns import get_alphas, run_backtest
from pathlib import Path

start = dt.date(2000, 1, 1)
end = dt.date(2024, 12, 31)

configs = [
    {
        'signal_expr': reversal,
        'gamma': 129,
        'constraints': ["ZeroBeta", "ZeroInvestment"]
    },
    {
        'signal_expr': momentum,
        'gamma': 43,
        'constraints': ["ZeroBeta", "ZeroInvestment"]
    },
    {
        'signal_expr': bab,
        'gamma': 37,
        'constraints': ["ZeroInvestment"]
    },
]

for config in configs:
    signal_expr = config['signal_expr']
    gamma = config['gamma']
    constraints = config['constraints']
    signal_name = signal_expr.__name__

    alphas = get_alphas(start, end, signal_expr())

    file_path = Path(f"data/alphas/{signal_name}.parquet")
    file_path.parent.mkdir(exist_ok=True, parents=True)

    alphas.write_parquet(file_path)

    run_backtest(
        signal_name=signal_name,
        alpha_file_path=file_path,
        gamma=gamma,
        constraints=constraints
    )