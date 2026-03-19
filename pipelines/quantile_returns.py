"""Script for computing quantile returns for each signal"""
from signals.quantile_returns import compute_quantile_returns
from signals.expr import momentum, reversal, bab
import datetime as dt
from pathlib import Path

start = dt.date(2000, 1, 1)
end = dt.date(2024, 12, 31)

for signal_expr in [momentum, reversal, bab]:
    returns = compute_quantile_returns(
        signal_expr=signal_expr(),
        start=start,
        end=end
    )
    file_path = Path(f"data/quantile_returns/{signal_expr.__name__}.parquet")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    returns.write_parquet(file_path)