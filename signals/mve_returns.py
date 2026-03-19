import polars as pl
from signals.utils import get_assets_data
import datetime as dt
from signals.expr import momentum
from sf_backtester import BacktestRunner, BacktestConfig, SlurmConfig

def get_alphas(start: dt.date, end: dt.date, signal_expr: pl.Expr) -> pl.DataFrame:
    assets = get_assets_data(start, end)

    signals = (
        assets
        .sort('date', 'barrid')
        .with_columns(
            signal=signal_expr
        )
    )

    scores = (
        signals
        .with_columns(
            pl.col('signal')
            .sub(pl.col('signal').mean())
            .truediv(pl.col('signal').std())
            .over('date')
            .alias('score')
        )
    )

    alphas = (
        scores
        .with_columns(
            pl.lit(0.05)
            .mul(pl.col('score'))
            .mul(pl.col('specific_risk'))
            .alias('alpha')
        )
    )

    print(alphas)

    filtered = (
        alphas
        .filter(
            pl.col('alpha').is_not_null(),
            pl.col('predicted_beta').is_not_null(),
            pl.col('price').gt(5),
        )
        .sort('date', 'barrid')
    )

    print(filtered)

def run_backtest(signal_name: str, alpha_file_path: str, gamma: int):
    slurm_config = SlurmConfig(
        n_cpus=8,
        mem="32G",
        time="03:00:00",
        mail_type="BEGIN,END,FAIL",
        max_concurrent_jobs=30,
    )

    config = BacktestConfig(
        signal_name=signal_name,
        gamma=gamma,
        data_path=alpha_file_path,
        project_root="/home/amh1124/Projects/",
        byu_email="amh1124@byu.edu",
        constraints=["ZeroBeta", "ZeroInvestment"],
        slurm=slurm_config,
    )

    runner = BacktestRunner(config)

    # Preview the sbatch script
    print(runner.dry_run())

    # Submit to SLURM
    runner.submit()

if __name__ == '__main__':
    print(get_alphas(dt.date(2000, 1, 1), dt.date(2024, 12, 31), momentum()))