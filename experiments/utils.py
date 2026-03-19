import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import polars as pl

def save_stackplot(dates: list[dt.date], *args, labels: list[str], file_path: str):
    plt.stackplot(dates, *args, labels=labels)
    plt.ylabel("Weights")
    plt.xlabel(None)
    plt.legend(loc="upper right")
    plt.savefig(file_path)
    plt.clf()

def save_lineplot(weights: pl.DataFrame, columns: list[str], labels: list[str], file_path: str):
    for column, label in zip(columns, labels):
        sns.lineplot(weights, x='date', y=column, label=label)
    plt.ylabel("Weights")
    plt.xlabel(None)
    plt.legend(loc="upper right")
    plt.savefig(file_path)
    plt.clf()
