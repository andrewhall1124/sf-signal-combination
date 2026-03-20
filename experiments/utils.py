import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import polars as pl

def save_weights_stackplot(dates: list[dt.date], *args, labels: list[str], file_path: str, title: str):
    plt.stackplot(dates, *args, labels=labels)
    plt.title(title)
    plt.ylabel("Weights")
    plt.xlabel(None)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.clf()

def save_weights_lineplot(weights: pl.DataFrame, columns: list[str], labels: list[str], file_path: str, title: str):
    for column, label in zip(columns, labels):
        sns.lineplot(weights, x='date', y=column, label=label)
    plt.title(title)
    plt.ylim(0, 1)
    plt.ylabel("Weights")
    plt.xlabel(None)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.clf()

def save_values_lineplot(values: pl.DataFrame, columns: list[str], labels: list[str], file_path: str, title: str, value_name: str):
    for column, label in zip(columns, labels):
        sns.lineplot(values, x='date', y=column, label=label)
    plt.title(title)
    plt.ylabel(value_name)
    plt.xlabel(None)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.clf()
