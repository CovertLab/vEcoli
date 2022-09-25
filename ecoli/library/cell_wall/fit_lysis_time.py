import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import expon

DATA = "data/lysis_initiation/time_for_lysis.csv"


def main():
    data = pd.read_csv(DATA, skipinitialspace=True)
    center_time = data[["T_lower(sec)", "T_upper(sec)"]].mean(axis=1)
    width = data["T_upper(sec)"] - data["T_lower(sec)"]

    mean = 192.8

    fig, ax = plt.subplots()
    ax.bar(x=center_time, height=data["P(T)"], width=width)

    x_range = np.arange(0, 500, 10)
    ax.plot(x_range, expon.pdf(x_range, scale=mean), "r--")

    fig.tight_layout()
    fig.savefig("out/lysis_time_fit.png")


if __name__ == "__main__":
    main()
