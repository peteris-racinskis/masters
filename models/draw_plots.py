from abc import abstractmethod, ABC
from curses import nonl
import pandas as pd
import numpy as np

from os import listdir
from os.path import exists
from pathlib import Path
from matplotlib import axes, figure
from matplotlib import pyplot as plt


DIR="models/comparison_tables/"
ODIR="models/comparison_charts/"
#IFILE1="models/comparison_tables/sequences-GAN-Epochs-Euclidean g-min.csv"
#IFILE2="models/comparison_tables/sequences-GAN-Epochs-Manhattan g-min.csv"
IFILE="models/comparison_tables/sequences-RNN-Old dataset-Missed by-min.csv"
#IFILE1="models/comparison_tables/categorical-All-ModelClass-Rel. vel. err-min.csv"
#IFILE2="models/comparison_tables/categorical-All-ModelClass-Rel. pos. err-min.csv"


class Chart(ABC):

    def __init__(self, ax: axes.Axes, fname, color="#444444", multi_title=None):
        self._df = pd.read_csv(fname)
        self._filename = Path(fname)
        self.color = color
        self.title = multi_title
        self.ax = ax
        self.parse_filename()
        self.get_data()

    def parse_filename(self):
        (
            self.ttype,
            self.mtype,
            self.argument,
            self.metric,
            self.cb
        ) = self._filename.stem.split("-")
        self.title = f"{self.mtype} - {self.metric} wrt {self.argument}"if self.title is None else self.title
        self.x_axis_name = self.argument
        self.y_axis_name = self.metric
    
    def get_data(self):
        self.xax = self._df.sort_values(self.argument)[self.argument].values
        self.yax = self._df.sort_values(self.argument)[self.metric].values


    @abstractmethod
    def plot(self):
        pass

class CategorialChart(Chart):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.str_xax = self.xax
        #assert self.ttype == "categorical", "Underlying data is sequence"

    def rename_xax(self, d):
        self.str_xax = [d[x] for x in self.xax]

    def plot(self, all=False):
        if not all:
            self.str_xax = self.str_xax[:-1]
            self.yax = self.yax[:-1]
        self.ax.bar(self.str_xax, self.yax, color=["orange", "red", "green"])
        self.ax.set_title(self.title)
        self.ax.set_ylabel(self.y_axis_name)
        self.ax.set_xlabel(self.x_axis_name)


class SequenceChart(Chart):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.ttype == "sequences", "Underlying data is categorical"
        

    def plot(self):
        self.ax.plot(self.xax, self.yax, color=self.color, label=self.metric)
        self.lines, self.handles = self.ax.get_legend_handles_labels()
        self.ax.set_title(self.title)
        self.ax.set_ylabel(self.y_axis_name)
        self.ax.set_xlabel(self.x_axis_name)

def comparison_charts(fnames):
    def condition(s):
        expressions = [
            "categorical" in s,
            "Old dataset" in s,
            "Train" in s,
            "Learning rate" in s,
            "Time signal" in s
        ]
        return any(expressions)

    remap_class = {
        1: "Naive",
        2: "RNN",
        3: "GAN"
    }

    # the learning rate is wrong to begin with.
    remap_lr = {
        1e-5: "1e-5",
        1e-4: "1e-4",
    }

    fnames = [x for x in filter(condition, fnames)]

    for f in fnames:
        if exists(ODIR+f):
            continue
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        chart1 = CategorialChart(ax1, DIR+f)
        chart2 = CategorialChart(ax2, DIR+f)
        if "categorical" in f:
            chart1.rename_xax(remap_class)
            chart2.rename_xax(remap_class)
            chart2.plot(False)
            fig2.savefig(f"{ODIR}{chart2._filename.stem}.png")
        if "Learning rate" in f:
            chart1.rename_xax(remap_lr)
        chart1.plot(True)
        fig1.savefig(f"{ODIR}{chart1._filename.stem}-all.png")
        plt.close('all')


def seq_charts(fnames):
    def condition(s):
        expressions = [
            "categorical" not in s,
            "Old dataset" not in s,
            "Train" not in s,
            "Learning rate" not in s,
            "Time signal" not in s
        ]
        return all(expressions)
    
    fnames = [DIR+x for x in filter(condition, fnames)]

    for f in fnames:
        fig1, ax1 = plt.subplots()
        chart1 = SequenceChart(ax1, f)
        chart1.plot()
        fig1.savefig(f"models/comparison_charts/{chart1._filename.stem}.png")
        plt.close('all')

def RNN_charts():
    pass

def GAN_charts():
    pass


if __name__ == "__main__":
    fnames = listdir(DIR)
    comparison_charts(fnames)
    seq_charts(fnames)