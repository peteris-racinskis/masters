from abc import abstractmethod, ABC
import pandas as pd
import numpy as np

from pathlib import Path
from matplotlib import axes, figure
from matplotlib import pyplot as plt

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

def comparison_charts():
    pass

def NaiveBC_charts():
    pass

def RNN_charts():
    pass

def GAN_charts():
    pass


if __name__ == "__main__":
    f, a = plt.subplots()
    #c = SequenceChart(a, IFILE, color="red")
    c = CategorialChart(a, IFILE)
    c.rename_xax({0:"New", 1:"Old"})
    c.plot(True)
    f.savefig("models/testimage.png")