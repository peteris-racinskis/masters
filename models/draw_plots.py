from abc import abstractmethod, ABC
import pandas as pd
import numpy as np

from random import randint
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

KEY="Key"
ARGUMENT="Argument"
VALUE="Value"


class Chart(ABC):

    def __init__(self, ax: axes.Axes, fname, color="#444444", multi_title=None):
        self._df = pd.read_csv(fname)
        self._filename = Path(fname)
        self.color = color
        self.title = multi_title
        self.ax = ax
        self.parse_filename()

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
        self.get_data()
        self.str_xax = self.xax
        #assert self.ttype == "categorical", "Underlying data is sequence"

    def rename_xax(self, d):
        self.str_xax = [d[x] for x in self.xax]

    def plot(self, all=False):
        if not all:
            self.str_xax = self.str_xax[:-1]
            self.yax = self.yax[:-1]
        self.ax.bar(self.str_xax, self.yax, color=["orange", "red", "green"])
        self.ax.set_title(self.title, fontsize=18)
        self.ax.set_ylabel(self.y_axis_name, fontsize=18)
        self.ax.set_xlabel(self.x_axis_name, fontsize=18)


class SequenceChart(Chart):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_data()
        assert self.ttype == "sequences", "Underlying data is categorical"
        

    def plot(self):
        self.ax.plot(self.xax, self.yax, color=self.color, label=self.metric)
        self.lines, self.handles = self.ax.get_legend_handles_labels()
        self.ax.set_title(self.title, fontsize=18)
        self.ax.set_ylabel(self.y_axis_name, fontsize=18)
        self.ax.set_xlabel(self.x_axis_name, fontsize=18)


class ManySequenceChart(Chart):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys = self._unique_sequences()
        self.base_colors = ["Red", "Orange", "Green", "Blue", "Gray", "Yellow"]
        self.colors = self.sample_rgb(len(self.keys))
    
    def _unique_sequences(self):
        return self._df[KEY].unique()
    
    def _filter_on_key(self, key):
        return self._df.loc[lambda d: d[KEY] == key]
    
    def _plot(self, key, color):
        df = self._filter_on_key(key).sort_values(ARGUMENT)
        values = df.values
        xax = values[:,1]
        yax = values[:,2]
        self.ax.plot(xax, yax, color=color, label=key)
    
    @staticmethod
    def _replace(dict, strings):
        res = []
        for s in strings:
            for k,v in dict.items():
                s = s.replace(k,v)
            res.append(s)
        return res
    
    def plot(self, title="None"):
        for k, c in zip(self.keys, self.colors):
            self._plot(k,c)
        handles, labels = self.ax.get_legend_handles_labels()
        labels = self._replace(
            {
                "1e-05": "1 × 10⁴",
                "0.0001": "1 × 10³",
            },
            labels
        )
        self.ax.legend(handles, labels, fontsize="x-small")
        self.ax.set_title(self.title, fontsize=18)
        self.ax.set_ylabel(self.y_axis_name, fontsize=18)
        self.ax.set_xlabel(self.x_axis_name, fontsize=18)
        
    
    #@staticmethod
    def sample_rgb(self, times):
        colors = []
        '''
        for t in range(times):
            s = "#"
            for i in range(3):
                value = randint(0,255)
                s += "{:02x}".format(value)
            colors.append(s)
        '''
        for t in range(times):
            colors.append(self.base_colors[t])
        return colors



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
        1e-5: "1 × 10⁴",
        1e-4: "1 × 10³",
    }

    fnames = [x for x in filter(condition, fnames)]

    for f in fnames:
        #if exists(ODIR+f):
        #    continue
        fig1, ax1 = plt.subplots()
        plt.gcf().subplots_adjust(left=0.2)
        plt.gcf().subplots_adjust(bottom=0.15)
        fig2, ax2 = plt.subplots()
        plt.gcf().subplots_adjust(left=0.2)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.rcParams['font.size'] = 16
        chart1 = CategorialChart(ax1, DIR+f)
        chart2 = CategorialChart(ax2, DIR+f)
        if "categorical" in f:
            chart1.rename_xax(remap_class)
            chart2.rename_xax(remap_class)
            chart2.plot(False)
            fig2.savefig(f"{ODIR}{chart2._filename.stem}.png")
            print(f"{ODIR}{chart2._filename.stem}.png")
        if "Learning rate" in f:
            chart1.rename_xax(remap_lr)
        chart1.plot(True)
        fig1.savefig(f"{ODIR}{chart1._filename.stem}-all.png")
        print(f"{ODIR}{chart1._filename.stem}-all.png")
        plt.close('all')


def seq_charts(fnames):
    def condition(s):
        expressions = [
            "categorical" not in s,
            "independent" not in s,
            "Old dataset" not in s,
            "Train" not in s,
            "Learning rate" not in s,
            "Time signal" not in s
        ]
        return all(expressions)
    
    fnames = [DIR+x for x in filter(condition, fnames)]

    for f in fnames:
        fig1, ax1 = plt.subplots()
        plt.rcParams['font.size'] = 18
        plt.subplots_adjust(left=0.1)
        plt.subplots_adjust(bottom=0.1)
        chart1 = SequenceChart(ax1, f)
        chart1.plot()
        fig1.savefig(f"models/comparison_charts/{chart1._filename.stem}.png")
        plt.close('all')

def independent_sequence_charts(fnames):
    def condition(s):
        return "independent" in s
        
    fnames = [DIR+x for x in filter(condition, fnames)]

    for f in fnames:
        print(f"processing: {f}")
        fig1, ax1 = plt.subplots(figsize=(10,7.5))
        plt.rcParams['font.size'] = 18
        plt.subplots_adjust(left=0.15)
        plt.subplots_adjust(bottom=0.1)
        chart1 = ManySequenceChart(ax1, f)
        chart1.plot()
        fig1.savefig(f"models/comparison_charts/{chart1._filename.stem}.png")
        plt.close('all')

if __name__ == "__main__":
    fnames = listdir(DIR)
    #comparison_charts(fnames)
    #seq_charts(fnames)
    independent_sequence_charts(fnames)