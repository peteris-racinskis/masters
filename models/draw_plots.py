from abc import abstractmethod
import pandas as pd
import numpy as np

from pathlib import Path
from matplotlib import axes, figure
from matplotlib import pyplot as plt

IFILE="models/comparison_tables/sequences-GAN-Epochs-Euclidean g-min.csv"


class Chart():

    def __init__(self, ax: axes, fname):
        self._df = pd.read_csv(fname)
        self._filename = Path(fname)
        self._ax = ax
        self.parse_filename()

    def parse_filename(self):
        (
            self.ttype,
            self.mtype,
            self.argument,
            self.metric,
            self.cb
        ) = self._filename.stem.split("-")
        self.title = f"{self.mtype} - {self.metric} wrt {self.argument}"
        self.x_axis_name = self.argument
        self.y_axis_name = self.metric
    
    @abstractmethod
    def plot(self):
        pass
    


class CategorialChart(Chart):

    def __init__(self, *args):
        super().__init__(*args)
        assert self.ttype == "categorical", "Underlying data is sequence"
        pass


class SequenceChart(Chart):

    def __init__(self, *args):
        super().__init__(*args)
        assert self.ttype == "sequences", "Underlying data is categorical"
    

if __name__ == "__main__":
    f, a = plt.subplots()
    c = SequenceChart(a, IFILE)
    c = CategorialChart(a, IFILE)
    pass