from os import listdir

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd

#IDIR="processed_data_old/split/"
#IDIR="processed_data_old/thresh/"
IDIR="processed_data_old/labelled/"
# found file: demo-22-03-2022-11:34:32.csv
# found file: demo-22-03-2022-11:34:32-labelled.csv
# found file: demo-22-03-2022-11:34:32-thresh.csv
IFILES=["processed_data_old/train_datasets/train-start-time-5e9156387f59cb9efb35.csv"]
IFILES=["demo-22-03-2022-11:34:32.csv"]
IFILES=["demo-22-03-2022-11:34:32-labelled.csv"]
STEPS = 500


if __name__ == "__main__":
    #ifiles =  listdir(IDIR)
    ifiles = IFILES
    for fname in ifiles:
        if fname[-3:] != "csv":
            continue
        ifile = IDIR+fname
        #ifile = fname
        df = pd.read_csv(ifile)
        #xax = df['Time'].values
        xax = np.arange(STEPS)
        yax_cols={
            "Bottle.position.x" : "blue",
            "Bottle.position.x-t" : "green",
            "Bottle.position.xpred" : "red",
            "Bottle.position.z": "black",
            "Bottle.position.z-t" : "grey",
            "Bottle.position.zpred" : "orange",
            "Freefall" : "cyan",
            "Passed" : "cyan"
        }
        colors = ["blue"]
        for yax_col, col in yax_cols.items():
            yax = df[yax_col].values
            plt.plot(xax, yax, color=col)
        #yax1 = df['position.x'].values[:STEPS]
        #yax2 = df['position.x-t'].values[:STEPS]
        #plt.plot(xax, yax1, color="blue")
        #plt.plot(xax, yax2, color="red")
        #plt.plot(xax, yax3, color="green")
        #plt.plot(xax, yax4, color="orange")
        #plt.plot(xax, yax5, color="grey")
        #plt.plot(xax, yax2, color="red")
        plt.ylim(-2.2,1.4)
        plt.xlim(100, 290)
        rcParams['savefig.dpi'] = 1000
        plt.xlabel("timestep")
        plt.ylabel("coordinate, m")
        plt.show()
        #plt.savefig(ifile[:-3]+"jpg")
        #plt.cla()
        break