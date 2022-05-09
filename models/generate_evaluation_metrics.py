import sys
from pathlib import Path

import pandas as pd
import numpy as np
from pyparsing import col


#IFILE="/home/user/repos/masters/models/validation/BCO-leaky_g-128x2-128x2-start-timesignal-doubled-ep200-b64-prepend-at-187-testval.csv"
#IFILE="/home/user/repos/masters/models/validation/naiveBC-RNNx128x2-olddata-ep300-norm-start-timesignal-trainval.csv"
IFILE="/home/user/repos/masters/models/validation/naiveBCx1024x2-olddata-fixed-data-fastrelease-ep20-norm-start-notimesignal-trainval.csv"


class ModelPerformanceDescriptor():

    def __init__(self, filename):
        self.filename = Path(filename)
        self._load_data()

    def _load_data(self):
        self._df = pd.read_csv(self.filename)
    
    def _slice_fn(self, start=0, stop=None):
        return self._split_fn()[start:stop]

    def _index_fn(self, index):
        return self._split_fn()[index]
    
    def _split_fn(self):
        return self.filename.stem.split("-")
    
    def _split_ds_mod_columns(self):
        slice_offs = 1 if self._ds_timesignal else 0
        return list(self._df.columns[slice_offs:8+slice_offs]), list(self._df.columns[11+slice_offs:])
    
    def _zip_ds_mod_columns(self):
        x, y = self._split_ds_mod_columns()
        return [z for z in zip(x,y)]
    
    def _split_and_flatten(self):
        dcols, mcols = self._split_ds_mod_columns()
        flat_data = self._df[dcols].to_numpy().flatten()
        flat_generated = self._df[mcols].to_numpy().flatten()
        return flat_data, flat_generated

    # naiveBC, recurrentBC, BCO-GAN
    def get_model_type(self):
        base_class, optional = self._slice_fn(0,2)
        self._nbc = ("naiveBC" in base_class and "RNN" not in optional)
        self._rnn = "RNN" in optional
        self._gan = base_class == "BCO"
    
    # number of neurons
    def get_model_parameter_count(self):

        if self._nbc:
            self._params = int(self._index_fn(0).split("x")[1])
        elif self._rnn:
            self._params = int(self._index_fn(1).split("x")[1])
        else:
            self._params = int(self._index_fn(2).split("x")[0])
        
        if self._gan:
            self._d_params = int(self._index_fn(3).split("x")[0])
        else:
            self._d_params = -1

    def get_dataset_type(self):
        self._ds_val_type_train = self._index_fn(-1) == "trainval"
        self._ds_val_type_test = not self._ds_val_type_train
        self._ds_new_data = ("newdata" in self.filename.stem)
        self._ds_old_data = not self._ds_new_data
        self._ds_state_history = 3 if "doubled" in self.filename.stem else 1
        self._ds_prepend = ("-prep" in self.filename.stem)
        self._ds_timesignal = not ("notimesignal" in self.filename.stem)
        
    def get_epochs(self):
        if "-at-" in self._split_fn():
            self._ep = int(self._index_fn(-2))
        else:
            self._ep = int([y for y in filter(lambda x: "ep" in x, self._split_fn())][0].replace("ep", ""))

    def get_metric_data_row(self, index=0):
        self.get_model_type()
        self.get_model_parameter_count()
        self.get_epochs()
        self.get_dataset_type()
        data_d = {
            "NaiveBC"       :   self._nbc,
            "RNN-BC"        :   self._rnn,
            "GAN-BC"        :   self._gan,
            "Params"        :   self._params,
            "D-Params"      :   self._d_params,
            "Epochs"        :   self._ep,
            "Old dataset"   :   self._ds_old_data,
            "New dataset"   :   self._ds_new_data,
            "Time signal"   :   self._ds_timesignal,
            "State history" :   self._ds_state_history,
            "Prepend"       :   self._ds_prepend,
            "Pearson"       :   self.pearson_correlation_coefficient(),
            "Euclidean g"   :   self.global_euclidean_distance(),
            "Manhattan g"   :   self.global_manhattan_distance(),
            "Cosine"        :   self.cosine_similarity_metric(),
            "Euclidean m"   :   self.mean_euclidean_distance(),
            "Manhattan m"   :   self.mean_manhattan_distance(),
            "Pos error"     :   self.mean_position_error(),
            "Rot error"     :   self.mean_angular_error(),
            "Release error" :   self.release_signal_error(),
            "Filename"      :   self.filename.stem
        }
        cols = data_d.keys()
        return pd.DataFrame(data=data_d, columns=cols, index=[index])

    def mean_euclidean_distance(self, dcols=None, mcols=None):
        return self._mean_stepwise_minkowski_distance(dcols, mcols, 2)

    def mean_manhattan_distance(self, dcols=None, mcols=None):
        return self._mean_stepwise_minkowski_distance(dcols, mcols, 1)

    def _mean_stepwise_minkowski_distance(self, dcols=None, mcols=None, degree=2):
        dcols, mcols = (dcols, mcols) if not (dcols is None or mcols is None) else self._split_ds_mod_columns()
        col_reductions = []
        for dcol, mcol in zip(dcols, mcols):
            col_reductions.append(self._df[[dcol,mcol]].diff(axis=1).pow(degree).abs().iloc[:,1])
        reduction_df = pd.concat(col_reductions, axis=1)
        return reduction_df.sum(axis=1).pow(1/degree).mean()

    def global_euclidean_distance(self):
        pass

    def global_manhattan_distance(self):
        pass

    def cosine_similarity_metric(self):
        d,g = self._split_and_flatten()
        return np.dot(d, g) / (np.linalg.norm(d) * np.linalg.norm(g))

    def pearson_correlation_coefficient(self):
        d, g = self._split_and_flatten()
        return np.corrcoef(d, g)[0,1] # returns n-row/column corr matrix. Need to specify corr between 0 and 1

    def mean_position_error(self):
        dcols, mcols = self._split_ds_mod_columns()
        slice_offs = 1 if self._ds_timesignal else 0
        return self.mean_euclidean_distance(dcols[slice_offs:3+slice_offs], mcols[slice_offs:3+slice_offs])

    def _row_normalize(self, array):
        row_norms = np.sqrt(np.square(array).sum(axis=1))
        return array / row_norms.reshape(-1,1)

    def mean_angular_error(self):
        dcols, mcols = self._split_ds_mod_columns()
        slice_offs = 1 if self._ds_timesignal else 0
        d_rots = self._df[dcols[slice_offs+3:7+slice_offs]].values
        m_rots = self._df[mcols[slice_offs+3:7+slice_offs]].values
        m_rots = self._row_normalize(m_rots)
        inner_prod = np.sum(d_rots * m_rots, axis=1)
        angular_error = np.arccos(2 * (inner_prod ** 2) - 1)
        return np.mean(angular_error)
    
    def _thresh(self, values, cutoff=0.5):
        return values >= cutoff # works because values is a numpy array

    def release_signal_error(self):
        dcols, mcols = self._split_ds_mod_columns()
        d_rel, m_rel = dcols[-1], mcols[-1]
        return 1 - self._df.loc[lambda d: d[d_rel] == self._thresh(d[m_rel].values)].size / self._df.size



if __name__ == "__main__":
    fnames = sys.argv
    fnames = [IFILE]
    i = 0
    df = pd.DataFrame()
    for fname in fnames:
        model_eval = ModelPerformanceDescriptor(fname)
        row = model_eval.get_metric_data_row(i)
        i += 1
        df = pd.concat([df,row], axis=0)
        pass
