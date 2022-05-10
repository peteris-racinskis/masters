import sys
from pathlib import Path
from os import listdir

import pandas as pd
import numpy as np

DIR="/home/user/repos/masters/models/validation/"
#DIR=""
OFILE="models/evaluation_data.csv"
#IFILE="/home/user/repos/masters/models/validation/BCO-leaky_g-128x2-128x2-start-timesignal-doubled-ep200-b64-prepend-at-187-testval.csv"
#IFILE="/home/user/repos/masters/models/validation/naiveBC-RNNx128x2-olddata-ep300-norm-start-timesignal-trainval.csv"
#IFILE="/home/user/repos/masters/models/validation/naiveBCx1024x2-olddata-fixed-data-fastrelease-ep20-norm-start-notimesignal-trainval.csv"
IFILE="/home/user/repos/masters/models/validation/naiveBC-RNNx128x2-newdata-ep1200-norm-start-timesignal-trainval.csv"
SAMPLE_RATE=100


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
    
    def _split_columns_with_slice(self, start=0, stop=None):
        dcols, mcols = self._split_ds_mod_columns()
        return dcols[start:stop], mcols[start:stop]
    
    def _split_pos_columns(self):
        return self._split_columns_with_slice(0,3)
    
    def _split_rot_columns(self):
        return self._split_columns_with_slice(3,7)

    def _split_release_columns(self):
        drel, mrel = self._split_columns_with_slice(-1)
        return drel[0], mrel[0]

    def _get_target_columns(self):
        slice_offs = 1 if self._ds_timesignal else 0
        return list(self._df.columns[slice_offs+8:11+slice_offs])
    
    def _zip_ds_mod_columns(self):
        x, y = self._split_ds_mod_columns()
        return [z for z in zip(x,y)]
    
    def _split_and_flatten(self):
        dcols, mcols = self._split_ds_mod_columns()
        flat_data = self._df[dcols].to_numpy().flatten()
        flat_generated = self._df[mcols].to_numpy().flatten()
        return flat_data, flat_generated

    def get_model_type(self):
        base_class, optional = self._slice_fn(0,2)
        self._nbc = ("naiveBC" in base_class and "RNN" not in optional)
        self._rnn = "RNN" in optional
        self._gan = base_class == "BCO"
    
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

    def get_learning_rate(self):
        if self._rnn and not ("lr5" in self._split_fn()):
            self._learning_rate = 10e-4
        else:
            self._learning_rate = 10e-5
        
    def get_epochs(self):
        if "at" in self._split_fn():
            self._ep = int(self._index_fn(-2))
        else:
            self._ep = int([y for y in filter(lambda x: "ep" in x, self._split_fn())][0].replace("ep", ""))

    def get_metric_data_row(self, index=0):
        self.get_model_type()
        self.get_model_parameter_count()
        self.get_epochs()
        self.get_dataset_type()
        self.get_learning_rate()
        self.release_parameter_similarity()
        data_d = {
            "NaiveBC"       :   self._nbc,
            "RNN-BC"        :   self._rnn,
            "GAN-BC"        :   self._gan,
            "Test"          :   self._ds_val_type_test,
            "Train"         :   self._ds_val_type_train,
            "Params"        :   self._params,
            "D-Params"      :   self._d_params,
            "Epochs"        :   self._ep,
            "Old dataset"   :   self._ds_old_data,
            "New dataset"   :   self._ds_new_data,
            "Time signal"   :   self._ds_timesignal,
            "State history" :   self._ds_state_history,
            "Prepend"       :   self._ds_prepend,
            "Learning rate" :   self._learning_rate,
            "Pearson"       :   self.pearson_correlation_coefficient(),
            "Euclidean g"   :   self.global_euclidean_distance(),
            "Manhattan g"   :   self.global_manhattan_distance(),
            "Cosine"        :   self.cosine_similarity_metric(),
            "Euclidean m"   :   self.mean_euclidean_distance(),
            "Manhattan m"   :   self.mean_manhattan_distance(),
            "Pos error"     :   self.mean_position_error(),
            "Rot error"     :   self.mean_angular_error(),
            "Release error" :   self.release_signal_error(),
            "Rel. pos. err" :   self._throw_position_error,
            "Rel. vel. err" :   self._throw_velocity_error,
            "Missed by"     :   self._throw_error_estimate,
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
        return self._global_norm_corrected()

    def global_manhattan_distance(self):
        return self._global_norm_corrected(1)
        
    def _global_norm_corrected(self, order=None):
        d, g = self._split_and_flatten()
        return np.linalg.norm(d - g, ord=order) / d.size

    def cosine_similarity_metric(self):
        d, g = self._split_and_flatten()
        return np.dot(d, g) / (np.linalg.norm(d) * np.linalg.norm(g))

    def pearson_correlation_coefficient(self):
        d, g = self._split_and_flatten()
        return np.corrcoef(d, g)[0,1] # returns n-row/column corr matrix. Need to specify corr between 0 and 1

    def mean_position_error(self):
        dcols, mcols = self._split_pos_columns()
        return self.mean_euclidean_distance(dcols, mcols)

    @staticmethod
    def _row_normalize(array):
        row_norms = np.sqrt(np.square(array).sum(axis=1))
        return array / row_norms.reshape(-1,1)

    def mean_angular_error(self):
        dcols, mcols = self._split_rot_columns()
        d_rots = self._df[dcols].values
        m_rots = self._df[mcols].values
        m_rots = self._row_normalize(m_rots)
        inner_prod = np.sum(d_rots * m_rots, axis=1)
        angular_error = np.arccos(2 * (inner_prod ** 2) - 1)
        return np.mean(angular_error)
    
    @staticmethod
    def _thresh(values, cutoff=0.5):
        return values >= cutoff # works because values is a numpy array

    def release_signal_error(self):
        d_rel, m_rel = self._split_release_columns()
        return 1 - self._df.loc[lambda d: d[d_rel] == self._thresh(d[m_rel].values)].size / self._df.size

    def _parabola_diff(self, d_index, m_index):
        dcols, mcols = self._split_pos_columns()
        tcols = self._get_target_columns()

        target_z = self._df[tcols[2]].iloc[d_index].values
        target_offs = np.concatenate([np.zeros((len(d_index),2)), target_z.reshape(-1,1)], axis=1)

        d_initial_positions = self._df[dcols].iloc[d_index].values - target_offs
        m_initial_positions = self._df[mcols].iloc[m_index].values - target_offs

        d_initial_velocities = self._df[dcols].diff().iloc[d_index].values * SAMPLE_RATE
        m_initial_velocities = self._df[mcols].diff().iloc[m_index].values * SAMPLE_RATE

        g = -9.81

        accelerations = np.repeat(np.asarray([0.0,0.0,g/2]).reshape(1,-1), repeats=len(d_index), axis=0)

        d_coeff_tensor = np.stack([d_initial_positions, d_initial_velocities, accelerations])
        m_coeff_tensor = np.stack([m_initial_positions, m_initial_velocities, accelerations])
        d_coeff_tensor = np.transpose(d_coeff_tensor, axes=(1,2,0))
        m_coeff_tensor = np.transpose(m_coeff_tensor, axes=(1,2,0))

        t_d = self._parabola_intersect_times(d_coeff_tensor)
        t_m = self._parabola_intersect_times(m_coeff_tensor)

        d_predicted_positions = (t_d * d_coeff_tensor).sum(axis=2)
        m_predicted_positions = (t_m * m_coeff_tensor).sum(axis=2)

        self._throw_error_estimate = np.linalg.norm(d_predicted_positions - m_predicted_positions, axis=1).mean()

    @staticmethod
    def _parabola_intersect_times(coeff_tensor):
        z_intersect_times = []
        for z_coefs in coeff_tensor[:,2,:]:
            roots = np.roots(z_coefs[::-1])
            root = np.where(roots > 0, roots, 0).sum()
            z_intersect_times.append(root)
        
        z_intersects = np.asarray(z_intersect_times)

        t = np.asarray(z_intersects).reshape(-1,1)
        t = np.concatenate([np.ones_like(t), t, t**2], axis=1)
        t = np.stack([t] * 3)
        t = np.transpose(t, axes=(1,0,2))
        return t

    def release_parameter_similarity(self):

        self._throw_velocity_error = np.nan
        self._throw_position_error = np.nan
        self._throw_error_estimate = np.nan

        dcols, mcols = self._split_pos_columns()
        drel, mrel = self._split_release_columns()

        d_release_points = self._df.diff().loc[lambda d: d[drel] > 0]
        d_final_positions = self._df.iloc[d_release_points.index][dcols].values
        d_final_velocities = self._df[dcols].diff().iloc[d_release_points.index].values * SAMPLE_RATE
        
        tcols = self._get_target_columns()
        demo_id = self._df[tcols].diff().any(axis=1).astype(int).cumsum()
        m_release_points = self._thresh(self._df[mrel]).astype(int).diff()
        m_release_points = pd.concat([demo_id, m_release_points], axis=1).loc[lambda d: (d[d.columns[-1]] > 0) & (pd.notna(d[d.columns[-1]]))]
        m_release_points = m_release_points.drop_duplicates(subset=m_release_points.columns[0])

        if not m_release_points.shape[0] == d_final_velocities.shape[0]:
            return
        
        m_final_positions = self._df[mcols].iloc[m_release_points.index].values
        m_final_velocities = self._df[mcols].diff().iloc[m_release_points.index].values * SAMPLE_RATE

        d_vel_magnitudes_squared = np.sum(d_final_velocities ** 2, axis=1)
        dot_products = np.sum(d_final_velocities * m_final_velocities, axis=1)
        vector_dissimilarities = np.abs(d_vel_magnitudes_squared - dot_products)
        positional_errors = np.sqrt(np.sum(np.square(m_final_positions - d_final_positions), axis=1))

        self._throw_velocity_error = np.mean(vector_dissimilarities)
        self._throw_position_error = np.mean(positional_errors)

        self._parabola_diff(d_release_points.index, m_release_points.index)


if __name__ == "__main__":
    fnames = sys.argv
    #fnames = [IFILE]
    fnames = [x for x in filter(lambda x: ".csv" in x, listdir(DIR))]
    fnames.sort()
    i = 0
    df = pd.DataFrame()
    for fname in fnames:
        print(f"processing: {fname}")
        model_eval = ModelPerformanceDescriptor(DIR+fname)
        row = model_eval.get_metric_data_row(i)
        i += 1
        df = pd.concat([df,row], axis=0)
    df.to_csv(OFILE, index=False)
