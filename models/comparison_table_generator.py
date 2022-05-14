import pandas as pd
IFILE="models/evaluation_data.csv"

NAIVE="NaiveBC"
RNN="RNN-BC"
GAN="GAN-BC"
TRAIN="Train"
OLD="Old dataset"
TIME="Time signal"
LR="Learning rate"
EPOCHS="Epochs"
PARAMS="Params"
PEARSON="Pearson"
EUC_G="Euclidean g"
MAN_G="Manhattan g"
COSINE="Cosine"
EUC_M="Euclidean m"
MAN_M="Manhattan m"
DPOS="Pos error"
DROT="Rot error"
RDPOS="Rel. pos. err"
RDVEL="Rel. vel. err"
RDXY="Missed by"

MIN="min"
MAX="max"

COLUMN_RENAME={
    GAN: "GAN",
    RNN: "RNN",
    NAIVE: NAIVE,
    None: "All"
}

class ModelEvalDataset():

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._masked = df
        self._cbs ={
            MAX: pd.DataFrame.max,
            MIN: pd.DataFrame.min
        }
        self._corpus_metrics = {
            PEARSON : MAX, 
            COSINE : MAX, 
            EUC_G : MIN, 
            MAN_G : MIN
        }
        self._stepwise_metrics = {x:MIN for x in [EUC_M, MAN_M, DPOS, DROT]}
        self._throw_metrics = {x:MIN for x in [RDPOS, RDVEL, RDXY]}
        self._metrics = {}
        for d in self._corpus_metrics, self._stepwise_metrics, self._throw_metrics:
            self._metrics.update(d)
        self._model_types = [NAIVE, RNN, GAN]

    def clear_mask(self):
        self._masked = self._df

    def _col_minmax(self, col, condition, cb):
        m = cb(self._masked.loc[condition][col])
        return self._masked.loc[lambda d: d[col] == m].drop_duplicates(subset=col)
    
    def _partition_minmaxes(self, col, part, cb):
        values = self._masked[part].unique()
        rows = []
        for v in values:
            condition = lambda x: x[part] == v
            rows.append(self._col_minmax(col, condition, cb))
        return pd.concat(rows)

    def partition_maxes(self, col, part):
        return self._partition_minmaxes(col, part, self._cbs[MAX])

    def partition_mins(self, col, part):
        return self._partition_minmaxes(col, part, self._cbs[MIN])
    
    def filter_by_col(self, col):
        if col == None:
            return
        self._masked = self._df.loc[lambda d: d[col]]

    def generate_categorical_dfs(self):

        self.clear_mask()

        class_index = self._df[NAIVE] * 1 + self._df[RNN] * 2 + self._df[GAN] * 3
        model_class = "ModelClass"
        class_index = class_index.rename(model_class)

        self._masked = pd.concat([self._masked, class_index], axis=1)
        sequence_keys = []
        for metric, cb in self._metrics.items():
            sequence_keys.append([None, metric, model_class, cb])
        
        self._create_tables(sequence_keys, "categorical")

    def generate_sequence_dfs(self):

        self.clear_mask()
        sequence_keys = []

        for metric, cb in self._metrics.items():
            for param in [EPOCHS, PARAMS, OLD, TIME]:
                sequence_keys.append([NAIVE, metric, param, cb])

        for metric, cb in self._metrics.items():
            for param in [EPOCHS, PARAMS, OLD, LR]:
                sequence_keys.append([RNN, metric, param, cb])
        
        for metric, cb in self._metrics.items():
            for param in [EPOCHS]:
                sequence_keys.append([GAN, metric, param, cb])
        
        self._create_tables(sequence_keys)

    def _create_tables(self, sequence_keys, table_type="sequences"):
        for mtype, metric, param, cb in sequence_keys:
            self.filter_by_col(mtype)
            mtype = COLUMN_RENAME[mtype]
            filename = f"models/comparison_tables/{table_type}-{mtype}-{param}-{metric}-{cb}.csv"
            table = self._partition_minmaxes(metric, param, self._cbs[cb])
            table = table.rename(columns={GAN:"GAN", RNN:"RNN"})
            table.to_csv(filename, index=False)


if __name__ == "__main__":
    df = pd.read_csv(IFILE)
    dataset = ModelEvalDataset(df)
    dataset.generate_sequence_dfs()
    dataset.generate_categorical_dfs()