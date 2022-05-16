from calendar import EPOCH
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
DREL="Release error"
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
        self._stepwise_metrics = {x:MIN for x in [EUC_M, MAN_M, DPOS, DROT, DREL]}
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

    def filter_by_column_keys(self, cols, values):
        for c,v in zip(cols,values):
            self._masked = self._masked.loc[lambda d: d[c] == v]


    def get_uniques(self, cols, mtype):
        uniques = {}
        self.filter_by_col(mtype)
        for c in cols:
            uniques[c] = self._masked[c].unique()
        self.clear_mask()
        return uniques

    @staticmethod
    def count_permutations(iterables):
        x = 1
        for i in iterables:
            x *= len(i)
        return x

    @staticmethod
    def get_permutation(index, iterables):
        lengths = []
        permutation = []
        for iterable in iterables:
            lengths.append(len(iterable))
        for i in range(len(iterables)):
            permutation.append(iterables[i][index % lengths[i]])
        return permutation

    @staticmethod
    def list_exclude(element, iterable):
        ret = []
        for e in iterable:
            if not element == e:
                ret.append(e)
        return ret
    
    @staticmethod
    def perm_key(cols, permutation):
        ret = ""
        for c,p in zip(cols, permutation):
            ret += f"{c}-{p}:"
        return ret
    
    def _gen_ind_seq(self, mtype, metrics, independent_columns):
        mtype_out = COLUMN_RENAME[mtype]
        uniques_d = self.get_uniques(independent_columns, mtype)

        for ic in independent_columns:
            if len(uniques_d[ic]) < 3:
                continue 
            rest = self.list_exclude(ic, independent_columns)
            uniques = [uniques_d[x] for x in rest]
            n_perm = self.count_permutations(uniques)
            permutations = []

            for p in range(n_perm):
                permutations.append(self.get_permutation(p, uniques))
            
            for metric in metrics:
                odf = pd.DataFrame()
                for permutation in permutations:
                    key = self.perm_key(rest, permutation)
                    self.clear_mask()
                    self.filter_by_col(mtype)
                    self.filter_by_column_keys(rest, permutation)
                    data = self._masked[[ic,metric]].values
                    key_list = [key] * data.shape[0]
                    data_dict = {
                        "Key": key_list,
                        "Argument": data[:,0],
                        "Value": data[:,1]
                    }
                    odf = pd.concat([odf,pd.DataFrame(data=data_dict)])

                filename = f"models/comparison_tables/independent-{mtype_out}-{ic}-{metric}.csv"
                print(f"processed: {filename} n_perm = {n_perm}")
                odf.to_csv(filename, index=False)

    def generate_independent_sequences(self):
                
        independent_columns_naive = [EPOCHS, PARAMS, OLD, TIME, TRAIN]
        self._gen_ind_seq(NAIVE, self._metrics.keys(), independent_columns_naive)

        independent_columns_rnn = [EPOCHS, PARAMS, OLD, LR, TRAIN]
        self._gen_ind_seq(RNN, self._metrics.keys(), independent_columns_rnn)

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
            print(f"processed: {filename}")
            table.to_csv(filename, index=False)


if __name__ == "__main__":
    df = pd.read_csv(IFILE)
    dataset = ModelEvalDataset(df)
    dataset.generate_sequence_dfs()
    dataset.generate_categorical_dfs()
    dataset.generate_independent_sequences()