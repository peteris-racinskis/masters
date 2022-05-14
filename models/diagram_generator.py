import pandas as pd
IFILE="models/evaluation_data.csv"

class ModelEvalDataset():

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._masked = df

    def _col_minmax(self, col, condition, callback=pd.DataFrame.max):
        m = callback(self._masked.loc[condition][col])
        return self._masked.loc[lambda d: d[col] == m].drop_duplicates(subset=col)
    
    def _partition_minmaxes(self, col, part, cb=pd.DataFrame.max):
        values = self._masked[part].unique()
        rows = []
        for v in values:
            condition = lambda x: x[part] == v
            rows.append(self._col_minmax(col, condition, cb))
        return pd.concat(rows)

    def partition_maxes(self, col, part):
        return self._partition_minmaxes(col, part)

    def partition_mins(self, col, part):
        return self._partition_minmaxes(col, part, pd.DataFrame.min)
    
    def filter_by_col(self, col):
        self._masked = self._df.loc[lambda d: d[col]]


    


if __name__ == "__main__":
    df = pd.read_csv(IFILE)
    dataset = ModelEvalDataset(df)
    dataset.filter_by_col("GAN-BC")
    dataset.partition_maxes("Pearson", "Epochs")
    pass