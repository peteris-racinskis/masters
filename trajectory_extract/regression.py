import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
IFILE="processed_data/demo-22-02-2022-10:44:48-thresh.csv"
TIME="Time"
BOTTLE="Bottle.position."
NET="CatchNet.position."
BX=BOTTLE+"x"
BY=BOTTLE+"y"
BZ=BOTTLE+"z"
PRED="pred"
TARGET="-t"
CZ=NET+"z"
FREEFALL="Freefall"
PASSED="Passed"

# Goals:
# 2. fit linear model y=My(t), x=Mx(t)
# 3. fit quadratic model z=Mz(t)
# 4. find intersection Mz(t0) = CatchNet.position.z <- use np.polynomial
# 5. let endpoint = (Mx(t0),My(t0),Mz(t0))

def fit_poly(df: pd.DataFrame, xcol: str, ycol: str, degree=1) -> pd.DataFrame:
    X = df[xcol].values.reshape(-1,1)
    X = X - X[0][0]
    Y = df[ycol].values.reshape(-1,1)
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X,Y)
    return model

def add_series_from_model(df: pd.DataFrame, model, X, name) -> pd.DataFrame:
    res = model.predict(X).flatten()
    return pd.concat([df, pd.Series(data=res, name=name+"pred", index=df.index)], axis=1)

def predict(df: pd.DataFrame, xmod, ymod, zmod, start) -> pd.DataFrame:
    T = df[TIME].values.reshape(-1,1) - start
    xp = add_series_from_model(df, xmod, T, BX)
    yp = add_series_from_model(xp, ymod, T, BY)
    return add_series_from_model(yp, zmod, T, BZ)

def find_target_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    pred = {s:s+PRED for s in [BX, BY, BZ]}
    last = df.loc[lambda d: d[pred[BZ]] >= d[CZ]].iloc[-1]
    tx, ty, tz = last[pred.values()]
    rows = [[tx, ty, tz]] * len(df)
    tname = {s:s+TARGET for s in pred.keys()}
    target = pd.DataFrame(data=rows, columns=tname.values(), index=df.index)
    return pd.concat([df, target], axis=1)


if __name__ == "__main__":
    fnames = sys.argv if len(sys.argv) > 1 else [IFILE]
    for fname in fnames:
        ofname = fname.replace("-thresh","-labelled")
        df = pd.read_csv(fname)
        df_ballistic = df.loc[lambda d: (d[FREEFALL] == 10) & (d[PASSED] == -10)]
        start = df_ballistic.iloc[0][TIME]
        xmod = fit_poly(df_ballistic, TIME, BX)
        ymod = fit_poly(df_ballistic, TIME, BY)
        zmod = fit_poly(df_ballistic, TIME, BZ, degree=2)
        pred = predict(df, xmod, ymod, zmod, start)
        labelled = find_target_coordinates(pred)
        labelled.to_csv(ofname, index=False)
