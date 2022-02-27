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
    res = model.predict(X).flatten()
    return pd.concat([df, pd.Series(data=res, name=ycol+"pred", index=df.index)], axis=1)

def predict(df: pd.DataFrame, xmod, ymod, zmod) -> pd.DataFrame:
    pass # figure it out tomorrow or after

if __name__ == "__main__":
    fnames = sys.argv if len(sys.argv) > 1 else [IFILE]
    for fname in fnames:
        ofname = fname.replace("-smooth","-labelled")
        df = pd.read_csv(fname).loc[lambda d: (d[FREEFALL] == 10) & (d[PASSED] == -10)]
        xmod = fit_poly(df, TIME, BX)
        ymod = fit_poly(xmod, TIME, BY)
        zmod = fit_poly(ymod, TIME, BZ, degree=2)
        last = zmod.loc[lambda d: d[BZ] >= d[CZ]].iloc[-1][[BX,BY]]
        pass
