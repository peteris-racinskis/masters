import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Goals:
# 2. fit linear model y=My(t), x=Mx(t)
# 3. fit quadratic model z=Mz(t)
# 4. find intersection Mz(t0) = CatchNet.position.z <- use np.polynomial
# 5. let endpoint = (Mx(t0),My(t0),Mz(t0))


if __name__ == "__main__":
    pass