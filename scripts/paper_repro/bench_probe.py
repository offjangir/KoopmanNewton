"""Time one sklearn LinearRegression fit of the probe size (1000 x 128), and show BLAS info."""
import time, numpy as np, sklearn, scipy
from sklearn.linear_model import LinearRegression
np.show_config()
from threadpoolctl import threadpool_info
for t in threadpool_info(): print(t)
X = np.random.randn(1000, 128).astype(np.float32); y = np.random.randn(1000)
for n in (1000, 100000):
    X = np.random.randn(n, 128).astype(np.float32); y = np.random.randn(n)
    t = time.time(); LinearRegression().fit(X, y); print(n, "sklearn fit s", time.time() - t)
    t = time.time(); np.linalg.lstsq(X.astype(np.float64), y, rcond=None); print(n, "numpy lstsq s", time.time() - t)
    t = time.time(); scipy.linalg.lstsq(X.astype(np.float64), y); print(n, "scipy lstsq s", time.time() - t)
