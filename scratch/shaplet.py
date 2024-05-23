from sktime.transformations.panel.shapelet_transform import (
    RandomShapeletTransform
)
from sktime.datasets import load_unit_test
X_train, y_train = load_unit_test(split="train", return_X_y=True)
t = RandomShapeletTransform(
    n_shapelet_samples=500,
    max_shapelets=10,
    batch_size=100,
) 
t.fit(X_train, y_train) 
RandomShapeletTransform(...)
X_t = t.transform(X_train)

import matplotlib.pyplot as plt
def plot(x):
    plt.plot(x)
    plt.show()

plot(X_train['dim_0'])
plot(X_t['0'])
