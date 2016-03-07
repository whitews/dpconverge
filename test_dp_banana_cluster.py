from dpconverge.data_set import DataSet
import numpy as np

n_features = 2
points_per_feature = 100
centers = [[2, 2], [4, 4]]

ds = DataSet(parameter_count=2)

n_samples = 500

outer_circ_x = 1.0 + np.cos(np.linspace(0, np.pi, n_samples)) / 2
outer_circ_y = 0.5 + np.sin(np.linspace(0, np.pi, n_samples))

X = np.vstack((outer_circ_x, outer_circ_y)).T

X[:, 0] += (np.random.rand(500) - 0.5) / 16
X[:, 1] += (np.random.rand(500) - 0.5) / 16

X[:, 0] += (np.random.rand(500) - 0.5) / 16
X[:, 1] += (np.random.rand(500) - 0.5) / 16

ds.add_blob(1, X)

ds.plot_blobs(ds.classifications, x_lim=[0, 6], y_lim=[0, 6])
ds.cluster(
    component_count=4,
    burn_in=2,
    iteration_count=50,
    random_seed=123
)

ds.plot_animated_trace()

# ds.plot_iteration_traces(0)
# ds.plot_iteration_traces(1)
# ds.plot_iteration_traces(2)
# ds.plot_iteration_traces(3)
