from dpconverge.data_set import DataSet
import numpy as np

from sklearn.datasets.samples_generator import make_blobs

n_features = 3
points_per_feature = 100
centers = [[2, 2, 1], [2, 4, 2], [4, 2, 3], [4, 4, 4]]

ds = DataSet(parameter_count=n_features)
rnd_state = np.random.RandomState()
rnd_state.seed(3)

for i, center in enumerate(centers):
    X, y = make_blobs(
        n_samples=points_per_feature,
        n_features=n_features,
        centers=center,
        cluster_std=0.2,
        random_state=rnd_state.randint(128)
    )

    ds.add_blob(i, X)

component_count = 6

ds.plot_blobs(ds.classifications, x_lim=[0, 6], y_lim=[0, 6])
ds.plot_blobs(ds.classifications, x=0, y=2, x_lim=[0, 6], y_lim=[0, 6])

ds.cluster(
    component_count=component_count,
    burn_in=100,
    iteration_count=400,
    random_seed=1
)

valid_components = ds.get_valid_components()

print "Recommended component count: ", len(valid_components)

for i in range(component_count):
    if i in valid_components:
        ds.plot_iteration_traces(i)

# for i in range(component_count):
#     if i not in valid_components:
#         print "Possible invalid Component"
#         ds.plot_iteration_traces(i)

ds.plot_animated_trace(x_lim=[0, 6], y_lim=[0, 6])
