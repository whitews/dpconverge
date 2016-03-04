from dpconverge.data_set import DataSet

from sklearn.datasets.samples_generator import make_blobs

n_features = 2
points_per_feature = 100
centers = [[2, 2], [4, 4]]

ds = DataSet(parameter_count=2)

for i, center in enumerate(centers):
    X, y = make_blobs(
        n_samples=points_per_feature,
        n_features=n_features,
        centers=center,
        cluster_std=0.3,
        random_state=5
    )

    ds.add_blob(i, X)

ds.plot(ds.classifications, x_lim=[0, 6], y_lim=[0, 6])
ds.cluster(
    component_count=4,
    burn_in=2,
    iteration_count=50,
    random_seed=123
)
ds.plot_iteration_traces(0)
ds.plot_iteration_traces(1)
ds.plot_iteration_traces(2)
ds.plot_iteration_traces(3)
