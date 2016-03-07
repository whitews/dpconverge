from dpconverge.data_set import DataSet
from dpconverge.data_collection import DataCollection

from sklearn.datasets.samples_generator import make_blobs

n_features = 2
points_per_feature = 100
centers = [[2, 2], [4, 4]]
seeds = [5, 6]

dc = DataCollection()

for i_ds in range(0, 2):
    ds = DataSet(parameter_count=2)

    for i, center in enumerate(centers):
        X, y = make_blobs(
            n_samples=points_per_feature,
            n_features=n_features,
            centers=center,
            cluster_std=0.3,
            random_state=seeds[i_ds]
        )

        ds.add_blob(i, X)

    ds.plot_blobs(ds.classifications, x_lim=[0, 6], y_lim=[0, 6])

    dc.add_data_set(ds)

dc.cluster(
    component_count=4,
    burn_in=2,
    iteration_count=50,
    random_seed=123
)

dc.data_sets[0].plot_iteration_traces(0)
