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

component_count = 5

ds.plot_blobs(ds.labels, x_lim=[0, 6], y_lim=[0, 6])
ds.cluster(
    component_count=component_count,
    burn_in=1000,
    iteration_count=200,
    random_seed=123
)

valid_components = ds.get_valid_components()

print "Recommended component count: ", len(valid_components)

for i in range(component_count):
    if i in valid_components:
        ds.plot_iteration_traces(i)

for i in range(component_count):
    if i not in valid_components:
        print "Possible invalid Component"
        ds.plot_iteration_traces(i)

ds.plot_animated_trace()
