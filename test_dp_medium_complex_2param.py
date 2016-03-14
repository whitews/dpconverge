from dpconverge.data_set import DataSet
from matplotlib import pyplot
from sklearn.datasets.samples_generator import make_blobs

n_features = 2
points_per_feature = 100
centers = [[2, 1.35], [2, 2], [2, 3], [2.5, 1.5], [2.5, 2], [2.5, 2.5]]

blob1, y1 = make_blobs(
    n_samples=1000,
    n_features=1,
    centers=centers[0],
    cluster_std=[0.1, 0.15],
    random_state=1
)

blob2, y2 = make_blobs(
    n_samples=6000,
    n_features=1,
    centers=centers[1],
    cluster_std=[0.2, 0.3],
    random_state=2
)

blob3, y3 = make_blobs(
    n_samples=3000,
    n_features=1,
    centers=centers[2],
    cluster_std=[0.2, 0.1],
    random_state=2
)

blob4, y4 = make_blobs(
    n_samples=250,
    n_features=1,
    centers=centers[3],
    cluster_std=[0.1, 0.1],
    random_state=2
)

blob5, y5 = make_blobs(
    n_samples=250,
    n_features=1,
    centers=centers[4],
    cluster_std=[0.1, 0.1],
    random_state=2
)

ds = DataSet(parameter_count=2)

ds.add_blob(1, blob1)
ds.add_blob(2, blob2)
ds.add_blob(3, blob3)
ds.add_blob(4, blob4)
ds.add_blob(5, blob5)

# ds.plot_blobs(ds.classifications, x_lim=[0, 4], y_lim=[0, 4])

component_count = 5

ds.cluster(
    component_count=component_count,
    burn_in=1000,
    iteration_count=1000,
    random_seed=1
)

valid_components = ds.get_valid_components()

print "Recommended component count: ", len(valid_components)

# for i in range(component_count):
#     if i in valid_components:
#         ds.plot_iteration_traces(i)
#
# for i in range(component_count):
#     if i not in valid_components:
#         print "Possible invalid Component"
#         ds.plot_iteration_traces(i)

# ds.plot_animated_trace()
ds.plot_log_likelihood_trace(use_scipy=True)
pyplot.show()
