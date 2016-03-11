from dpconverge.data_set import DataSet
from dpconverge.data_collection import DataCollection

from sklearn.datasets.samples_generator import make_blobs

dc = DataCollection()
centers = [[2, 1.35], [2, 2], [2, 3], [2.5, 1.5], [2.5, 2], [2.5, 2.5]]

# begin creating blobs for 1st data set
ds1_blob1, y1 = make_blobs(
    n_samples=1000,
    n_features=1,
    centers=centers[0],
    cluster_std=[0.1, 0.15],
    random_state=1
)

ds1_blob2, y2 = make_blobs(
    n_samples=6000,
    n_features=1,
    centers=centers[1],
    cluster_std=[0.2, 0.3],
    random_state=2
)

ds1_blob3, y3 = make_blobs(
    n_samples=3000,
    n_features=1,
    centers=centers[2],
    cluster_std=[0.2, 0.1],
    random_state=2
)

ds1_blob4, y4 = make_blobs(
    n_samples=250,
    n_features=1,
    centers=centers[3],
    cluster_std=[0.1, 0.1],
    random_state=2
)

ds1_blob5, y5 = make_blobs(
    n_samples=250,
    n_features=1,
    centers=centers[4],
    cluster_std=[0.1, 0.1],
    random_state=2
)

ds = DataSet(parameter_count=2)

ds.add_blob(1, ds1_blob1)
ds.add_blob(2, ds1_blob2)
ds.add_blob(3, ds1_blob3)
ds.add_blob(4, ds1_blob4)
ds.add_blob(5, ds1_blob5)

ds.plot_blobs(ds.classifications, x_lim=[0, 4], y_lim=[0, 4])

dc.add_data_set(ds)

# begin creating blobs for 2nd data set
ds2_blob1, y1 = make_blobs(
    n_samples=1000,
    n_features=1,
    centers=centers[0],
    cluster_std=[0.1, 0.15],
    random_state=4
)

ds2_blob2, y2 = make_blobs(
    n_samples=6000,
    n_features=1,
    centers=centers[1],
    cluster_std=[0.2, 0.3],
    random_state=4
)

ds2_blob3, y3 = make_blobs(
    n_samples=3000,
    n_features=1,
    centers=centers[2],
    cluster_std=[0.2, 0.1],
    random_state=4
)

ds2_blob4, y4 = make_blobs(
    n_samples=250,
    n_features=1,
    centers=centers[3],
    cluster_std=[0.1, 0.1],
    random_state=4
)

ds2_blob5, y5 = make_blobs(
    n_samples=250,
    n_features=1,
    centers=centers[4],
    cluster_std=[0.1, 0.1],
    random_state=4
)

ds = DataSet(parameter_count=2)

ds.add_blob(1, ds2_blob1)
ds.add_blob(2, ds2_blob2)
ds.add_blob(3, ds2_blob3)
ds.add_blob(4, ds2_blob4)
ds.add_blob(5, ds2_blob5)

ds.plot_blobs(ds.classifications, x_lim=[0, 4], y_lim=[0, 4])

dc.add_data_set(ds)

component_count = 6

dc.cluster(
    component_count=component_count,
    burn_in=1000,
    iteration_count=2000,
    random_seed=1
)

for ds in dc.data_sets:
    valid_components = ds.get_valid_components()

    print "Recommended component count: ", len(valid_components)

    for i in range(component_count):
        if i in valid_components:
            ds.plot_iteration_traces(i)

    for i in range(component_count):
        if i not in valid_components:
            print "Possible invalid Component"
            ds.plot_iteration_traces(i)

    ds.plot_log_likelihood_trace()
    ds.plot_animated_trace()
