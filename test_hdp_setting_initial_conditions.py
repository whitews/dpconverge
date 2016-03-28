from dpconverge.data_set import DataSet
from dpconverge.data_collection import DataCollection
from sklearn.datasets.samples_generator import make_blobs


centers = [
    [2, 1.35],
    [2, 2],
    [2, 3],
    [2.5, 1.5],
    [2.5, 2],
    [2.5, 2.5],
    [1, 1]
]

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
    random_state=3
)

ds = DataSet(parameter_count=2)

ds.add_blob(1, blob1)
ds.add_blob(2, blob2)
ds.add_blob(3, blob3)
ds.add_blob(4, blob4)
ds.add_blob(5, blob5)

ds.plot_blobs(ds.labels, x_lim=[0, 4], y_lim=[0, 4])

# begin creating blobs for 2nd data set
ds2_blob1, y6 = make_blobs(
    n_samples=1000,
    n_features=1,
    centers=centers[0],
    cluster_std=[0.1, 0.15],
    random_state=4
)

ds2_blob2, y7 = make_blobs(
    n_samples=6000,
    n_features=1,
    centers=centers[1],
    cluster_std=[0.2, 0.3],
    random_state=4
)

ds2_blob3, y8 = make_blobs(
    n_samples=3000,
    n_features=1,
    centers=centers[2],
    cluster_std=[0.2, 0.1],
    random_state=4
)

ds2_blob4, y9 = make_blobs(
    n_samples=250,
    n_features=1,
    centers=centers[5],
    cluster_std=[0.1, 0.1],
    random_state=4
)

ds2_blob5, y10 = make_blobs(
    n_samples=250,
    n_features=1,
    centers=centers[4],
    cluster_std=[0.1, 0.1],
    random_state=5
)

# rare blob
ds2_blob6, y11 = make_blobs(
    n_samples=4,
    n_features=1,
    centers=centers[6],
    cluster_std=[0.05, 0.05],
    random_state=5
)

ds2 = DataSet(parameter_count=2)

ds2.add_blob(1, ds2_blob1)
ds2.add_blob(2, ds2_blob2)
ds2.add_blob(3, ds2_blob3)
ds2.add_blob(4, ds2_blob4)
ds2.add_blob(5, ds2_blob5)
ds2.add_blob(6, ds2_blob6)

ds2.plot_blobs(ds2.labels, x_lim=[0, 4], y_lim=[0, 4])

# create our data collection to run HDP
dc = DataCollection()
dc.add_data_set(ds)
dc.add_data_set(ds2)

component_count, initial_conditions = dc.estimate_initial_conditions(
    max_comp=32,
    max_iter=5000
)

dc.cluster(
    component_count=component_count,
    burn_in=0,
    iteration_count=500,
    random_seed=1,
    initial_conditions=initial_conditions
)

# get valid components, but ignore weight since one sample may have
# a component the others do not. Since the mus and sigmas are the same
# for all samples, we can just test one sample
valid_components = dc.data_sets[0].get_valid_components(ignore_weight=True)

print "Recommended component count: ", len(valid_components)

for r_ds in dc.data_sets:
    for i in range(component_count):
        if i in valid_components:
            r_ds.plot_iteration_traces(i)

    for i in range(component_count):
        if i not in valid_components:
            print "Possible invalid Component"
            r_ds.plot_iteration_traces(i)

    r_ds.plot_log_likelihood_trace()
    r_ds.plot_animated_trace()

    pass
