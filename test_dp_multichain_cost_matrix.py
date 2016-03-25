from dpconverge.data_set import DataSet
from matplotlib import pyplot
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from dpmix.utils import mvn_weighted_logged, sample_discrete
from dpmix.munkres import munkres, _get_cost


def update_labels(data, mus, sigmas, pis):
    densities = mvn_weighted_logged(data, mus, sigmas, pis)
    Z = np.asarray(densities.argmax(1), dtype='i')

    return sample_discrete(densities).squeeze(), Z


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
    random_state=3
)

ds = DataSet(parameter_count=2)

ds.add_blob(1, blob1)
ds.add_blob(2, blob2)
ds.add_blob(3, blob3)
ds.add_blob(4, blob4)
ds.add_blob(5, blob5)


ds.plot_blobs(ds.classifications, x_lim=[0, 4], y_lim=[0, 4])

component_count = 7
iteration_count = 100

last_iterations = []

for seed in [1, 2]:
    ds.results = None

    ds.cluster(
        component_count=component_count,
        burn_in=1000,
        iteration_count=iteration_count,
        random_seed=seed
    )

    ds.plot_classifications(iteration_count - 1)

    last_iterations.append(
        ds.raw_results.get_iteration(iteration_count - 1)
    )

# get reference from 1st run
labels, zref = update_labels(
    np.vstack(ds.blobs.values()),
    last_iterations[0].mus,
    last_iterations[0].sigmas,
    last_iterations[0].pis
)

c0 = np.zeros((component_count, component_count), dtype=np.float)
for j in xrange(component_count):
    c0[j, :] = np.sum(zref == j)

cost = c0.copy()

# get zhat from 2nd run
labels, zhat = update_labels(
    np.vstack(ds.blobs.values()),
    last_iterations[1].mus,
    last_iterations[1].sigmas,
    last_iterations[1].pis
)

_get_cost(zref, zhat, cost)

_, iii = np.where(munkres(cost))

pass
