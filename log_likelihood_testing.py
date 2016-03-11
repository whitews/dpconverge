import pandas as pd
import numpy as np
from matplotlib import pyplot
from dpconverge.data_set import DataSet

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

component_list = range(3, 8)
burn_in = 0
iteration_count = 2000
seeds = range(1, 9)

run_data_frames = []

cmap = pyplot.cm.get_cmap('jet')
colors = [cmap(i) for i in np.linspace(0, 1, len(component_list))]
alphas = np.linspace(1, 0.4, len(component_list))

fig = pyplot.figure(figsize=(8, 8))

for i, comp_count in enumerate(component_list):
    for seed in seeds:
        ds = DataSet(parameter_count=2)

        ds.add_blob(1, blob1)
        ds.add_blob(2, blob2)
        ds.add_blob(3, blob3)
        ds.add_blob(4, blob4)
        ds.add_blob(5, blob5)

        ds.cluster(
            component_count=comp_count,
            burn_in=burn_in,
            iteration_count=iteration_count,
            random_seed=seed
        )

        log_likelihoods = ds.get_log_likelihood_trace()

        comp_df = pd.DataFrame(
            {
                'comp': comp_count,
                'seed': seed,
                'iter': range(iteration_count),
                'likelihood': log_likelihoods
            }
        )

        comp_df.plot(
            x='iter',
            y='likelihood',
            c=colors[i],
            linewidth=1,
            alpha=alphas[i]
        )

        run_data_frames.append(comp_df)

df = pd.concat(run_data_frames, ignore_index=True)
df.to_pickle('log_like_data.pkl')

pyplot.show()
