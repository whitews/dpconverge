import pandas as pd
from dpconverge.data_set import DataSet

from sklearn.datasets.samples_generator import make_blobs

n_features = 2
points_per_feature = 100
centers = [[2, 1.35], [2, 2], [2, 3], [2.5, 1.5], [2.5, 2], [2.5, 2.5]]

blob1, y = make_blobs(
    n_samples=1000,
    n_features=1,
    centers=centers[0],
    cluster_std=[0.1, 0.15],
    random_state=1
)

blob2, y = make_blobs(
    n_samples=6000,
    n_features=1,
    centers=centers[1],
    cluster_std=[0.2, 0.3],
    random_state=2
)

blob3, y = make_blobs(
    n_samples=3000,
    n_features=1,
    centers=centers[2],
    cluster_std=[0.2, 0.1],
    random_state=2
)

blob4, y = make_blobs(
    n_samples=250,
    n_features=1,
    centers=centers[3],
    cluster_std=[0.1, 0.1],
    random_state=2
)

blob5, y = make_blobs(
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

#ds.plot_blobs(ds.classifications, x_lim=[0, 4], y_lim=[0, 4])

component_count = 6
burn_in = 0
iteration_count = 50

save_dir = "~/test_output/"

run_data_frames = []

for comp_count in range(4, component_count+1):
    for seed in range(1, 3):
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

        #ds.plot_animated_trace(iter_start=800)

        log_likelihoods = ds.get_log_likelihood_trace()

        run_data_frames.append(
            pd.DataFrame(
                {
                    'comp': comp_count,
                    'seed': seed,
                    'iter': range(iteration_count),
                    'likelihood': log_likelihoods
                }
            )
        )

        # fig = ds.plot_log_likelihood_trace()
        #
        # filename = "c%d_b%d_i%d_s%d.png" % (
        #     comp_count,
        #     burn_in,
        #     iteration_count,
        #     seed
        # )
        #
        # fig.savefig(save_dir + filename)

df = pd.concat(run_data_frames, ignore_index=True)

colors = ['blue', 'orange', 'lime']

for comp_count in range(4, component_count+1):
    color = colors[comp_count-4]
    for seed in range(1, 3):

        df[(df.comp == comp_count) & (df.seed == seed)].plot(
            x='iter',
            y='likelihood',
            c=color
        )

exit()

valid_components = ds.get_valid_components()

print "Recommended component count: ", len(valid_components)

for i in range(component_count):
    if i in valid_components:
        ds.plot_iteration_traces(i)

for i in range(component_count):
    if i not in valid_components:
        print "Possible invalid Component"
        ds.plot_iteration_traces(i)
