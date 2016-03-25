from dpconverge.data_set import DataSet
import numpy as np
import pandas as pd
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
    random_state=3
)

ds = DataSet(parameter_count=2)

ds.add_blob(1, blob1)
ds.add_blob(2, blob2)
ds.add_blob(3, blob3)
ds.add_blob(4, blob4)
ds.add_blob(5, blob5)

# ds.plot_blobs(ds.classifications, x_lim=[0, 4], y_lim=[0, 4])

component_count = 128
iteration_count = 5000

# use multiple runs of BEM to estimate the number of components
# and get initial conditions

max_log_like = None  # the highest value for all runs
converged = False

results = []  # will be a list of dicts to convert to a DataFrame

while not converged:
    print component_count

    new_comp_counts = []

    for seed in range(1, 17):
        ds.results = None  # reset results

        ds.cluster(
            component_count=component_count,
            burn_in=0,
            iteration_count=iteration_count,
            random_seed=seed,
            model='bem'
        )

        log_like = ds.get_log_likelihood_trace()[0]
        print log_like

        if log_like > max_log_like:
            max_log_like = log_like

        # if the new log_like is close to the max (within 1%),
        # see if there are any empty components (pi < 0.0001)
        if abs(max_log_like - log_like) < abs(max_log_like * 0.01):
            tmp_comp_count = np.sum(ds.raw_results.pis > 0.0001)
            new_comp_counts.append(tmp_comp_count)

            # save good run to our results
            results.append(
                {
                    'comp': component_count,
                    'true_comp': tmp_comp_count,
                    'seed': seed,
                    'log_like': log_like,
                    'pis': ds.raw_results.pis,
                    'mus': ds.raw_results.mus,
                    'sigmas': ds.raw_results.sigmas
                }
            )

            # ds.plot_classifications(0)

    if len(new_comp_counts) > 0:
        if int(np.mean(new_comp_counts)) < component_count:
            component_count = int(np.mean(new_comp_counts))
        else:
            converged = True
    else:
        converged = True

results_df = pd.DataFrame(
    results,
    columns=['comp', 'true_comp', 'seed', 'log_like']
)

min_comp_count = results_df.comp.min()
best_index = results_df[results_df.comp == min_comp_count].log_like.argmax()

best_run = results[best_index]

ds.results = None

ds.cluster(
    component_count=best_run['comp'],
    burn_in=0,
    iteration_count=iteration_count,
    random_seed=best_run['seed'],
    model='bem'
)

log_like = ds.get_log_likelihood_trace()[0]
print log_like

ds.plot_classifications(0)

# Re-run a chain using the initial conditions from the last iteration
last_iter = ds.raw_results.get_iteration(0)

initial_conditions = {
    'pis': last_iter.pis.flatten(),
    'mus': last_iter.mus,
    'sigmas': last_iter.sigmas
}

# reset DataSet results
ds.results = None

ds.cluster(
    component_count=best_run['comp'],
    burn_in=0,
    iteration_count=iteration_count,
    random_seed=1,
    initial_conditions=initial_conditions
)

ds.plot_log_likelihood_trace()
pyplot.show()

valid_components = ds.get_valid_components()

for i in range(best_run['comp']):
    ds.plot_iteration_traces(i)

ds.plot_animated_trace()

pass
