from dpconverge.data_set import DataSet
from dpconverge.data_collection import DataCollection
import numpy as np
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs


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
    centers=centers[3],
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

ds2 = DataSet(parameter_count=2)

ds2.add_blob(1, ds2_blob1)
ds2.add_blob(2, ds2_blob2)
ds2.add_blob(3, ds2_blob3)
ds2.add_blob(4, ds2_blob4)
ds2.add_blob(5, ds2_blob5)

ds2.plot_blobs(ds2.labels, x_lim=[0, 4], y_lim=[0, 4])

# now create a 3rd data set that's the combination of both
prelim_ds = DataSet(parameter_count=2)
prelim_ds.add_blob(1, np.vstack(ds.blobs.values()))
prelim_ds.add_blob(2, np.vstack(ds2.blobs.values()))

prelim_ds.plot_blobs(prelim_ds.labels, x_lim=[0, 4], y_lim=[0, 4])

# now run bem on 1st data set to get initial conditions
max_log_like = None  # the highest value for all runs
converged = False
component_count = 6
iteration_count = 1000

results = []  # will be a list of dicts to convert to a DataFrame

while not converged:
    print component_count

    new_comp_counts = []

    for seed in range(1, 9):
        prelim_ds.results = None  # reset results

        prelim_ds.cluster(
            component_count=component_count,
            burn_in=0,
            iteration_count=iteration_count,
            random_seed=seed,
            model='bem'
        )

        log_like = prelim_ds.get_log_likelihood_trace()[0]
        print log_like

        if log_like > max_log_like:
            max_log_like = log_like

        # if the new log_like is close to the max (within 1%),
        # see if there are any empty components (pi < 0.0001)
        if abs(max_log_like - log_like) < abs(max_log_like * 0.01):
            tmp_comp_count = np.sum(prelim_ds._raw_results.pis > 0.0001)
            new_comp_counts.append(tmp_comp_count)

            # save good run to our results
            results.append(
                {
                    'comp': component_count,
                    'true_comp': tmp_comp_count,
                    'seed': seed,
                    'log_like': log_like,
                    'pis': prelim_ds._raw_results.pis,
                    'mus': prelim_ds._raw_results.mus,
                    'sigmas': prelim_ds._raw_results.sigmas
                }
            )

            # prelim_ds.plot_classifications(0)

    if len(new_comp_counts) > 0:
        if int(np.mean(new_comp_counts)) < component_count:
            component_count = int(np.min(new_comp_counts))
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

prelim_ds.results = None

prelim_ds.cluster(
    component_count=best_run['comp'],
    burn_in=0,
    iteration_count=iteration_count,
    random_seed=best_run['seed'],
    model='bem'
)

log_like = prelim_ds.get_log_likelihood_trace()[0]
print log_like

# each label in prelim_ds is a data set, get classifications to calculate
# weights for each data set...we'll use these as initial weights in HDP
pis = []
for label in sorted(prelim_ds.labels):
    label_classes = prelim_ds.get_classifications(0, [label])

    ds_pis = []

    for c in sorted(np.unique(label_classes)):
        ds_pis.append(np.sum(label_classes == c) / float(len(label_classes)))

    pis.append(ds_pis)  # list of lists

# convert LoL pis to numpy array
pis = np.array(pis)

prelim_ds.plot_classifications(0)

# Re-run a chain using the initial conditions from the last iteration
last_iter = prelim_ds._raw_results.get_iteration(0)

initial_conditions = {
    'pis': pis,
    'mus': last_iter.mus,
    'sigmas': last_iter.sigmas
}

# create our data collection to run HDP
dc = DataCollection()
dc.add_data_set(ds)
dc.add_data_set(ds2)

# # best initial conditions determined previously
# initial_conditions = {
#     'pis': np.array(
#         [
#             [
#                 0.09705467,
#                 0.02443813,
#                 0.02515411,
#                 0.28562875,
#                 0.56772434
#             ],
#             [
#                 0.09705467,
#                 0.02443813,
#                 0.02515411,
#                 0.28562875,
#                 0.56772434
#             ],
#         ]
#     ),
#     'mus': np.array(
#         [
#             [2.00527002, 1.35488267],
#             [2.49589273, 2.00544444],
#             [2.49914699, 1.50246998],
#             [2.00215572, 2.99699475],
#             [1.99747818, 2.00077871]
#         ]
#     ),
#     'sigmas': np.array(
#         [
#             [
#                 [0.00999859, -0.00043552],
#                 [-0.00043552, 0.02284242]
#             ],
#             [
#                 [0.0108694, -0.0001163],
#                 [-0.0001163, 0.01321305]
#             ],
#             [
#                 [0.01093973, -0.00076813],
#                 [-0.00076813, 0.01154952]
#             ],
#             [
#                 [0.03909857, -0.00024484],
#                 [-0.00024484, 0.01006793]
#             ],
#             [
#                 [0.03957697, 0.00043603],
#                 [0.00043603, 0.08887145]
#             ]
#         ]
#     )
# }

dc.cluster(
    component_count=min_comp_count,
    burn_in=0,
    iteration_count=500,
    random_seed=1,
    initial_conditions=initial_conditions
)

for r_ds in dc.data_sets:
    valid_components = r_ds.get_valid_components()

    print "Recommended component count: ", len(valid_components)

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
