from data_set import DataSet
from flowstats import cluster
import numpy as np
import pandas as pd


class DataCollection(object):
    """
    A collection of DataSet objects
    """
    def __init__(self):
        self._parameter_count = None
        self.data_sets = []

    @property
    def data_set_count(self):
        return len(self.data_sets)

    def add_data_set(self, data_set):
        if not isinstance(data_set, DataSet):
            raise TypeError("data_set must be of type DataSet")

        if self._parameter_count is None:
            self._parameter_count = data_set.parameter_count

        if self._parameter_count != data_set.parameter_count:
            raise ValueError(
                "Data set parameter count must match the existing data sets"
            )
        else:
            self.data_sets.append(data_set)

    def estimate_initial_conditions(self, max_comp=128, max_iter=5000):
        # create a data set that's the combination of all data sets
        prelim_ds = DataSet(parameter_count=self._parameter_count)

        for i, ds in enumerate(self.data_sets):
            # start blob labels at 1 (i + 1)
            prelim_ds.add_blob(i + 1, np.vstack(ds.blobs.values()))

        prelim_ds.plot_blobs(prelim_ds.labels, x_lim=[0, 4], y_lim=[0, 4])

        # now run bem on the combined data set to get initial conditions
        max_log_like = None  # the highest value for all runs
        converged = False
        component_count = max_comp
        iteration_count = max_iter

        results = []  # will be a list of dicts to convert to a DataFrame

        while not converged:
            print component_count

            new_comp_counts = []

            for seed in range(1, 13):
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
                    tmp_comp_count = np.sum(prelim_ds.raw_results.pis > 0.0001)
                    new_comp_counts.append(tmp_comp_count)

                    # save good run to our results
                    results.append(
                        {
                            'comp': component_count,
                            'true_comp': tmp_comp_count,
                            'seed': seed,
                            'log_like': log_like,
                            'pis': prelim_ds.raw_results.pis,
                            'mus': prelim_ds.raw_results.mus,
                            'sigmas': prelim_ds.raw_results.sigmas
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

        min_comp = results_df.comp.min()
        best_index = results_df[results_df.comp == min_comp].log_like.argmax()

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

        # get classifications to calculate weights for each data set
        pis = []
        for label in sorted(prelim_ds.labels):
            label_classes = prelim_ds.get_classifications(0, [label])

            ds_pis = []

            for c in range(best_run['comp']):
                ds_pis.append(np.sum(label_classes == c) / float(len(label_classes)))

            pis.append(ds_pis)  # list of lists

        # convert LoL pis to numpy array
        pis = np.array(pis)

        prelim_ds.plot_classifications(0)

        # Re-run a chain using the initial conditions from the last iteration
        last_iter = prelim_ds.raw_results.get_iteration(0)

        initial_conditions = {
            'pis': pis,
            'mus': last_iter.mus,
            'sigmas': last_iter.sigmas
        }

        return best_run['comp'], initial_conditions

    def cluster(
            self,
            component_count,
            burn_in,
            iteration_count,
            random_seed,
            initial_conditions=None
    ):
        # local 'data_sets' holds the raw data values for each DataSet
        data_sets = list()

        for ds in self.data_sets:
            data = np.vstack(ds.blobs.values())
            if data.size == 0:
                raise ValueError("Found an empty data set")
            data_sets.append(data)

        if len(data_sets) < 2:
            # nothing for us to do
            raise ValueError("HDP needs at least 2 data sets")

        model = cluster.HDPMixtureModel(
            component_count,
            iteration_count,
            burn_in
        )

        if initial_conditions is not None:
            # should check keys of initial values, the
            # shapes & values should be taken care of in FlowStats
            initial_weights = initial_conditions['pis']
            model.load_mu(initial_conditions['mus'])
            model.load_sigma(initial_conditions['sigmas'])
        else:
            initial_weights = None

        fitted_results = model.fit(
            data_sets,
            True,
            seed=random_seed,
            munkres_id=False,
            verbose=True,
            initial_weights=initial_weights
        )

        # save results for each DataSet
        for i, ds in enumerate(self.data_sets):
            ds.add_results(fitted_results[i])
