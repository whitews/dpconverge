from data_set import DataSet
from flowstats import cluster
import numpy as np


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
            model.load_pi(initial_conditions['pis'])
            model.load_mu(initial_conditions['mus'])
            model.load_sigma(initial_conditions['sigmas'])

        fitted_results = model.fit(
            data_sets,
            True,
            seed=random_seed,
            munkres_id=True,
            verbose=True
        )

        # save results for each DataSet
        for i, ds in enumerate(self.data_sets):
            ds.add_results(fitted_results[i])
