import numpy as np
import pandas as pd
from matplotlib import pyplot
from itertools import cycle
from flowstats import cluster


class DataSet(object):
    """
    A single multi-variate data set
    """
    def __init__(self, parameter_count):
        if not isinstance(parameter_count, int):
            raise TypeError("'parameter_count' must be an integer")

        self._parameter_count = parameter_count
        self._data = np.empty([0, parameter_count])
        self.blobs = {}
        self.length = None
        self.results = None

    @property
    def classifications(self):
        return self.blobs.keys()

    def add_blob(self, classification, blob_data):
        if not isinstance(classification, int):
            raise TypeError("'classification' must be an integer")

        if not isinstance(blob_data, np.ndarray):
            raise TypeError("'blob_data' must be a NumPy 'ndarray'")

        if blob_data.shape[1] != self._parameter_count:
            raise ValueError("blob does not match data set's parameter count")

        if classification in self.blobs.keys():
            raise ValueError("This classification is already in use")
        else:
            self.blobs[classification] = blob_data

    def plot(
            self,
            classifications,
            x=0,
            y=1,
            figure_size=(8, 8),
            x_lim=None,
            y_lim=None
    ):
        pyplot.figure(figsize=figure_size)

        if x_lim is not None:
            pyplot.xlim(xmin=x_lim[0], xmax=x_lim[1])
        if y_lim is not None:
            pyplot.ylim(ymin=y_lim[0], ymax=y_lim[1])

        colors = cycle(
            [
                'dodgerblue',
                'gold',
                'red',
                'green',
                'orange',
                'purple',
                'darkolivegreen',
                'darkblue',
                'goldenrod',
                'indianred',
                'seagreen',
                'peru'
            ]
        )

        for c in classifications:
            pyplot.scatter(
                self.blobs[c][:, x],
                self.blobs[c][:, y],
                s=5,
                c=colors.next(),
                edgecolors='none'
            )

        pyplot.show()

    def _create_results_dataframe(self, results, n_components, n_iterations):
        component_dicts = []

        for iter_i in range(0, n_components * n_iterations, n_components):
            iter_components = results[iter_i:iter_i + n_components]
            n_iter = iter_i / n_components

            for component_i, _component in enumerate(iter_components):
                component_dict = {
                    'iteration': n_iter,
                    'component': component_i,
                    'weight': _component.pi,
                }

                for loc_i in range(0, self._parameter_count):
                    component_dict['loc' + str(loc_i)] = _component.mu[loc_i]

                component_dicts.append(component_dict)

        return pd.DataFrame(component_dicts)

    def cluster(self, component_count, burn_in, iteration_count, random_seed):
        model = cluster.DPMixtureModel(
            component_count,
            iteration_count,
            burn_in,
            model='dp'
        )

        fitted_results = model.fit(
            np.vstack(self.blobs.values()),
            True,
            seed=random_seed,
            munkres_id=True,
            verbose=True
        )

        self.results = self._create_results_dataframe(
            fitted_results,
            component_count,
            iteration_count
        )

    def plot_iteration_traces(self, component):
        fig_n = 0

        ds_comp = self.results[self.results.component == component]

        for param in ['loc'+str(i) for i in range(self._parameter_count)]:
            fig = pyplot.figure(fig_n, figsize=(16, 4))
            pyplot.title(
                'Component: %d, Param: %s' %
                (component, param)
            )

            ax = fig.add_subplot(111)
            ax.set_xlim(0, len(ds_comp.iteration))
            ax.set_ylim(ds_comp[param].min()/1.5, ds_comp[param].max())
            ax.plot(
                ds_comp.iteration,
                ds_comp[param],
                'dodgerblue',
                lw='0.5',
                alpha=0.8
            )

            ax2 = ax.twinx()
            ax2.set_xlim(0, len(ds_comp.iteration))
            ax2.set_ylim(ds_comp.weight.min(), ds_comp.weight.max() * 1.5)
            #ax2.plot(ds_comp.iteration, ds_comp.weight, 'sienna', lw='0.5', alpha=0.09)
            ax2.fill_between(
                ds_comp.iteration,
                ds_comp.weight,
                where=ds_comp.iteration > 0,
                interpolate=True,
                color='salmon',
                lw='1',
                alpha=0.5)

            fig_n += 1

        pyplot.show()
