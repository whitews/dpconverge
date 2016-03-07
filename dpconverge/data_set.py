import numpy as np
import pandas as pd
from matplotlib import pyplot
from itertools import cycle
from flowstats import cluster

colors = [
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
        self._raw_results = None  # holds DPMixture object
        self.results = None

    @property
    def classifications(self):
        return self.blobs.keys()

    @property
    def parameter_count(self):
        return self._parameter_count

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

    def plot_blobs(
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

        color_cycle = cycle(colors)

        for c in classifications:
            pyplot.scatter(
                self.blobs[c][:, x],
                self.blobs[c][:, y],
                s=5,
                c=color_cycle.next(),
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
        if self.results is not None:
            raise ValueError("Data set already has clustering results")

        model = cluster.DPMixtureModel(
            component_count,
            iteration_count,
            burn_in,
            model='dp'
        )

        self._raw_results = model.fit(
            np.vstack(self.blobs.values()),
            True,
            seed=random_seed,
            munkres_id=True,
            verbose=True
        )

        self.results = self._create_results_dataframe(
            self._raw_results,
            component_count,
            iteration_count
        )

    def plot_iteration_traces(self, component):
        fig = pyplot.figure(figsize=(16, 4 * self._parameter_count))
        subplot_n = 1

        ds_comp = self.results[self.results.component == component]

        for param in ['loc'+str(i) for i in range(self._parameter_count)]:
            ax = fig.add_subplot(self._parameter_count, 1, subplot_n)

            ax.set_title(
                'Component: %d, Param: %s' %
                (component, param)
            )

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
            ax2.set_ylim(0.0, 1.0)
            ax2.plot(
                ds_comp.iteration,
                ds_comp.weight,
                'sienna',
                lw='0.5',
                alpha=0.5
            )
            ax2.fill_between(
                ds_comp.iteration,
                ds_comp.weight,
                where=ds_comp.iteration >= 0,
                interpolate=True,
                color='salmon',
                lw='1',
                alpha=0.5)

            subplot_n += 1

        pyplot.show()

    def plot_classifications(self, iteration, x=0, y=1, x_lim=None, y_lim=None):
        dp_mixture_iter = self._raw_results.get_iteration(iteration)

        raw_data = np.vstack(self.blobs.values())
        classifications = dp_mixture_iter.classify(raw_data)

        pyplot.figure(figsize=(8, 8))

        cmap = pyplot.cm.get_cmap('gist_rainbow')
        cmap_list = [cmap(i) for i in range(cmap.N)]
        cmap.from_list(np.linspace(0, 1, len(dp_mixture_iter)))
        cs = [cmap[i] for i in classifications]

        pyplot.scatter(
            raw_data[:, x],
            raw_data[:, y],
            s=8,
            c=cs,
            edgecolors='none',
            alpha=1.0
        )

        if x_lim is not None:
            pyplot.xlim(xmin=x_lim[0])
            pyplot.xlim(xmax=x_lim[1])
        if y_lim is not None:
            pyplot.ylim(ymin=y_lim[0])
            pyplot.ylim(ymax=y_lim[1])

        for i, dp_cluster in enumerate(dp_mixture_iter):
            pyplot.text(
                dp_cluster.mu[x],
                dp_cluster.mu[y],
                str(i),
                va='center',
                ha='center',
                color='lime',
                size=14,
                bbox=dict(facecolor='black')
            )
        pyplot.title('Fitted clusters')

        pyplot.show()

    def plot_animated_trace(self, x=0, y=1, x_lim=None, y_lim=None):
        dp_mixture_iter = self._raw_results.get_iteration(iteration)

        raw_data = np.vstack(self.blobs.values())
        classifications = dp_mixture_iter.classify(raw_data)

        pyplot.figure(figsize=(8, 8))

        _colors = pyplot.cm.rainbow(np.linspace(0, 1, len(dp_mixture_iter)))
        cs = [_colors[i] for i in classifications]

        pyplot.scatter(
            raw_data[:, x],
            raw_data[:, y],
            s=8,
            c=cs,
            edgecolors='none',
            alpha=1.0
        )

        if x_lim is not None:
            pyplot.xlim(xmin=x_lim[0])
            pyplot.xlim(xmax=x_lim[1])
        if y_lim is not None:
            pyplot.ylim(ymin=y_lim[0])
            pyplot.ylim(ymax=y_lim[1])

        for i, dp_cluster in enumerate(dp_mixture_iter):
            pyplot.text(
                dp_cluster.mu[x],
                dp_cluster.mu[y],
                str(i),
                va='center',
                ha='center',
                color='lime',
                size=14,
                bbox=dict(facecolor='black')
            )
        pyplot.title('Fitted clusters')

        pyplot.show()