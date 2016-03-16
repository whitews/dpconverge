import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
from scipy.stats import skew, kurtosis
from matplotlib import pyplot, animation
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

        # noinspection PyUnresolvedReferences
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

        for c in classifications:
            c_array = np.empty(self.blobs[c].shape[0])
            c_array.fill(c)

            pyplot.scatter(
                self.blobs[c][:, x],
                self.blobs[c][:, y],
                s=4,
                c=c_array,
                cmap=pyplot.cm.get_cmap('jet'),
                vmin=min(self.blobs.keys()),
                vmax=max(self.blobs.keys()),
                edgecolors='none',
                alpha=0.7
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

    def add_results(self, dp_mixture):
        if not isinstance(dp_mixture, cluster.DPMixture):
            raise TypeError("Data set results must be a 'DPMixture'")
        elif self._raw_results is not None:
            raise ValueError("Data set already has clustering results")

        if len(dp_mixture) % dp_mixture.niter != 0:
            raise ValueError("Failed to parse DPMixture components")

        iteration_count = dp_mixture.niter
        component_count = len(dp_mixture) / iteration_count

        self._raw_results = dp_mixture

        self.results = self._create_results_dataframe(
            self._raw_results,
            component_count,
            iteration_count
        )

    def test_component(self, component_dataframe):
        """
        Tests a given component dataframe for convergence, returning
        True for converged components
        :param component_dataframe: Pandas dataframe
        :return: boolean
        """

        # define our acceptable bounds
        skew_range = [-0.6, 0.6]
        kurt_range = [-1.5, 0.5]  # accept shorter tails for bang-on data
        weight_low = 0.008

        # perform weight test first
        if component_dataframe.weight.mean() < weight_low:
            return False

        if skew(component_dataframe.weight) < skew_range[0]:
            return False

        if skew(component_dataframe.weight) > skew_range[1]:
            return False

        if kurtosis(component_dataframe.weight) < kurt_range[0]:
            return False

        if kurtosis(component_dataframe.weight) > kurt_range[1]:
            return False

        # component_dataframe.weight.std()

        # now for the component parameter locations
        for param in ['loc'+str(i) for i in range(self._parameter_count)]:
            if skew(component_dataframe[param]) < skew_range[0]:
                return False

            if skew(component_dataframe[param]) > skew_range[1]:
                return False

            if kurtosis(component_dataframe[param]) < kurt_range[0]:
                return False

            if kurtosis(component_dataframe[param]) > kurt_range[1]:
                return False

            # component_dataframe[param].std()

        # all tests passed
        return True

    def get_valid_components(self):
        if self._raw_results is None:
            raise ValueError("Data set has no saved results")

        # list of good components to return
        good_comps = []

        for comp in self.results.component.unique():
            comp_data = self.results[self.results.component == comp]
            comp_passed = self.test_component(comp_data)
            if comp_passed:
                good_comps.append(comp)

        return good_comps

    @staticmethod
    def _get_likelihood(dp_mixture, data):
        likelihood = np.sum(
            [
                sum(pi * multivariate_normal.pdf(data, mu, sigma))
                for (pi, mu, sigma) in zip(
                    dp_mixture.pis,
                    dp_mixture.mus,
                    dp_mixture.sigmas
                )
            ]
        )

        return likelihood

    @staticmethod
    def _get_scipy_log_likelihood(dp_mixture, data):
        log_likelihood = np.sum(
            np.log(
                sum(
                    pi * multivariate_normal.pdf(data, mu, sigma)
                    for (pi, mu, sigma) in zip(
                        dp_mixture.pis,
                        dp_mixture.mus,
                        dp_mixture.sigmas
                    )
                )
            )
        )

        return log_likelihood

    def get_log_likelihood_trace(self, use_scipy=False):
        if self._raw_results is None:
            raise ValueError("Data set has no saved results")

        log_likelihoods = []
        data = np.vstack(self.blobs.values())

        for i in range(self._raw_results.niter):
            dp_mixture_iter = self._raw_results.get_iteration(i)

            if use_scipy:
                log_likelihoods.append(
                    self._get_scipy_log_likelihood(
                        dp_mixture_iter,
                        data
                    )
                )
            else:
                log_likelihoods.append(
                    dp_mixture_iter.log_likelihood(data)
                )
        return log_likelihoods

    def get_likelihood_trace(self):
        if self._raw_results is None:
            raise ValueError("Data set has no saved results")

        likelihoods = []
        data = np.vstack(self.blobs.values())

        for i in range(self._raw_results.niter):
            dp_mixture_iter = self._raw_results.get_iteration(i)

            likelihoods.append(
                self._get_likelihood(
                    dp_mixture_iter,
                    data
                )
            )

        return likelihoods

    def plot_log_likelihood_trace(self, use_scipy=False):
        log_likelihoods = self.get_log_likelihood_trace(use_scipy=use_scipy)
        n_iterations = self._raw_results.niter

        fig = pyplot.figure(figsize=(16, 4))

        ax = fig.add_subplot(1, 1, 1)

        ax.set_title('Log likelihood trace')

        ax.plot(
            range(n_iterations),
            log_likelihoods,
            'dodgerblue',
            lw='1.0',
            alpha=0.8
        )

        return fig

    def plot_likelihood_trace(self):
        likelihoods = self.get_likelihood_trace()
        n_iterations = self._raw_results.niter

        fig = pyplot.figure(figsize=(16, 4))

        ax = fig.add_subplot(1, 1, 1)

        ax.set_title('Log likelihood trace')

        ax.plot(
            range(n_iterations),
            likelihoods,
            'dodgerblue',
            lw='1.0',
            alpha=0.8
        )

        return fig

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

        cmap = pyplot.cm.get_cmap('jet')

        pyplot.scatter(
            raw_data[:, x],
            raw_data[:, y],
            s=8,
            c=classifications,
            edgecolors='none',
            cmap=cmap,
            vmax=len(dp_mixture_iter) - 1,
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
        pyplot.title('Iteration %d' % iteration)

        pyplot.show()

    def plot_animated_trace(
            self,
            x=0,
            y=1,
            x_lim=None,
            y_lim=None,
            iter_start=0
    ):
        def update_plot(frame):
            pyplot.title('Iteration: %d' % frame)
            scatter.set_array(classifications[frame - iter_start])

        n_iterations = self._raw_results.niter
        n_clusters = len(self._raw_results.get_iteration(0))
        raw_data = np.vstack(self.blobs.values())

        classifications = []
        for i in range(iter_start, n_iterations):
            new_iter = self._raw_results.get_iteration(i)
            classifications.append(new_iter.classify(raw_data))

        fig = pyplot.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, axisbg='gray')

        scatter = ax.scatter(
            raw_data[:, x],
            raw_data[:, y],
            s=16,
            c=classifications[0],  # start with 1st iteration
            edgecolors='none',
            cmap=pyplot.cm.get_cmap('jet'),
            vmax=n_clusters - 1,
            alpha=1.0
        )

        if x_lim is not None:
            ax.xlim(xmin=x_lim[0])
            ax.xlim(xmax=x_lim[1])
        if y_lim is not None:
            ax.ylim(ymin=y_lim[0])
            ax.ylim(ymax=y_lim[1])

        anim = animation.FuncAnimation(
            fig,
            update_plot,
            interval=150,
            frames=xrange(iter_start, n_iterations),
            fargs=()
        )

        pyplot.title('Fitted clusters')

        pyplot.show()

        return anim
