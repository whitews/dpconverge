from dpconverge.data_set import DataSet
from dpconverge.data_collection import DataCollection
import sys
import numpy as np
import flowio
import flowutils

fcs_files = [
    sys.argv[1],
    sys.argv[2]
]

spill_text = """4, Blue B-A, Blue A-A, Red C-A, Green E-A,
1, 6.751e-3, 0, 2.807e-3,
0, 1, 0.03, 0,
0, 5.559e-3, 1, 0,
2.519e-3, 0.115, 3.482e-3, 1
"""

spill, markers = flowutils.compensate.get_spill(spill_text)
fluoro_indices = [6, 7, 8, 9]

subsample_count = 10000
rng = np.random.RandomState()
rng.seed(123)

dc = DataCollection()

for fcs_file in fcs_files:

    fd = flowio.FlowData(fcs_file)
    events = np.reshape(fd.events, (-1, fd.channel_count))

    comp_events = flowutils.compensate.compensate(
        events,
        spill,
        fluoro_indices
    )

    xform_comp_events = flowutils.transforms.asinh(
        comp_events,
        fluoro_indices,
        pre_scale=0.003
    )

    shuffled_indices = np.arange(fd.event_count)
    rng.shuffle(shuffled_indices)

    # save indices
    subsample_indices = shuffled_indices[:subsample_count]

    # sub-sample FCS events using given indices
    subsample = xform_comp_events[subsample_indices][:, [0, 3, 6, 7, 8, 9]]

    # now create our DataSet
    ds = DataSet(parameter_count=6)

    ds.add_blob(1, subsample)

    # ds.plot_blobs(ds.labels)
    # ds.plot_blobs(ds.labels, x=0, y=2)
    # ds.plot_blobs(ds.labels, x=1, y=2)

    dc.add_data_set(ds)

iteration_count = 1000
component_count = 16
seeds = range(1, 9)
converged = False
results = []  # will be a list of dicts to convert to a DataFrame

while not converged:
    print "Comp count:", component_count

    last_results_count = len(results)

    for seed in seeds:
        print "Seed:", seed

        dc.reset_results()

        tmp_converged = False

        hdp_mixture = dc.cluster(
            component_count=component_count,
            burn_in=1000,
            iteration_count=iteration_count,
            random_seed=seed
        )

        # get valid components, but ignore weight since one sample may have
        # a component the others do not. Since the mus and sigmas are the same
        # for all samples, we can just test one sample
        valid_components = dc.data_sets[0].get_valid_components(
            ignore_weight=True
        )
        valid_comp_count = len(valid_components)

        print "Stable component count: ", valid_comp_count

        if valid_comp_count < component_count:
            progress = True
            while progress:
                # build initial conditions from last iteration
                pis = []
                for i, ds in enumerate(dc.data_sets):
                    last_iter = hdp_mixture[i].get_iteration(
                        iteration_count - 1
                    )
                    pis.append(last_iter.pis)

                initial_conditions = {
                    'pis': np.array(pis),
                    'mus': last_iter.mus,
                    'sigmas': last_iter.sigmas
                }

                # reset results
                dc.reset_results()

                # continue cluster for another round of iterations
                hdp_mixture = dc.cluster(
                    component_count=component_count,
                    burn_in=0,  # we want to start off where it left off
                    iteration_count=iteration_count,
                    random_seed=seed,
                    initial_conditions=initial_conditions
                )

                new_valid_comps = dc.data_sets[0].get_valid_components(
                    ignore_weight=True
                )

                print "Stable component count: ", len(new_valid_comps)

                if len(new_valid_comps) > valid_comp_count:
                    # check if it's equal to comp count
                    if len(new_valid_comps) == component_count:
                        # chain stabilized, we're good to save these results
                        tmp_converged = True
                        break
                    else:
                        valid_comp_count = len(new_valid_comps)
                else:
                    progress = False

        else:
            # valid comps equal comp count
            tmp_converged = True

        if tmp_converged:  # valid comps equal comp count
            # average results and get the sum of log likelihoods across
            # data sets
            hdp_mixture_avg = hdp_mixture.average()

            log_likelihoods = []

            for i, ds in enumerate(dc.data_sets):
                data = np.vstack(ds.blobs.values())

                dp_mixture = hdp_mixture_avg[i]

                log_likelihoods.append(
                    dp_mixture.log_likelihood(data)
                )

            # save results for this run
            results.append(
                {
                    'seed': seed,
                    'comp': component_count,
                    'log_like': sum(log_likelihoods),
                    'results': hdp_mixture_avg
                }
            )

            # we found a stable run at this component count, no need to
            # keep testing it, move on to the next higher comp count
            break

    if len(results) > last_results_count:
        # still generating stable chains, bump the component count and
        # keep going
        component_count += 1
    else:
        converged = True

# now, what to do with all these results?
print len(results)
