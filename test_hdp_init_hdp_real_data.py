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

component_count = 8
seeds = range(1, 9)
converged = False
results = []  # will be a list of dicts to convert to a DataFrame

while not converged:
    print component_count

    for seed in seeds:
        dc.reset_results()

        dc.cluster(
            component_count=component_count,
            burn_in=1000,
            iteration_count=1000,
            random_seed=seed
        )

        # get valid components, but ignore weight since one sample may have
        # a component the others do not. Since the mus and sigmas are the same
        # for all samples, we can just test one sample
        valid_components = dc.data_sets[0].get_valid_components(
            ignore_weight=True
        )

        print "Recommended component count: ", len(valid_components)

        for r_ds in dc.data_sets:
            for i in range(component_count):
                if i not in valid_components:
                    print "Possible invalid Component"
                    r_ds.plot_iteration_traces(i)

            r_ds.plot_log_likelihood_trace()
            # r_ds.plot_animated_trace()
