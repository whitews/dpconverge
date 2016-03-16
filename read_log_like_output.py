import pandas as pd
import numpy as np
from matplotlib import pyplot

df = pd.read_pickle('log_like_data.pkl')

start_iter = 200

component_list = df.comp.unique()
seeds = df.seed.unique()

cmap = pyplot.cm.get_cmap('jet')
colors = [cmap(i) for i in np.linspace(0, 1, len(seeds))]
alphas = np.linspace(1, 0.4, len(component_list))

fig = pyplot.figure(figsize=(8, 8))

for i, comp_count in enumerate(component_list):
    for j, seed in enumerate(seeds):

        df[(df.comp == comp_count) & (df.seed == seed) & (df.iter >= start_iter)].plot(
            x='iter',
            y='likelihood',
            c=colors[j],
            linewidth=1.0,
            alpha=0.8
        )

pyplot.show()

pass
