# dt-linear

main.py parses a game winning strategy or an mdp AS-winning strategy and creates a couple of decision trees, each fully representing the strategy.

First two arguments to main.py are required, after them additional optional flags can be provided in arbitrary order.

* 1st argument: folder name F that contains "games" OR "mdps"
* 2nd argument: full name (including extension) D of the dataset

Dataset "F/D" will then be parsed and represented.

Optional flags:
* onlyparse - Only parse the dataset, do not create any trees
* do_sklearn - Also create a DT with scikit_learn
* do_auc_reg - Also create a DT with slightly worse but faster DT-LC-AUC
* dot - Save DTs in .dot format in "results/dot"
* dotpng - Save DTs in .dot and .png format in "results/dot" and "results/png"

<br><br>

experiments.py runs main.py on all datasets in a given folder, collects the statistics and saves them in "results/reports".

First two arguments to experiments.py are required, third optional.
* 1st argument: folder name that contains "games" OR "mdps"
* 2nd argument: "debug" OR "release" (whether to check code assertions)
* 3rd optional argument: "replace" (whether to replace old report file)
