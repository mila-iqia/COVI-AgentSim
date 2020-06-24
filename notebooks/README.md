# Rules for pushing notebooks to the repo

Notebooks are not git friendly because they are colossol HTML files. Please observe the following rules while pushing any notebook to the repo
1. If you are the **author**, print it at the top of the notebook so that you can be contacted by anyone who wants to use it.
2. Edit this readme with one or two sentences to describe what can be achieved with that notebook.
3. Only push the notebook if it is easily usable by others so that anyone can do what is intended in the notebook just by replacing the folder or file path.
4. If possible, comment the code in the notebook to make it easier for someone to automate it.
5. Please follow good naming practices so that the intent of the notebook is clear from it's name.
6. **If you want to use a notebook** and are not the author of the notebook, please copy the notebook locally and use the clone instead. If you don't, it will show up in the diff, and you will have to either lose your progress by stashing it or commit the entire notebook. This commit will be a nightmare to resolve if there are any conflicts in it somewhere.
7. If you want to make changes to the existing notebook, contact the author of the notebook and work out a solution.
8. Please use `utils/` to write any python script that you want to use in your notebooks.

# Notebooks
1. `spy_humans_pro` - Visualize the timeline of phone message exchange between infector and infectee in an infection chain.
2. `intervention_impact` - Visualize cumulative cases per day and effective reproduction number.
3. `pareto-adoption-analysis` - Visualize the trade-off between mobility metrics and the severity of the outbreak as a result of different interventions.
4. `viral_load_plots` - Plots the viral load curves (individual or group average).
