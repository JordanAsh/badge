### Result analysis script for plotting:
- learning curves,
- comparison matrices,
- normalized error CDFs.

### Example

Suppose we have the following directory structure that stores all experimental results (in all output.txt's below):

```
~/deep_active/philly_data/
|--- pull_list.txt
|--- result_cluster_1
|    |--- expt_result_6_albl_100_mlp
|    |    |--- output.txt
|    |--- expt_result_6_albl_1000_mlp
|    |    |--- output.txt
|    ...
|--- result_cluster_2
|    |--- expt_result_6_albl_100_mlp
|    |    |--- output.txt
|    |--- expt_result_6_albl_1000_mlp
|    |    |--- output.txt
|    ...
```


In the above structure, each first level subdirectory represents a separate run of the full collection of settings; each second-level directory represents a separate configuration of (dataset, algorithm, batch size, network architecture); its corresponding output.txt stores the output of an execution run.py on that specific configuration.

In addition, `pull_list.txt` is a text file that has two lines, indicating which two subdirectories to parse:
```
result_cluster_1
result_cluster_2
```

Given the above setup, we run the following command to generate all plots:
```
python agg_results.py ~/deep_active/philly_data/ pull_list.txt output_figs 1 0
```

Here:
- `~/deep_active/philly_data/` is the folder that stores all the results
- `pull_list.txt` is the names of subdirectories under `deep_active/philly_data/` that stores all
- `output_figs` is the name of the subdirectory that stores the output figures
- the second-to-last argument '1' means that we overwrite (therefore, do not use) the cache file for the output results; after first run of the command, cache files will be created in each subdirectories, therefore, in subsequent runs, we can set the argument to '0' to directly load parsed and cached data.
- the last argument '0' means that we do not turn on the interactive mode

See also explanations of command line arguments in `agg_results.py`.
