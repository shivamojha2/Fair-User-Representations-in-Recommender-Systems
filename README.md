# Fair-User-Representations-in-Recommender-Systems

COMS 6998 Fair &amp; Robust Algorithms Final Project

## Code and system dependencies

The dependencies are mentioned in `requirements.txt`.

```pip install -r requirements.txt```

## Model architecture

![image](https://github.com/shivamojha2/Fair-User-Representations-in-Recommender-Systems/blob/main/static/images/architecture.png)

## Code execution

Use the following commands to execute the training after updating the configuration json file.

```
cd ./Fair-User-Representations-in-Recommender-Systems

python3 src/main.py -j "src/experiments/MLP_no_filter.json"
```

Logs and trained models will be periodically save in the directory `save_dir/<experiment_name>` from the configuration JSON file. This path and other useful parameters can be updated depending on the experiment in the `src/experiments`.
