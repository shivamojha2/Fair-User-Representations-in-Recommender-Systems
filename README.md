# Fair-User-Representations-in-Recommender-Systems

This repository is on a project to try and create a fair user representation in terms of its embedding to ensure that the pre-existing demographic biases that are seen and prevalent in these systems can be mitigated while maintaining the recommendation scores. This idea of personalized user fairness in recommendation is introduced here in this project and the idea to create feature-independent embeddings using an adversarial approach has been explored. The idea is to then test this approach with various recommender models to see if this representation is robust.

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
