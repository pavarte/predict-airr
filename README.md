## Adaptive Immune Profiling Challenge 2025 - Code Template Repository

This repository provides a code template to encourage a unified way of running the models of different participants/teams from the AIRR-ML-25 community challenge: https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025

As described in the [official competition page](https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025), to win the prize money, a prerequisite is that the code has to be made open-source. In addition, the top 10 submissions/teams will be invited to become co-authors in a scientific paper that involves further stress-testing of their models in a subsequent phase with many other datasets outside Kaggle platform. To enable such further analyses and re-use of the models by the community, **we strongly encourage** the participants to adhere to a [code template](https://github.com/uio-bmi/predict-airr) that we provide through this repository that enables a uniform interface of running models: https://github.com/uio-bmi/predict-airr

Ideally, all the methods can be run in a unified way, e.g.,

`python3 -m submission.main --train_dir /path/to/train_dir --test_dir /path/to/test_dir --out_dir /path/to/output_dir --n_jobs 4 --device cpu`

This requires that participants/teams adhere to the code template in `ImmuneStatePredictor` class provided in `predictor.py` by filling in their implementations within the placeholders and replacing any example code lines with actual code that makes sense. 

It will also be important for the participants/teams to provide the exact requirements/dependencies to be able to containerize and run their code. If the participants/teams fork the provided repository and make their changes, it has to be remembered to also replace the dependencies in `requirements.txt` with their dependencies and exact versions.

Those participants that make use of Kaggle resources and Kaggle notebooks to make submissions are also strongly encouraged to copy the code template, particularly the `ImmuneStatePredictor` class and any utility functions from the provided code template repository and adhere to the code template to enable unified way of running different methods.