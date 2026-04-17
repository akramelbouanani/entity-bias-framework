# A Scalable Entity-Based Framework for Auditing Bias in LLMs



This repository contains the data, execution scripts, and analysis pipelines for the paper "A Scalable Entity-Based Framework for Auditing Bias in LLMs". (Accepted to **ACL 2026, Findings.**)

## Overview

Existing approaches to bias evaluation in large language models trade ecological validity for statistical control. This repository implements a scalable framework using named entities as probes to measure structural disparities in model behavior. By systematically substituting entities within consistent synthetic task templates, the methodology isolates the effect of entity identity on model predictions. 

The infrastructure provided here supports the generation and analysis of the 1.9 billion data points evaluated in the study, spanning three entity types (politicians, countries, companies), twelve tasks, three languages, and sixteen models.

## Repository Structure

The codebase is organized into five primary directories to separate data, experiment execution, processing, and analysis.

* **`data/`** Contains the datasets for both the validation phase and the large-scale generalization audit.
    * `validation/{task}/`: Contains `entities.csv`, `generated_sentences.csv`, and `prepared_sentences.csv`. This directory stores the real-world benchmark sentences alongside their synthetic counterparts to assess synthetic-to-real alignment.
    * `generalization/{entity_type}/`: Contains `{task}.csv` and `entities.csv`. This directory stores the generated evidence-light templates, keywords, labels, and the hierarchical entity lists (politicians, companies, countries) used for the main audit.

* **`src/`** Contains the core Python scripts for defining and executing the experiments.
    * `utils_{entity_type}.py`: Defines the specific downstream tasks, including system prompts, admissible label sets, and label-to-token mapping dictionaries.
    * `run_generalization.py`: Executes the large-scale bias auditing experiments across the specified models and configurations.
    * `run_validation.py`: Executes the validation experiments comparing model performance on natural datasets versus the synthetic proxy.
    * `python_count.py`: Performs sanity checks on the integrity and completeness of the output files.

* **`slurm/`** Contains job submission scripts configured for running the large-scale inference experiments on a high-performance computing cluster.

* **`processing/`** Contains scripts that ingest the raw output files from the validation and generalization runs and condense the results into unified files for downstream evaluation.

* **`analysis/`** Contains scripts and Jupyter notebooks for computing bias metrics (such as normalized bias scores and statistical significance tests) and generating visualizations.

## Introducing a New Task

The framework is modular. To audit a new task or domain, follow these steps:

1.  **Define Entities:** Gather the target entities and store them in `data/generalization/{entity_type}/entities.csv`.
2.  **Configure Task Parameters:** Define the task specifications, including the prompt structure, label sets, and label weights, within `src/utils_{entity_type}.py`.
3.  **Run Experiments:** Execute the pipeline using `src/run_generalization.py` locally or via the provided scripts in the `slurm/` directory for cluster execution.
4.  **Process Outputs:** Consolidate the raw inference outputs using the scripts located in the `processing/` directory.
5.  **Analyze Results:** Compute the structural bias scores and evaluate statistical significance using the tools provided in the `analysis/` directory.

## Citation

If you find this code or data useful for your research, please cite the following paper:

```bibtex
@article{elbouanani2026scalable,
  title={A Scalable Entity-Based Framework for Auditing Bias in LLMs},
  author={Elbouanani, Akram and Tuo, Aboubacar and Popescu, Adrian},
  journal={arXiv preprint arXiv:2601.12374},
  year={2026}
}
```