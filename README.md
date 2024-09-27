<!-- I still have to figure out whether I like the title, badges, description and image to be centered or not.
Just change the alignment to "left" if you prefer -->
<div align="center">

# Consistency-based Sheaf Diffusion
<!-- Here you can put some badges, as the following: -->
[![Version][version-badge]][example-site]
[![Licence][license-badge]][license-site] <br>
[![Python][python-badge]][python-site]
[![PyTorch][pytorch-badge]][pytorch-site]
[![Lit][lit-badge]][lit-site]

</div>

## Getting started
<!-- This section summarizes the basic requirments and the installation process to properly run and reproduce the code -->

### Basic requirements
<!-- List of basic requirements needed to properly run the code -->
The code have been tested on a cluster of Linux nodes using [SLURM][slurm-site].<br>
We _cannot guarantee_ the functioning of the code if the following requirements are _not_ met:


### Installation
<!-- List the steps needed to properly install and run the code -->
> To properly install and run our code we recommend using a virtual environment (e.g., created via [`pyenv-virtualenv`][pyenv-virtualenv-site] or [`conda`][conda-site]).

The entire installation process consists of 3(_+1_) steps. You can skip step 0 at you own "risk".

#### (_Optional_) Step 0: create a virtual environment
In the following we show how to create the environment via [`pyenv`][pyenv-site] and [`pyenv-virtualenv`][pyenv-virtualenv-site].
The steps are the following:
- install [`pyenv`][pyenv-site] (if you don't have it yet) by following the [original guidelines][pyenv-install-site];
- install the correct Python version:
    ```sh
    pyenv install 3.10.4
    ```
- create a virtual environment with the correct version of Python:
    ```sh
    pyenv virtualenv 3.10.4 <PROJECT_NAME>
    ```

#### Step 1: clone the repository, change into it and (_optional_) activate the environment
This step allows you to download the code in your machine, move into the correct directory and (_optional_) activate the correct environment.
The steps are the following:
- clone the repository:
    ```sh
    git clone <PROJECT_URL>
    ```
- change into the repository:
    ```sh
    cd <PROJECT_NAME>
    ```
- (_optional_) activate the environment (everytime you'll enter the folder, the environment will be automatically activated)
    ```sh
    pyenv local <PROJECT_NAME> 
    ```

#### Step 2: install PyTorch and PyTorch Geometric
Depending on your CPU/GPU setup you might need to install [`PyTorch`][pytorch-site] and [`PyTorch Geometric`][pyg-site] in different ways.<br>
We provide a detailed explanation on how to install these packages [here][installation-guide-ref]. Please also consider to refer to the original documentation ([PyTorch][pytorch-install-site], [PyG][pyg-install-site]) in case of any doubts.

> Make sure the environment is active, in case you created it.

#### Step 3: install the code as a local package
All the required packages (beside Pytorch/Pytorch Geometric) are defined in the `pyproject.toml` file and can be easily installed via [`pip`][pip-site] as following:
```sh
pip install --editable .
``` 
or equivalently:
```sh
pip install -e .
```

## Running Experiments

1. Set up the sweeper's parameters in the `config/exp/<YOUR_EXP>.yaml` file according to the experiments you want to run.
2. Run `python exec.py --multirun +exp=YOUR_EXP hydra/launcher/qos=YOUR_QOS` to automatically run experiments on the SLURM cluster, or `python exec.py --multirun +exp=YOUR_EXP hydra/launcher=local` to run them locally.

## Reading Results

An example how to load and read results is given in `notebooks/VisualizeResults.ipynb`

## Support
If you have any questions regarding the code and its reproducibility, feel free to contact the [reference author](#authors).

## Authors
- **David Reifferscheidt**\*:
    > Master's student at the [Technical University of Munich (TUM)][tum-site]
    - Email: [david.reifferscheidt@tum.de][email-dr]
\* Reference author
- **Filippo Guerranti**\*:
    > PhD student at the [DAML][daml-site] group, [Technical University of Munich (TUM)][tum-site]
    - Email: [f.guerranti@tum.de][email-fg]
\* Reference author
### Citation
If you build upon this work, please cite our paper as follows: TBD
```
@inproceedings{CITATION_KEY,
    title = {{PAPER_TITLE}},
    author = {}
    booktitle = {{BOOK_TITLE}},
    year = {YEAR}
}
```


<!-- Variables -->
<!-- Badges -->
[version-badge]: https://img.shields.io/badge/version-1.0.1--alpha-brightgreen?style=flat
[license-badge]: https://img.shields.io/badge/licence-MIT-green
[python-badge]: https://img.shields.io/badge/python-3.10.4-3776AB?style=flat&logo=python&logoColor=white
[pytorch-badge]: https://img.shields.io/badge/pytorch-2.0.1-EE4C2C?style=flat&logo=PyTorch&logoColor=white
[pyg-badge]: https://img.shields.io/badge/pytorch--geometric-2.0.4-3C2179?style=flat&logo=pyg&logoColor=white
[lit-badge]: https://img.shields.io/badge/pytorch--lightning-1.7.0-792EE5?style=flat&logo=pytorchlightning&logoColor=white
<!-- Emails -->
[email-fg]: mailto:f.guerranti@tum.de
[email-dr]: mailto:david.reifferscheidt@tum.de
[email-sg]: mailto:s.guennemann@tum.de
<!-- Socials -->
[twitter-fg]: https://twitter.com/guerrantif
[twitter-sg]: https://twitter.com/guennemann
[daml-site-fg]: https://www.cs.cit.tum.de/en/daml/team/filippo-guerranti/
[daml-site-sg]: https://www.cs.cit.tum.de/en/daml/team/damlguennemann/
[personal-site-fg]: https://filippoguerranti.github.io
<!-- Institutions-->
[tum-site]: https://www.tum.de/en/
[daml-site]: https://www.cs.cit.tum.de/en/daml/home/
<!-- License website -->
[license-site]: https://choosealicense.com/licenses/mit/
<!-- Python & libraries websites -->
[python-site]: https://www.python.org
[pytorch-site]: https://pytorch.org
[pytorch-install-site]: https://pytorch.org/get-started/locally/
[pyg-site]: https://pytorch-geometric.readthedocs.io/en/latest/index.html#
[pyg-install-site]: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
[lit-site]: https://www.pytorchlightning.ai
[slurm-site]: https://slurm.schedmd.com/documentation.html
[pyenv-virtualenv-site]: https://github.com/pyenv/pyenv-virtualenv
[pyenv-site]: https://github.com/pyenv/pyenv
[pyenv-install-site]: https://github.com/pyenv/pyenv#installation
[conda-site]: https://docs.conda.io/en/latest/
[pip-site]: https://pip.pypa.io/en/stable/
<!-- Internal references -->
[installation-guide-ref]: ./docs/installation.md
<!-- Other variables -->
[example-site]: http://example.com
