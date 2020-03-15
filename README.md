## Setup

1. Make sure you have [Miniconda](https://conda.io/docs/user-guide/install/index.html#regular-installation) installed
    1. Conda is a package manager that sandboxes your project’s dependencies in a virtual environment
    2. Miniconda contains Conda and its dependencies with no extra packages by default (as opposed to Anaconda, which installs some extra packages)

2. cd into src, run `conda env create -f environment.yml`
    1. This creates a Conda environment called `squad`

3. Run `source activate squad`
    1. This activates the `squad` environment
    2. Do this each time you want to write/test your code
  
4. Run `python setup.py`
    1. This downloads SQuAD 2.0 training and dev sets, as well as the GloVe 300-dimensional word vectors (840B)
    2. This also pre-processes the dataset for efficient data loading
    3. For a MacBook Pro on the Stanford network, `setup.py` takes around 30 minutes total  

5. Browse the code in `train.py`
    0. Prior to running the `train.py` script, please type in your console `export CUDA_VISIBLE_DEVICES=0` and check that only one GPU is visible: `echo $CUDA_VISIBLE_DEVICES`. This is required as multi-GPU creates issues with the code.
    1. The `train.py` script is the entry point for training a model. It reads command-line arguments, loads the SQuAD dataset, and trains a model.
    2. You may find it helpful to browse the arguments provided by the starter code. Either look directly at the `parser.add_argument` lines in the source code, or run `python train.py -h`.
    3. To train the baseline, use `python train.py --model bidaf  --name baseline --num_workers 0`
    4. To train the bidafextra model, use `python train.py --model bidafextra  --name bidafextra --num_workers 0`
    5. To train the FusionNet model, use `python train.py --model fusionnet  --name fusionnet --num_workers 0`
