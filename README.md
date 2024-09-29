# AI CNC Implementation

This repository contains all codes that have been developed for the deliverable 6GSMART-SP3-L5-P2-P3-D2.3.2, which belongs to the task T2.2.3.

This file contains instructions for installing this code, its structure and usage guides.

## Installation

Before running the software, some requirements need to be fulfilled. For python packages, the versions that are mentioned are those that have been used, but others could be valid.

- python3 installed (used version 3.10.12)
- pip (version 22.0.2 or higher)
- colorlog v.6.8.2
- gymnasium v.0.29.1
- IPython v.8.24.0
- matplotlib v.3.8.4
- networkX v.3.3
- numpy v.1.26.4
- pandas 2.2.2
- torch 2.3.0

Python packages can be easily installed by typing the following console command:

```console
pip install <PackageName>
```

where `PackageName` is the name of the dependency to install.

This repository can be downloaded and set up by following these steps:

1. Open a console at the directory where the repository is desired to be downloaded.
2. Introduce the following command:

```console
git clone https://github.com/sergigarciacanton/AI_CNC_Implementation.git
```

3. Create necessary auxiliary directories inside the project's root folder. Type the following commands:

```console
cd AI_CNC_Implementation
mkdir models models/DQN models/DQN/cent models/DQN/dist plots plots/DQN plots/DQN/cent plots/DQN/dist logs
```

After following these steps, the code is ready to be used. The structure of files and directories inside the project's root folder should be as follows:

```bash
|-- README.md
|-- centralised
|   |-- Environment.py
|   |-- config.py
|   |-- dqn.py
|   |-- main.py
|   |-- vnf_generator.py
|-- distributed
|   |-- Environment.py
|   |-- Environment_A.py
|   |-- Environment_B.py
|   |-- config.py
|   |-- dqn.py
|   |-- dqn_A.py
|   |-- dqn_B.py
|   |-- main.py
|   |-- vnf_generator.py
|   |-- vnf_generator_A.py
|   |-- vnf_generator_B.py
|-- logs
|   |-- *.log
|-- models
|   |-- DQN
|       |-- cent
|           |-- *.pt
|       |-- dist
|           |-- *.pt
|-- plots
|   |-- DQN
|       |-- cent
|           |-- *.png
|       |-- dist
|           |-- *.png
```

## Code usage

Before running any script, it is important to configure both the agent and the environment as desired by setting the variables declared at the `config.py` file related to the model type that is desired (centralised or distributed). More detail about each variable's meaning is given in comments along the file.

Both centralised and distributed models can be trained and evaluated by running the `main.py` file that is located at their folders. Opening a console in the desired folder, they can be run by typing the following command:

```console
python3 main.py
```

Once it is executed, a simple console-based menu will appear asking for the operation mode to run. Depending on the model type, options are different.

For centralised models, options are the following ones:

0. Option 0 will train a centralized model following the configuration given in the `config.py` file. Once the option is selected, the program allows you to write additional information to be added to the log file. During training, logs will be printed with the agent's learning progress both on the screen and in the log file. At the end of the training, a plot with the evolution of rewards and losses of the entire training and the generated model will be saved in the folders configured in the configuration file.
1. Option 1 will evaluate a centralized model for all routes in the scenario. Once this option is entered, the program asks for the name of the file (with full path) where the model to be evaluated is saved.
2. Option 2 will evaluate a centralized model for the route configured in the `config.py` file. Once this option is entered, the program asks for the name of the file (with full path) where the model to be evaluated is saved.

For distributed models, the following options are available:

0. Option 0 will train a model from TSN zone A following the configuration given in the `config.py` file. Once the option is selected, the program allows you to write additional information to be added to the log file. During training, logs will be printed with the agent's learning progress both on the screen and in the log file. At the end of the training, a plot with the evolution of rewards and losses of the entire training and the generated model will be saved in the folders configured in the configuration file.
1. Option 1 will train a model from the TSN B zone following the configuration given in the `config.py` file. Once the option is selected, the program allows you to write additional information to be added to the log file. During training, logs will be printed with the agent's learning progress both on the screen and in the log file. At the end of the training, a plot with the evolution of rewards and losses of the entire training and the generated model will be saved in the folders configured in the configuration file.
2. Option 2 will evaluate a model of TSN zone A for all routes in the scenario. Once this option is entered, the program asks for the name of the file (with full path) where the model to be evaluated is saved.
3. Option 3 will evaluate a TSN zone A model for the route configured in the config.py file. Once this option is entered, the program asks for the name of the file (with full path) where the model to be evaluated is saved.
4. Option 4 will evaluate a model of the TSN B zone for all routes in the scenario. Once this option is entered, the program asks for the name of the file (with full path) where the model to be evaluated is saved.
5. Option 5 will evaluate a TSN B zone model for the route configured in the config.py file. Once this option is entered, the program asks for the name of the file (with full path) where the model to be evaluated is saved.
6. Option 6 will carry out a joint evaluation of a model from the TSN A zone and a model from the TSN B zone. Once this option has been entered, the program asks for the name of the files (with full path) where the models to be evaluated are saved.

