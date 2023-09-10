# Solve mTSP with pymoo
Leveraging the pymoo framework, our package solves mTSP problems through the adept application of genetic algorithms -- NSGA-II. By constraining the original model designated for the ST-SR-TA MRTA problem, we have honed a tool capable of unraveling the intricacies of mTSP.


## Background
In its nascent stage, the primary objective of this project was to unlock solutions for the Single-Task Single-Robot Task-Allocation (ST-SR-TA) Multi-Robot Task Allocation (MRTA) issue. The journey towards achieving this milestone beckoned a reconfiguration of our strategy to focus on the mTSP problem, a well-established benchmark in the world of optimization.

## Getting Started
To dive in, visit our installation guide and documentation to set up the environment and acquaint yourself with the operational aspects of the package.

0. Prerequisite: Install poetry
    ```
    curl -sSL https://install.python-poetry.org | python3 -
    ```

1. Clone this repo
   ```
   git clone https://github.com/airuchen/mrta_pymoo.git
   ```
2. Install dependencies and activate the virtual env

    ```
    cd mrta_pymoo/
    poetry install
    poetry shell
    ```

3. Run the Benchmark by specifying the desired tsp dataset and the number of the robots.
    ```
    python3 ./mrta_pymoo/main.py
    ```

## Potential Applications
Beyond serving as a benchmarking tool, the potential of this package unfurls in various domains including logistics, supply chain management, and robotic path planning, offering solutions characterized by cost-efficiency and optimal routing.

## License
This project is licensed under the MIT License.