# KronEM in Python

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.1007/978--3--031--53468--3_22-blue)](https://doi.org/10.1007/978-3-031-53468-3_22)

## Overview

This repository provides the source code for the controlled experiments used as a baseline comparison in the research paper, **"Graph Completion Through Local Pattern Generalization."** The experiments leverage the KronEM algorithm, a classic method for fitting Kronecker graph models, to establish a benchmark for evaluating graph completion methodologies.

The implementation is written in Python and interfaces with the C++ source code of the KronEM algorithm from the Stanford Network Analysis Platform (SNAP) library.

## Publication & Citation

This code is a supplementary component of our publication. If you use this code or the experimental results in your research, please cite our paper:

Zhang, Z., Tao, R.†, Tao, Y.†, Qi, M., Zhang, J. (2024). Graph Completion Through Local Pattern Generalization. In: Cherifi, H., Rocha, L.M., Cherifi, C., Donduran, M. (eds) *Complex Networks & Their Applications XII*. COMPLEX NETWORKS 2023. Studies in Computational Intelligence, vol 1141. Springer, Cham.

**DOI:** `https://doi.org/10.1007/978-3-031-53468-3_22`

†*These authors contributed equally.*

```bibtex
@inproceedings{zhang2024graph,
  title={Graph Completion Through Local Pattern Generalization},
  author={Zhang, Zhang and Tao, Ruyi and Tao, Yongzai and Qi, Mingze and Zhang, Jiang},
  booktitle={Complex Networks \& Their Applications XII},
  pages={260--271},
  year={2024},
  organization={Springer}
}
```

## Background on KronEM

KronEM is an Expectation-Maximization (EM) algorithm used to estimate the initiator matrix for stochastic Kronecker graphs. These generative models can produce synthetic networks that mimic the statistical properties of real-world graphs. In our work, KronEM serves as a baseline by:
1.  Fitting a generative model to an observed subgraph.
2.  Sampling from this model to infer missing nodes and edges.

## Getting Started

### Prerequisites

*   Python 3.8+

### Installation and Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ytao-tcd/KronEM.git
    cd KronEM
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run an experiment:**
    ```bash
    python src/KronEM.py
    ```
    Please refer to the source code for detailed arguments and configuration options.

## Contribution

As a co-second author of the publication, I created this repository to ensure the reproducibility and transparency of our baseline comparisons. Contributions and suggestions are welcome. Please open an issue or submit a pull request.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
