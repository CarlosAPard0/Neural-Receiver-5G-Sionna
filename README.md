# Neural Receiver in 5G - Sionna

This project is based on an existing example from the [Sionna repository](https://github.com/NVlabs/sionna/tree/main).  
The goal is to extend and adapt it in order to provide **more flexibility** for experimenting with different neural receiver architectures and custom configurations.

---

## 🚀 Project Overview

- Built on **[Sionna](https://github.com/NVlabs/sionna/blob/main/tutorials/phy/Neural_Receiver.ipynb)**, NVIDIA’s link-level simulator for 5G and beyond.  
- Runs inside a **Docker container** for full control and reproducibility (see the included `Dockerfile`).  
- Provides an easy way to:
  - Define and test new **neural receiver architectures**.
  - Customize training and simulation settings via YAML configuration files.
  - Run experiments and store results in an organized structure.

---

## 🧠 Neural Receiver Architectures

All neural receiver models are stored inside the `Architectures/` folder.  
To integrate a new architecture, you must also update the mapping inside `architecture_factory.py`:

```python
architecture_map = {  
    'neural_receiver1': 'Architecture.neural_receiver1.NeuralReceiver',  
    'neural_receiver2': 'Architecture.neural_receiver2.NeuralReceiverInception',  
    'neural_receiver3': 'Architecture.neural_receiver3.NeuralReceiverSE',  
    'neural_receiver4': 'Architecture.neural_receiver4.NeuralReceiverCNNLSTM',
}
```
## ⚙️ Configuration

Experiment settings are fully controlled via **YAML files** located in the `Config/` folder.  
Each configuration file includes:

- **Simulation parameters** (modulation, Eb/No range, number of samples, frequency, sub carriers, etc.)
- **Training setup** (batch size, epochs, optimizer, learning rate, resume previous train, etc.)
- **Testing setup** (evaluation metrics, checkpoints, etc.)

This approach allows you to easily switch between experiments without modifying the core code.

---

## ▶️ Usage

### Training a model
Run the main script in training mode:

```bash
python3 main.py --mode train --config Config/path_to_config.yaml
```

### Testing / Evaluation

```bash
python3 main.py --mode test --config Config/path_to_config.yaml
```

All results (CSV logs and plots) are automatically stored under the **Results/** folder.

## 📂 Repository Structure
```bash
├── Architectures/           # Neural receiver implementations
│   ├── neural_receiver1.py
│   ├── neural_receiver2.py
│   ├── neural_receiver3.py
│   └── architecture_factory.py  # Mapping between architecture names and classes
│
├── Config/                  # YAML configuration files
├── Results/                 # Training & evaluation results (CSV + plots)
├── Dockerfile               # Environment for reproducibility
└── main.py                  # Entry point for training and testing


```
