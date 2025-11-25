# Hybrid Quantum GAN

A hybrid quantum-classical Generative Adversarial Network project.

## Project Structure

```
├── data/           # Datasets and data processing
├── models/         # Classical and hybrid model architectures
├── quantum/        # Quantum circuits and quantum layers
├── modules/        # Reusable components and modules
├── utils/          # Utility functions and helpers
├── results/        # Output results
│   └── screenshots/  # Generated images and visualizations
```

## Setup

### Using Conda (Recommended)

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate quantum-gan
```

### Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- **PyTorch**: Deep learning framework
- **PennyLane**: Quantum machine learning library
- **Qiskit**: IBM's quantum computing framework
- **RDKit**: Cheminformatics and molecular modeling
- **NetworkX**: Graph/network analysis
- **NumPy**: Numerical computing
- **Matplotlib**: Visualization

