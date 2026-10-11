# A_02 NumPy Tutorial

ECE 5831, Fall 2026

For this assignment I went through the NumPy section of the CS231n Python tutorial (https://cs231n.github.io/python-numpy-tutorial/) and worked every example in a Jupyter notebook inside VS Code, using my ece-5831-2026 conda environment.

## What's in this folder

- numpy-tutorials.ipynb – the notebook. Each part of the tutorial gets its own section with the code as runnable cells, the outputs saved, and a few notes in Markdown where I wanted to remember why something works the way it does.
- README.md – this file.

## Notes

- Done in Jupyter in VS Code, not Colab.
- Python 3 via Anaconda, env ece-5831-2026, NumPy
- To rerun: open the notebook, select the ece-5831-2026 kernel, Run All. Nothing needs to be installed beyond NumPy.

# A_03 Logic Gates with a Perceptron
LogicGate class with AND, NAND, OR, NOR and XOR gates, implemented with NumPy.

## Files

- logic_gate.py: the LogicGate class
- module3.py: test script for the class
- module3.ipynb: the same tests in a Jupyter notebook
## How to run

- run python module3.py

## requirments

- NumPy

## A_04 Multilayer Perceptron

- MultilayerPerceptron class implemented with NumPy. The network has 3 input neurons, hidden layers with 5, 5 and 2 neurons, and 2 output neurons. Layers 1–3 use sigmoid, and layer 4 uses ReLU.

## Files

- multilayer_perceptron.py: the MultilayerPerceptron class
- module4.py: test script using example inputs
- module4.ipynb: forward-propagation results in a Jupyter notebook

## How to run

- run python module4.py

## Requirements
- NumPy
