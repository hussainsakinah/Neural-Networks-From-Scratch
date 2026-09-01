# Neural Networks From Scratch

A scalar-valued automatic differentiation (autograd) engine and neural network library, built entirely from first principles — no PyTorch, no TensorFlow/Keras, no autodiff libraries. Every operation, gradient, and backward pass is implemented and derived by hand.

## Why this exists

Most of my other work sits on top of high-level frameworks (TensorFlow/Keras, LlamaIndex). This project goes the opposite direction: strip everything away and rebuild backpropagation and gradient descent from the ground up, to actually understand what those frameworks are doing underneath.

Before writing a line of code, I worked through the derivatives by hand on paper — chain rule, gradients through each operation — and only then translated that understanding into code, also worked through manually before running it.

## Acknowledgment

This project follows the structure and spirit of [Andrej Karpathy](https://karpathy.ai/)'s *"Neural Networks: Zero to Hero"* course, specifically the [micrograd](https://github.com/karpathy/micrograd) lecture on building an autograd engine from scratch. All code, comments, explanations, and the notebook itself are my own work — typed and reasoned through independently, not copied — but the course was the guide and inspiration for this build, and full credit for the teaching goes to him.

## What's inside

- **[`micrograd_from_scratch.ipynb`](https://github.com/hussainsakinah/Neural-Networks-From-Scratch/blob/main/micrograd_from_scratch.ipynb)** — the complete implementation, with my own explanations and notes woven in throughout, covering:
  - A `Value` object that tracks data, gradients, and the computational graph
  - Manual implementation of core operations (addition, multiplication, power, tanh, etc.) with their backward passes
  - Reverse-mode automatic differentiation via topological sort and backpropagation
  - A small neural network (neurons → layers → MLP) built on top of the engine
  - Training on toy examples using manual gradient descent

## Key concepts covered

- Computational graphs and how forward/backward passes move through them
- The chain rule as the mechanical engine behind backpropagation
- Why gradients accumulate at nodes with multiple outgoing edges
- How a neural network is really just a large expression graph
- Gradient descent implemented manually, without an optimizer library

## How to run

Open the notebook directly in Google Colab or Jupyter:

```bash
git clone https://github.com/hussainsakinah/Neural-Networks-From-Scratch.git
cd Neural-Networks-From-Scratch
jupyter notebook micrograd_from_scratch.ipynb
```

No dependencies beyond a standard Python environment — the engine itself has no external ML library requirements.

## Status

Work in progress — actively being extended as I continue through the course material.

## Author

**Sakinah Faiza Hussain**
[GitHub](https://github.com/hussainsakinah) · [LinkedIn](https://www.linkedin.com/in/sakinah-hussain-410138302/)
