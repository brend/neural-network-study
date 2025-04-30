# Neural Network Study

An implementation of a toy [neural network](https://en.wikipedia.org/wiki/Neural_network_(machine_learning)) from scratch. The idea is to learn about the workings of neural networks and the underlying math, but also to produce something useful.

This project follows a series on neural networks on [The Coding Train](https://youtu.be/ntKn5TPHHAk?si=0DsBd9O9Rp-bqC-H).

## What does it do?

### Stage 1: Perceptron

The first step is creating a perceptron, a single "neuron" that can be used to solve linearly separable problems. An example is classifying points on a plane to determine if they're above or below a line:

<figure>
  <img src="assets/point-classification.png" alt="Depiction of point classification">
  <figcaption>The perceptron identifies if points are above or below the line</figcaption>
</figure>

### Stage 2: Multilayer Perceptron - The Neural Network

Once the principle of the perceptron is understood, the next major step in developing a neural network is to create a *multilayer perceptron*, that is, a small network with inputs, a hidden layer, and outputs.

This evolution of the concept can solve a variety of problems. A classic example is a game of moving a basket to catch falling apples -- and in this enhanced version, avoid the sour lemons!

<img src="assets/catch-the-apples.gif" alt="Animation showing a basket moving left and right to catch falling apples">

## About This Project

![Made with Rust](https://img.shields.io/badge/Made%20with-Rust-orange)
