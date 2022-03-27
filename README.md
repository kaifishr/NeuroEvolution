# NeuroEvolution

A hybrid optimization algorithm combining gradient descent for network parameter optimization 
coupled with an evolutionary algorithm for network topology and training hyperparameter 
optimization.

Work in progress.

## Networks

Currently, only fully-connected neural networks (Multilayer perceptrons (MLPs)) are supported.

## Mutation Operators

This implementation uses a very simple mutation operator for hyperparameter and network topology
optimization.

```python
eta = value * local_mutation_rate * random.random()
value = value + eta
```

Although genetic optimization already works well with this simple rule, many improvements are 
conceivable.

## Random Subset Dataloader

To speed up genetic optimization, a custom dataloader allows to train every epoch on a new set of
randomly selected data coming from the original dataset using a random mapping from subset indices
to original dataset indices:

```python
subset_length = int(len(data) * subset_ratio)
rand_map = random.sample(list(range(len(data))), subset_length)
```

Random subsets are also used for testing the current population to further accelerate the genetic
optimization process.

Setting the random subset ratio to small values such as 1 % significantly accelerates optimization.
