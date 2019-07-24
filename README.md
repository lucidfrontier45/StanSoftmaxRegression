# StanSoftmaxRegression

Stan implementation of scikit-learn style Softmax Regression (a.k.a Multi-Class Logistic Regression).
This model implements not only parameter inference but also prediction using posterior prediction.

## model

In Softmax Regression, the probability of class k is given by

$$
P(y=k|x, w) = \mathrm{softmax}(\mathrm{dot}(w, x))[k].
$$

Byesian inference tries to find the posterior distribution

$$
P(w|X, Y) = \frac{P(Y| w, X)P(w)}{P(Y|X)}
$$

where $P(w)$ is the prior distribution.

With the posterior samples $w_i$, the prediction of new data $x_{new}$ can be obtained by the following posterior average

$$
P(y_{new}=k|x_{new}, X, Y) = \sum_{i} P(y_{new}=k|x_{new}, w_i).
$$

## requirements

- numpy
- scipy
- scikit-learn
- pystan

## Example

```
$ python example.py --help
usage: StanSoftmaxRegression Example [-h] [--data DATA] [--mode MODE]

optional arguments:
  -h, --help   show this help message and exit
  --data DATA  dataset name, (iris|digits), default=iris
  --mode MODE  inference mode, (sampling|vb), default=vb
```
