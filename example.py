from sklearn import datasets, model_selection
from stansotfmax import StanSoftmaxRegression


def main(data="iris", mode="sampling"):
    if data == "iris":
        data = datasets.load_iris()
    else:
        data = datasets.load_digits()

    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.1)
    model = StanSoftmaxRegression(mode=mode)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    for y_true, p in zip(y_test, probs):
        print("true class = {}, ".format(y_true),
              "predicted prob = ", p.round(3))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog="StanSoftmaxRegression Example")
    parser.add_argument("--data", default="iris",
                        help="dataset name, (iris|digits), default=iris")
    parser.add_argument("--mode", default="sampling",
                        help="inference mode, (sampling|vb), default=vb")
    args = parser.parse_args()

    if args.data not in ["iris", "digits"]:
        raise ValueError("data must be either 'iris' or 'digits'")

    if args.mode not in ["sampling", "vb"]:
        raise ValueError("mode must be either 'sampling' or 'vb'")

    main(args.data, args.mode)
