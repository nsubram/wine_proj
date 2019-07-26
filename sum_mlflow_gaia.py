from acumos.modeling import Model
from acumos.session import AcumosSession

import os
import warnings
import sys

import numpy as np

import mlflow

session = AcumosSession()

def add_numbers(x: int, y: int) -> int:
    '''Returns the sum of x and y'''
    return x + y

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    a = float(sys.argv[1]) if len(sys.argv) > 1 else 1
    b = float(sys.argv[2]) if len(sys.argv) > 2 else 2

    with mlflow.start_run():
        sum = add_numbers(a, b)

        print("Addition code (a=%f, b=%f):" % (a, b))
        print("  Sum: %f" % sum)

        mlflow.log_param("a", a)
        mlflow.log_param("b", b)
        mlflow.log_metric("sum", sum)

        model = Model(add=add_numbers)

        session.dump(model, 'my-model', '.')  # creates ~/my-model
