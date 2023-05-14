import numpy.linalg.linalg

from core.utils import CallsCount
from core.gradient_descent import *
from core.utils import *


def test_batch(fs, dfs, x0):
    f = fn_sum(*fs)
    df = fn_sum(*dfs)

    target_point = steepest_descent(
        f, df, x0,
        bin_search,
        lambda f, steps: len(steps) > 100
    )[-1]

    target_val = f(target_point)
    print(f"target point = {target_point}, with value = {target_val}")

    for batch in range(1, len(fs) + 1):
        point = gradient_descent_minibatch(
            fs, dfs, batch, x0,
            exponential_learning_scheduler(0.3, 0.2),
            lambda f, steps: len(steps) > 20
        )[-1]

        print(f"with batch = {batch} point = {point}, value = {f(point)}")



def test_perfomance(funs, dfuns, batch_size, x0):
    wrap_input = lambda fs, dfs: ([CallsCount(f) for f in fs], [CallsCount(f) for f in dfs])
    target_point = steepest_descent(
        fn_sum(*funs), fn_sum(*dfuns), x0,
        wolfe_conditions_search(0.9, 0.95),
        lambda f, steps: len(steps) > 40
    )[-1]

    target_val = fn_sum(*funs)(target_point)

    print(target_point)
    print(target_val)

    f1, df1 = wrap_input(funs, dfuns)
    f2, df2 = wrap_input(funs, dfuns)
    f3, df3 = wrap_input(funs, dfuns)

    def terminate(f, steps):
        # print(f"last is {steps[-1]}")
        # print(f"f = {f(steps[-1])}")
        return abs(f(steps[-1]) - target_val) < 1

    gradient_descent_minibatch(
        f1, df1, batch_size, x0,
        exponential_learning_scheduler(0.3, 0.2),
        terminate
    )

    gradient_descent_minibatch_with_momentum(0.5)(
        f2, df2, batch_size, x0,
        exponential_learning_scheduler(0.3, 0.2),
        terminate
    )

    gradient_descent_minibatch_with_momentum(0.5, True)(
        f3, df3, batch_size, x0,
        exponential_learning_scheduler(0.3, 0.2),
        terminate
    )

    return [
        (f"{batch_size}-Minibatch gradient descent", sum(f.calls for f in df1)),
        (f"Minibatch with momentum", sum(f.calls for f in df2)),
        (f"Nesterov", sum(f.calls for f in df3))
    ]
