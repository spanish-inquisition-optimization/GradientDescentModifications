import numpy.linalg.linalg

from core.utils import CallsCount
from core.gradient_descent import *
from core.utils import *


def test_batch(fs, dfs, x0, scheduler):
    f = fn_sum(*fs)
    df = fn_sum(*dfs)

    target_point = steepest_descent(
        f, df, x0,
        bin_search,
        lambda f, steps: len(steps) > 100
    )[-1]

    target_val = f(target_point)
    # print(f"target point = {target_point}, with value = {target_val}")

    result = []
    for batch in range(1, len(fs) + 1):
        points = gradient_descent_minibatch(
            fs, dfs, batch, x0,
            scheduler(batch),
            lambda f, steps: abs(f(steps[-1]) - target_val) < 0.01 or len(steps) > 200
        )

        result.append((f"batch = {batch}", len(points)))

    return result



def test_perfomance(funs, dfuns, batch_size, x0):
    wrap_input = lambda: ([CallsCount(f) for f in funs], [CallsCount(f) for f in dfuns])
    f1, df1 = wrap_input()
    f2, df2 = wrap_input()
    f3, df3 = wrap_input()
    f4, df4 = wrap_input()
    f5, df5 = wrap_input()

    def terminate(f, steps):
        return f(steps[-1]) < 0.001 or len(steps) > 100

    gradient_descent_minibatch(
        f1, df1, batch_size, x0,
        exponential_learning_scheduler(0.3, 0.2, batch_size, len(funs)),
        terminate
    )

    gradient_descent_minibatch_with_momentum(0.2)(
        f2, df2, batch_size, x0,
        exponential_learning_scheduler(0.3, 0.2, batch_size, len(funs)),
        terminate
    )

    gradient_descent_minibatch_with_momentum(0.7, True)(
        f3, df3, batch_size, x0,
        exponential_learning_scheduler(0.3, 0.2, batch_size, len(funs)),
        terminate
    )

    gradient_descent_minibatch_adagrad(
        f4, df4, batch_size, x0,
        fixed_step_search(5),
        terminate
    )

    gradient_descent_minibatch_rms_prop(0.2)(
        f5, df5, batch_size, x0,
        exponential_learning_scheduler(0.3, 0.2, batch_size, len(funs)),
        terminate
    )

    return [
        (f"Minibatch", sum(f.calls for f in df1)),
        (f"Momentum", sum(f.calls for f in df2)),
        (f"Nesterov", sum(f.calls for f in df3)),
        (f"AdaGrad", sum(f.calls for f in df4)),
        (f"RMSProp", sum(f.calls for f in df5)),
    ]
