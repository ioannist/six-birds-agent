from sbt_agency.viability import viability_kernel, viability_kernel_history


def _setup():
    states = [0, 1, 2, 3]
    actions = [0, 1]

    def safe(s):
        return s != 3

    def feasible_actions(_s):
        return actions

    transitions = {
        (0, 0): {0},
        (0, 1): {1},
        (1, 0): {1},
        (1, 1): {2},
        (2, 0): {2, 3},
        (2, 1): {3},
        (3, 0): {3},
        (3, 1): {3},
    }

    def post_support(s, a):
        return transitions[(s, a)]

    return states, actions, feasible_actions, post_support, safe


def test_viability_kernel_exact():
    states, actions, feasible_actions, post_support, safe = _setup()
    K = viability_kernel(states, actions, feasible_actions, post_support, safe)
    assert K == {0, 1}


def test_viability_kernel_history():
    states, actions, feasible_actions, post_support, safe = _setup()
    hist = viability_kernel_history(states, actions, feasible_actions, post_support, safe)

    assert hist[0] == {0, 1, 2}
    for i in range(len(hist) - 1):
        assert hist[i + 1].issubset(hist[i])
    assert hist[-1] == {0, 1}
    assert len(hist) <= len(states) + 1

