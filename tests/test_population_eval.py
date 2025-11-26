import torch

from hyperfunc import Example, HyperSystem, LMParam, hyperfunction


def test_evaluate_population_matches_single_eval():
    @hyperfunction(hp_type=LMParam, optimize_hparams=True)
    def target_fn(x: float, hp: LMParam) -> float:
        # Simple objective that depends only on temperature.
        return -abs(hp.temperature - x)

    # Create a custom system that routes inputs to target_fn
    class TestSystem(HyperSystem):
        def run(self, x: float):
            return target_fn(x)

    system = TestSystem()
    system.register_hyperfunction(target_fn)

    # One example where the "target" is 0.5 for temperature.
    # New Example format: just inputs (kwargs to run) and expected output
    examples = [Example({"x": 0.5}, 0.0)]

    def metric_fn(preds, expected):
        # Average the objective values.
        return sum(preds) / len(preds)

    # Build two candidate hp states with different temperatures.
    with torch.no_grad():
        base_state = system.get_hp_state()
        base_vec = base_state["target_fn"].detach().clone()

        # Candidate A: temperature at 0.0
        hp_a = base_vec.clone()
        hp_a[0] = 0.0

        # Candidate B: temperature at 0.5
        hp_b = base_vec.clone()
        hp_b[0] = 0.5

    scores = system.evaluate_population(
        [
            {"target_fn": hp_a},
            {"target_fn": hp_b},
        ],
        examples,
        metric_fn,
    )

    # Evaluate individually and compare.
    with torch.no_grad():
        system.set_hp_state({"target_fn": hp_a})
    score_a_single = system.evaluate(examples, metric_fn)

    with torch.no_grad():
        system.set_hp_state({"target_fn": hp_b})
    score_b_single = system.evaluate(examples, metric_fn)

    assert scores[0] == score_a_single
    assert scores[1] == score_b_single
