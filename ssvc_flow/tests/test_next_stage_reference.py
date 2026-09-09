import itertools
import math
import unittest

from reference.finite_group_reference import (
    bank_group_rates,
    population_group_rates,
    zero_epsilon_sensitivity_energy,
)
from reference.finite_group_reference import (
    normalized_advantage_and_derivative as evaluate,
)


class MathTests(unittest.TestCase):
    def test_all_valid_constant_shift(self):
        cs = ["X", "S", "W", "W"]
        a0, _ = evaluate(cs, 0)
        for lam in [1e-6, 0.01, 1, 2]:
            a, d = evaluate(cs, lam)
            self.assertLess(max(abs(x - y) for x, y in zip(a, a0, strict=False)), 1e-12)
            self.assertEqual(d, [0] * 4)

        for c in ["X", "S", "W", "I"]:
            a, d = evaluate([c] * 4, 0, epsilon=0)
            self.assertEqual(a, [0] * 4)
            self.assertEqual(d, [0] * 4)

    def test_two_level_positive_scale_invariance(self):
        for cs in [["X", "I", "I"], ["W", "S", "I"]]:
            a, d = evaluate(cs, 0.01, epsilon=0)
            b, _ = evaluate(cs, 2, epsilon=0)
            self.assertLess(max(abs(x - y) for x, y in zip(a, b, strict=False)), 1e-12)
            self.assertLess(max(abs(x) for x in d), 1e-10)

    def test_activation_not_continuous_at_zero_epsilon(self):
        cs = ["W", "S", "I", "I"]
        a, d = evaluate(cs, 0, epsilon=0)
        self.assertEqual(a, [0] * 4)
        self.assertTrue(all(math.isnan(x) for x in d))
        b, _ = evaluate(cs, 1e-8, epsilon=0)
        self.assertAlmostEqual(sum(x * x for x in b) / len(b), 1)

    def test_derivative_finite_difference(self):
        cs = ["X", "W", "W", "S", "I", "I", "I", "I"]
        for lam in [0.01, 0.25, 1, 2]:
            _a, d = evaluate(cs, lam)
            h = 1e-6
            ap, _ = evaluate(cs, lam + h)
            am, _ = evaluate(cs, lam - h)
            fd = [(x - y) / (2 * h) for x, y in zip(ap, am, strict=False)]
            self.assertLess(max(abs(x - y) for x, y in zip(fd, d, strict=False)), 1e-7)

    def test_energy_formula(self):
        for cs in [["X", "W", "I"], ["X"] * 3 + ["S"] * 2 + ["W"] + ["I"] * 2]:
            for lam in [0, 0.01, 0.25, 1, 2]:
                _, d = evaluate(cs, lam, epsilon=0)
                self.assertAlmostEqual(
                    sum(x * x for x in d) / len(d),
                    zero_epsilon_sensitivity_energy(cs, lam),
                    places=11,
                )

    def test_k2_no_three_level(self):
        for pair in itertools.product(["X", "S", "W", "I"], repeat=2):
            _a, d = evaluate(list(pair), 0.25, epsilon=0)
            if not any(math.isnan(v) for v in d):
                self.assertLess(max(abs(v) for v in d), 1e-10)
        self.assertAlmostEqual(
            population_group_rates([0.2, 0.1, 0.3, 0.4], 2)["three_level_X_B_I"], 0
        )

    def test_bank_subsets_agree_with_enumeration(self):
        bank = list("XXSWWIII")
        k = 4
        total = list(itertools.combinations(range(len(bank)), k))
        direct = {name: 0 for name in bank_group_rates([2, 1, 2, 3], k)}
        for idx in total:
            group = [bank[j] for j in idx]
            nx = group.count("X")
            ni = group.count("I")
            nb = k - nx - ni
            flags = {
                "no_X": nx == 0,
                "X_informative": 0 < nx < k,
                "validity_mixed": 0 < ni < k,
                "aux_only_activation": nx == 0 and ni > 0 and nb > 0,
                "three_level_X_B_I": nx > 0 and ni > 0 and nb > 0,
            }
            for name, v in flags.items():
                direct[name] += int(v) / len(total)
        expected = bank_group_rates([2, 1, 2, 3], k)
        for name in direct:
            self.assertAlmostEqual(direct[name], expected[name])

    def test_bank_estimator_unbiased_over_iid_banks(self):
        # Enumerate every possible n=6 bank under one three-state population.
        probs = {"X": 0.2, "W": 0.3, "I": 0.5}
        accum = {name: 0.0 for name in bank_group_rates([1, 0, 2, 3], 3)}
        for bank in itertools.product(probs, repeat=6):
            mass = math.prod(probs[c] for c in bank)
            rates = bank_group_rates([bank.count(c) for c in ["X", "S", "W", "I"]], 3)
            for name in accum:
                accum[name] += mass * rates[name]
        expected = population_group_rates([0.2, 0, 0.3, 0.5], 3)
        for name in accum:
            self.assertAlmostEqual(accum[name], expected[name], places=12)

    def test_no_extrapolation(self):
        with self.assertRaises(ValueError):
            bank_group_rates([1, 1, 6, 8], 32)


if __name__ == "__main__":
    unittest.main()
