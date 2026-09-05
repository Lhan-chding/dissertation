"""Model-independent probability and adapter boundary tests; no model weights."""
import unittest

from src.model_adapters.base import pure_generation_options, select_language_mlp_modules
from src.likelihood import completion_mask, cache_comparison
from src.grpo_update import grouped_advantages, ppo_surrogate


class ModelContractTests(unittest.TestCase):
    def test_sampling_discards_model_default_transforms(self):
        options = pure_generation_options(64)
        self.assertEqual(options['temperature'], 1.0)
        self.assertEqual(options['top_p'], 1.0)
        self.assertEqual(options['top_k'], 0)
        self.assertEqual(options['min_p'], 0.0)
        self.assertEqual(options['repetition_penalty'], 1.0)
        self.assertEqual(options['num_beams'], 1)
        self.assertIsNone(options['forced_eos_token_id'])
        self.assertIsNone(options['suppress_tokens'])
        with self.assertRaises(ValueError):
            pure_generation_options(0)

    def test_module_selection_never_selects_vision_or_attention(self):
        names = [f'model.language_model.layers.{i}.mlp.{p}'
                 for i in range(32) for p in ('gate_proj', 'up_proj', 'down_proj')]
        names += ['model.visual.blocks.0.mlp.up_proj',
                  'model.language_model.layers.0.self_attn.q_proj']
        selected = select_language_mlp_modules(names, 32)
        self.assertEqual(len(selected), 96)
        self.assertFalse(any('visual' in name for name in selected))
        with self.assertRaises(ValueError):
            select_language_mlp_modules(names[1:], 32)

    def test_mask_excludes_prompt_padding_includes_only_first_eos(self):
        self.assertEqual(completion_mask([4, 9, 2, 2, 0], 1, eos_ids={2}, pad_id=0),
                         [False, True, True, False, False])
        self.assertEqual(completion_mask([4, 9, 2, 2], 1, eos_ids={2}, pad_id=2),
                         [False, True, True, False])
        self.assertEqual(completion_mask([0, 4, 5, 6], 2, eos_ids={2}, pad_id=0),
                         [False, False, True, True])

    def test_reward_ties_and_population_epsilon_convention(self):
        values = grouped_advantages([3.0, 1.0], epsilon=0.0)
        self.assertEqual(values['advantages'], [1.0, -1.0])
        self.assertEqual(values['std'], 1.0)
        self.assertEqual(grouped_advantages([1.0] * 8)['advantages'], [0.0] * 8)
        with self.assertRaises(ValueError):
            grouped_advantages([float('nan')])

    def test_ppo_uses_fixed_length_not_per_sequence_length(self):
        # Four actual tokens over two sequences, B*K=2, fixed Lnorm=64.
        loss = ppo_surrogate([[0, 0, 0], [0]], [[0, 0, 0], [0]], [1, -1], 64)
        self.assertAlmostEqual(loss, -2 / 128)
        # Larger positive ratios clip; negative advantage takes larger ratio.
        import math
        self.assertAlmostEqual(ppo_surrogate([[math.log(2)]], [[0]], [1], 64), -1.2/64)
        self.assertAlmostEqual(ppo_surrogate([[math.log(2)]], [[0]], [-1], 64), 2/64)

    def test_cache_comparison_measures_thresholds_without_claiming_exactness(self):
        measured = cache_comparison([1, 2], [1, 2], [-1.0, -2.0], [-1.0, -2.001])
        self.assertTrue(measured['passed'])
        failed = cache_comparison([1, 2], [1, 3], [-1.0, -2.0], [-1.0, -2.001])
        self.assertFalse(failed['passed'])
        with self.assertRaises(ValueError):
            cache_comparison([], [], [], [])


if __name__ == '__main__':
    unittest.main()
