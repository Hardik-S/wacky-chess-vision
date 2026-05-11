import unittest

from src.wacky_chess_vision import LABELS, evaluate, generate_samples, predict_baseline, render_board_snapshot


class WackyChessVisionTests(unittest.TestCase):
    def test_generator_balances_labels_deterministically(self):
        samples = generate_samples(seed=42, samples=18)
        labels = [sample.label for sample in samples]

        self.assertEqual(labels.count("standard"), 6)
        self.assertEqual(labels.count("rotated"), 6)
        self.assertEqual(labels.count("occluded"), 6)
        self.assertEqual(tuple(sorted(set(labels))), tuple(sorted(LABELS)))

    def test_baseline_matches_synthetic_label_rules(self):
        samples = generate_samples(seed=42, samples=18)
        predictions = [predict_baseline(sample) for sample in samples]

        self.assertEqual(predictions, [sample.label for sample in samples])
        self.assertEqual(evaluate(samples)["accuracy"], 1.0)

    def test_occluded_samples_hide_at_least_one_anchor(self):
        samples = generate_samples(seed=7, samples=12)
        occluded = [sample for sample in samples if sample.label == "occluded"]

        self.assertTrue(occluded)
        self.assertTrue(all(len(sample.visible_anchors) < 6 for sample in occluded))

    def test_render_board_snapshot_marks_visible_and_hidden_anchors(self):
        sample = next(sample for sample in generate_samples(seed=7, samples=12) if sample.label == "occluded")
        snapshot = render_board_snapshot(sample)

        self.assertIn("sample=synthetic-0002 label=occluded rotation=0", snapshot)
        self.assertIn("a1", snapshot)
        self.assertIn("--", snapshot)
        self.assertEqual(len(snapshot.splitlines()), 10)


if __name__ == "__main__":
    unittest.main()

