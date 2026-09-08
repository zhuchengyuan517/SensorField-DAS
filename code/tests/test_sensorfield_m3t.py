import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import SensorFieldM3T
from models.sensorfield_m3t import GCTI


class SensorFieldM3TShapeTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = SensorFieldM3T(
            task_output_dims={"event_type": 4, "distance_cls": 3},
            hidden_dim=64,
            num_anchors=6,
            num_heads=4,
            fac_loss_weight=0.2,
            taef_loss_weight=0.1,
            gcti_loss_weight=0.1,
            view_drop_prob=1.0,
            enable_view_consistency=True,
            view_consistency_weight=0.05,
            return_auxiliary=True,
        )

    def test_tensor_input_shapes(self) -> None:
        self.model.train()
        inputs = torch.randn(2, 1, 6, 192)
        task_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
        outputs = self.model(inputs, task_mask=task_mask)

        self.assertEqual(outputs["event_type"].shape, (2, 4))
        self.assertEqual(outputs["distance_cls"].shape, (2, 3))
        self.assertEqual(outputs["location"].shape, (2, 3))
        self.assertEqual(outputs["fac_outputs"]["shared_anchors"].shape, (2, 6, 64))
        self.assertEqual(outputs["taef_outputs"]["task_tokens"].shape, (2, 2, 64))
        self.assertEqual(outputs["taef_outputs"]["evidence_reliability"].shape, (2, 4))
        self.assertEqual(outputs["gcti_outputs"]["relation_matrix"].shape, (2, 2, 2))
        self.assertEqual(tuple(outputs["view_reliability"].keys()), ("raw", "stf", "gaf"))
        self.assertIn("fac_loss", outputs["aux_losses"])
        self.assertIn("taef_loss", outputs["aux_losses"])
        self.assertIn("gcti_loss", outputs["aux_losses"])
        self.assertIn("gcti_consistency", outputs)
        self.assertTrue(torch.isfinite(outputs["gcti_consistency"]["prediction"]))
        self.assertTrue(torch.isfinite(outputs["gcti_consistency"]["representation"]))

    def test_dict_input_shapes(self) -> None:
        self.model.eval()
        batch = {
            "raw": torch.randn(2, 1, 384),
            "stft": torch.randn(2, 1, 64, 64),
            "gaf": torch.randn(2, 1, 64, 64),
        }
        outputs = self.model(batch)
        self.assertEqual(outputs["event_type"].shape, (2, 4))
        self.assertEqual(outputs["distance_cls"].shape, (2, 3))

    def test_gcti_matches_paper_relation_and_residual_equations(self) -> None:
        module = GCTI(num_tasks=2, hidden_dim=8, num_heads=2, dropout=0.0)
        module.eval()
        task_tokens = torch.randn(3, 2, 8)

        outputs = module(task_tokens)
        query = module.q_proj(task_tokens)
        key = module.k_proj(task_tokens)
        expected_relation = torch.softmax(
            torch.matmul(query, key.transpose(-1, -2)) / (8**0.5)
            + module.task_relation_bias.unsqueeze(0),
            dim=-1,
        )
        expected_updated = task_tokens + module.out_proj(torch.matmul(expected_relation, task_tokens))

        torch.testing.assert_close(outputs["relation_matrix"], expected_relation)
        torch.testing.assert_close(outputs["updated_task_tokens"], expected_updated)
        torch.testing.assert_close(
            outputs["relation_matrix"].sum(dim=-1),
            torch.ones(3, 2),
        )


if __name__ == "__main__":
    unittest.main()
