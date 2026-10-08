"""Check deterministic top-k pair targets and daily gradient isolation."""
import unittest

import torch
from torch.nn import functional as F

from research.models import loss_by_day, rank_loss
from research.ranking import topk_membership_loss


class RankingTests(unittest.TestCase):
    def test_topk_loss_and_gradients_match_an_explicit_pair_list(self):
        scores = torch.tensor([0.5, -0.1, 0.8, -0.3, 0.0], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([4.0, 1.0, 3.0, 0.0, 2.0], dtype=torch.float64)
        actual = topk_membership_loss(scores, labels, topk=2)
        reference = torch.stack([F.softplus(scores[j]-scores[i]) for i in [0, 2] for j in [1, 3, 4]]).mean()
        torch.testing.assert_close(actual, reference)
        actual_gradient, = torch.autograd.grad(actual, scores, retain_graph=True)
        reference_gradient, = torch.autograd.grad(reference, scores)
        torch.testing.assert_close(actual_gradient, reference_gradient)
        self.assertTrue((actual_gradient[[0, 2]] < 0).all())
        self.assertTrue((actual_gradient[[1, 3, 4]] > 0).all())

    def test_ties_do_not_create_preferences_and_order_does_not_change_targets(self):
        scores = torch.tensor([0.8, -0.2, 0.5, -1.0], requires_grad=True)
        labels = torch.tensor([2.0, 1.0, 1.0, 0.0])
        # Tied boundary stocks share membership and must not depend on input order.
        order = torch.tensor([2, 3, 0, 1])
        actual = topk_membership_loss(scores, labels, 2)
        torch.testing.assert_close(actual, topk_membership_loss(scores[order], labels[order], 2))
        expected = (0.5*F.softplus(scores[1]-scores[0]) +
                    0.5*F.softplus(scores[2]-scores[0]) +
                    F.softplus(scores[3]-scores[0]) +
                    0.5*F.softplus(scores[3]-scores[1]) +
                    0.5*F.softplus(scores[3]-scores[2]))/3.0
        torch.testing.assert_close(actual, expected)
        tied = topk_membership_loss(scores, torch.ones(4), 2)
        self.assertEqual(tied.item(), 0.0)
        gradient, = torch.autograd.grad(tied, scores)
        torch.testing.assert_close(gradient, torch.zeros_like(scores))
        small = topk_membership_loss(scores, labels, 30)
        self.assertEqual(small.item(), 0.0)

    def test_packed_top30_losses_and_gradients_match_separate_dates(self):
        torch.manual_seed(19)
        scores = torch.randn(101, dtype=torch.float64, requires_grad=True)
        labels = torch.randn(101, dtype=torch.float64)
        counts = [47, 54]
        packed = loss_by_day(scores, labels, counts, "top30_pair").mean()
        reference = 0.5*(rank_loss(scores[:47], labels[:47], "top30_pair") +
                         rank_loss(scores[47:], labels[47:], "top30_pair"))
        torch.testing.assert_close(packed, reference)
        packed_gradient, = torch.autograd.grad(packed, scores, retain_graph=True)
        reference_gradient, = torch.autograd.grad(reference, scores, retain_graph=True)
        torch.testing.assert_close(packed_gradient, reference_gradient)
        self.assertGreater(abs((packed-rank_loss(scores, labels, "top30_pair")).item()), 1e-4)


if __name__ == "__main__":
    unittest.main()
