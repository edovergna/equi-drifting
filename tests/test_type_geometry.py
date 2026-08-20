"""Regression tests for configurable atom-type geometry."""

import sys
import unittest
from unittest.mock import patch

import torch

from model.drift_loss import compute_drift_loss
from model.lit_module import MoleculeGenerator
from model.spherical_utils import probs_to_sphere
from parse_args import parse_args


class ParseArgsTests(unittest.TestCase):
    def test_spherical_space_defaults_to_enabled(self):
        with patch.object(sys, "argv", ["train.py"]):
            self.assertTrue(parse_args().spherical_space)

    def test_spherical_space_can_be_disabled(self):
        with patch.object(sys, "argv", ["train.py", "--no-spherical_space"]):
            self.assertFalse(parse_args().spherical_space)


class TypeGeometryTests(unittest.TestCase):
    @staticmethod
    def _loss_config(**overrides):
        config = {
            "eps": 1e-8,
            "p_eta": 0.5,
            "t_eta": 0.5,
            "p_sigma": 1.0,
            "t_sigma": 1.0,
            "scale_eucl": 1.0,
            "scale_spher": 1.0,
            "num_atom_types": 3,
        }
        config.update(overrides)
        return config

    @staticmethod
    def _inputs(spherical_space):
        gen_pos = torch.tensor(
            [
                [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
                [[0.0, -0.5, 0.0], [0.0, 0.5, 0.0]],
            ],
            requires_grad=True,
        )
        real_pos = torch.tensor(
            [
                [[-0.4, 0.1, 0.0], [0.4, -0.1, 0.0]],
                [[0.1, -0.4, 0.0], [-0.1, 0.4, 0.0]],
            ]
        )
        probabilities = torch.tensor(
            [
                [[0.70, 0.20, 0.10], [0.10, 0.65, 0.25]],
                [[0.15, 0.25, 0.60], [0.55, 0.30, 0.15]],
            ],
            requires_grad=True,
        )
        generated_types = (
            probs_to_sphere(probabilities) if spherical_space else probabilities
        )
        real_types = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            ]
        )
        return gen_pos, real_pos, generated_types, real_types, probabilities

    def _assert_loss_works(self, spherical_space):
        inputs = self._inputs(spherical_space)
        gen_pos, real_pos, generated_types, real_types, probabilities = inputs

        loss, stats = compute_drift_loss(
            gen_pos,
            real_pos,
            generated_types,
            real_types,
            num_atoms=2,
            chem_refinement=False,
            cfg=self._loss_config(spherical_space=spherical_space),
        )

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(gen_pos.grad)
        self.assertIsNotNone(probabilities.grad)
        self.assertTrue(torch.isfinite(gen_pos.grad).all())
        self.assertTrue(torch.isfinite(probabilities.grad).all())
        return stats

    def test_spherical_loss_remains_the_default_for_legacy_configs(self):
        gen_pos, real_pos, generated_types, real_types, _ = self._inputs(True)
        _, stats = compute_drift_loss(
            gen_pos,
            real_pos,
            generated_types,
            real_types,
            num_atoms=2,
            chem_refinement=False,
            cfg=self._loss_config(),
        )

        self.assertIn("mean_spherical_distance", stats)

    def test_spherical_loss_is_finite_and_differentiable(self):
        stats = self._assert_loss_works(spherical_space=True)
        self.assertIn("mean_spherical_distance", stats)
        self.assertNotIn("mean_euclidean_types_distance", stats)

    def test_euclidean_probability_loss_is_finite_and_differentiable(self):
        stats = self._assert_loss_works(spherical_space=False)
        self.assertIn("mean_euclidean_types_distance", stats)
        self.assertNotIn("mean_spherical_distance", stats)

    def test_model_space_conversion_respects_configuration(self):
        probabilities = torch.tensor([[0.70, 0.20, 0.10]])

        spherical_model = MoleculeGenerator(
            generator_cfg={"hidden_nf": 8, "n_layers": 1, "num_atom_types": 3}
        )
        euclidean_model = MoleculeGenerator(
            generator_cfg={"hidden_nf": 8, "n_layers": 1, "num_atom_types": 3},
            drift_cfg={"spherical_space": False, "num_atom_types": 3},
        )

        spherical_types = spherical_model._types_to_model_space(probabilities)
        euclidean_types = euclidean_model._types_to_model_space(probabilities)

        torch.testing.assert_close(spherical_types.norm(dim=-1), torch.ones(1))
        torch.testing.assert_close(euclidean_types, probabilities)
        torch.testing.assert_close(
            spherical_model._types_to_probabilities(spherical_types), probabilities
        )


if __name__ == "__main__":
    unittest.main()
