from __future__ import annotations

import inspect
import unittest
from unittest.mock import Mock, patch

import numpy as np

from sdde_model import solar_dynamo
from sdde_model import solar_dynamo_jupiter


ORIGINAL_THETA = (2.0, 1.0, 4.0, 0.1, 5.0)
JUPITER_THETA = ORIGINAL_THETA + (0.2, 0.3)


class OriginalBridgeTests(unittest.TestCase):
    def test_original_api_has_not_gained_a_model_argument(self):
        self.assertNotIn("model", inspect.signature(solar_dynamo.sn).parameters)
        self.assertNotIn("model", inspect.signature(solar_dynamo.sn_batch).parameters)
        self.assertNotIn("model", inspect.signature(solar_dynamo.sn_from_noise).parameters)
        self.assertNotIn("model", inspect.signature(solar_dynamo.sn_from_noise_batch).parameters)
        self.assertNotIn("model", inspect.signature(solar_dynamo.sn_nrep).parameters)

    def test_explicit_noise_wrapper_forwards_bare_increments(self):
        fake_julia = Mock()
        fake_julia.sn_from_noise.return_value = [1.0, 2.0]
        eps = np.arange(20, dtype=np.float64)

        with (
            patch.object(solar_dynamo, "_init_julia"),
            patch.object(solar_dynamo, "jl", fake_julia),
        ):
            result = solar_dynamo.sn_from_noise(
                ORIGINAL_THETA,
                eps,
                Twarmup=1,
                Tobs=1,
                dt=0.1,
                saveat=1.0,
            )

        self.assertEqual(result, [1.0, 2.0])
        fake_julia.sn_from_noise.assert_called_once_with(
            ORIGINAL_THETA,
            eps,
            Twarmup=1,
            Tobs=1,
            dt=0.1,
            saveat=1.0,
        )

    def test_explicit_noise_batch_validates_shapes_before_julia(self):
        theta_batch = np.tile(ORIGINAL_THETA, (2, 1))
        with patch.object(solar_dynamo, "_init_julia") as initialize:
            with self.assertRaisesRegex(ValueError, "eps_batch must have shape"):
                solar_dynamo.sn_from_noise_batch(
                    theta_batch,
                    np.ones((2, 19)),
                    Twarmup=1,
                    Tobs=1,
                    dt=0.1,
                )
        initialize.assert_not_called()


class JupiterValidationTests(unittest.TestCase):
    def test_requires_seven_parameters(self):
        with self.assertRaisesRegex(ValueError, "requires 7 parameters"):
            solar_dynamo_jupiter._validate_theta(ORIGINAL_THETA)

    def test_rejects_amplitude_outside_allowed_range(self):
        for amplitude in (-0.1, 1.0, 1.2):
            with self.subTest(amplitude=amplitude):
                with self.assertRaisesRegex(ValueError, "0 <= epsilon < 1"):
                    solar_dynamo_jupiter._validate_theta(
                        ORIGINAL_THETA + (amplitude, 0.0)
                    )

    def test_accepts_valid_parameters(self):
        self.assertEqual(
            solar_dynamo_jupiter._validate_theta(JUPITER_THETA),
            JUPITER_THETA,
        )


class JupiterDispatchTests(unittest.TestCase):
    def test_scalar_wrapper_calls_only_jupiter_function(self):
        fake_julia = Mock()
        fake_julia.sn_jupiter_nd.return_value = [1.0, 2.0]

        with (
            patch.object(solar_dynamo_jupiter, "_init_julia"),
            patch.object(solar_dynamo_jupiter, "jl", fake_julia),
        ):
            result = solar_dynamo_jupiter.sn(
                JUPITER_THETA,
                Twarmup=10,
                Tobs=2,
                dt=0.1,
                saveat=1.0,
                seed=123,
            )

        self.assertEqual(result, [1.0, 2.0])
        fake_julia.sn_jupiter_nd.assert_called_once_with(
            JUPITER_THETA,
            Twarmup=10,
            Tobs=2,
            dt=0.1,
            saveat=1.0,
            seed=123,
        )

    def test_batch_wrapper_validates_shape_before_julia_initialization(self):
        with patch.object(solar_dynamo_jupiter, "_init_julia") as initialize:
            with self.assertRaisesRegex(ValueError, r"shape \(n_batch, 7\)"):
                solar_dynamo_jupiter.sn_batch(np.ones((2, 5)))
        initialize.assert_not_called()

    def test_explicit_noise_wrapper_forwards_bare_increments(self):
        fake_julia = Mock()
        fake_julia.sn_from_noise_jupiter_nd.return_value = [3.0, 4.0]
        eps = np.arange(20, dtype=np.float64)

        with (
            patch.object(solar_dynamo_jupiter, "_init_julia"),
            patch.object(solar_dynamo_jupiter, "jl", fake_julia),
        ):
            result = solar_dynamo_jupiter.sn_from_noise(
                JUPITER_THETA,
                eps,
                Twarmup=1,
                Tobs=1,
                dt=0.1,
                saveat=1.0,
            )

        self.assertEqual(result, [3.0, 4.0])
        fake_julia.sn_from_noise_jupiter_nd.assert_called_once_with(
            JUPITER_THETA,
            eps,
            Twarmup=1,
            Tobs=1,
            dt=0.1,
            saveat=1.0,
        )

    def test_explicit_noise_batch_validates_shapes_before_julia(self):
        theta_batch = np.tile(JUPITER_THETA, (2, 1))
        with patch.object(solar_dynamo_jupiter, "_init_julia") as initialize:
            with self.assertRaisesRegex(ValueError, "eps_batch must have shape"):
                solar_dynamo_jupiter.sn_from_noise_batch(
                    theta_batch,
                    np.ones((2, 19)),
                    Twarmup=1,
                    Tobs=1,
                    dt=0.1,
                )
        initialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
