import unittest

from env.racerenv import RacerEnv


class TestRacerEnv(unittest.TestCase):
    def test_denormalize(self):
        environment = RacerEnv()

        # mean value
        self.assertEqual(
            environment.denormalize(0, 20, 40), 30, "Should be mean value."
        )
        self.assertEqual(
            environment.denormalize(0, -100, 100), 0, "Should be mean value."
        )
        self.assertEqual(
            environment.denormalize(0, -60, 20), -20, "Should be mean value."
        )

        # min value
        self.assertEqual(
            environment.denormalize(-1, 20, 40), 20, "Should be min value."
        )
        self.assertEqual(
            environment.denormalize(-1, -100, 100), -100, "Should be min value."
        )
        self.assertEqual(
            environment.denormalize(-1, -60, 20), -60, "Should be min value."
        )

        # max value
        self.assertEqual(environment.denormalize(1, 20, 40), 40, "Should be max value.")
        self.assertEqual(
            environment.denormalize(1, -100, 100), 100, "Should be max value."
        )
        self.assertEqual(
            environment.denormalize(1, -60, 20), 20, "Should be max value."
        )

        # 75th percentile
        self.assertEqual(
            environment.denormalize(0.5, 20, 40), 35, "Should be 75th percentile."
        )
        self.assertEqual(
            environment.denormalize(0.5, -100, 100), 50, "Should be 75th percentile."
        )
        self.assertEqual(
            environment.denormalize(0.5, -60, 20), 0, "Should be 75th percentile."
        )


if __name__ == "__main__":
    unittest.main()
