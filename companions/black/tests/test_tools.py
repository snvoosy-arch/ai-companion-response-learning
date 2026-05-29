from __future__ import annotations

import unittest

from predictive_bot.core.tools import OpenMeteoWeatherService


class WeatherToolTests(unittest.TestCase):
    def test_normalize_location_query_maps_korean_aliases(self) -> None:
        self.assertEqual(OpenMeteoWeatherService._normalize_location_query("서울"), "Seoul")
        self.assertEqual(OpenMeteoWeatherService._normalize_location_query("서울특별시"), "Seoul")
        self.assertEqual(OpenMeteoWeatherService._normalize_location_query("부산"), "Busan")

    def test_normalize_location_query_keeps_unknown_names(self) -> None:
        self.assertEqual(OpenMeteoWeatherService._normalize_location_query("춘천"), "춘천")
        self.assertEqual(OpenMeteoWeatherService._normalize_location_query("Seoul"), "Seoul")


if __name__ == "__main__":
    unittest.main()
