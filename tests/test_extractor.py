"""Tests for Instagram Reel Extractor"""

import unittest
from src.downloader import ReelDownloader
from src.metadata import MetadataExtractor


class TestReelDownloader(unittest.TestCase):
    def setUp(self):
        self.downloader = ReelDownloader()
    
    def test_extract_reel_id_standard_url(self):
        url = "https://www.instagram.com/reel/ABC123/"
        self.assertEqual(self.downloader.extract_reel_id(url), "ABC123")
    
    def test_extract_reel_id_reels_plural(self):
        url = "https://www.instagram.com/reels/ABC123/"
        self.assertEqual(self.downloader.extract_reel_id(url), "ABC123")


class TestMetadataExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = MetadataExtractor()
    
    def test_extract_hashtags(self):
        text = "Amazing sunset! #sunset #nature"
        hashtags = self.extractor._extract_hashtags(text)
        self.assertEqual(hashtags, ["#sunset", "#nature"])


if __name__ == '__main__':
    unittest.main()