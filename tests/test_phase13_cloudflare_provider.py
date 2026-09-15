import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import json
import os
import sys

# Ensure d:\mvid is in sys.path
sys.path.insert(0, r"d:\mvid")

from autonomous.providers.cloudflare_workers_ai_provider import CloudflareWorkersAIProvider

class TestCloudflareProvider(unittest.TestCase):
    def setUp(self):
        self.usage_file = Path("data/cloudflare_usage.json")
        if self.usage_file.exists():
            self.usage_file.unlink()

    def tearDown(self):
        if self.usage_file.exists():
            self.usage_file.unlink()
            
    @patch('os.environ.get')
    def test_missing_credentials(self, mock_env):
        mock_env.return_value = None
        provider = CloudflareWorkersAIProvider()
        self.assertFalse(provider.is_available())
        
    @patch('os.environ.get')
    def test_availability_with_creds(self, mock_env):
        def env_side_effect(k, default=None):
            if k == "CLOUDFLARE_ACCOUNT_ID": return "test_acc"
            if k == "CLOUDFLARE_API_TOKEN": return "test_tok"
            return default
        mock_env.side_effect = env_side_effect
        
        provider = CloudflareWorkersAIProvider()
        self.assertTrue(provider.is_available())

    @patch('os.environ.get')
    def test_usage_limit(self, mock_env):
        def env_side_effect(k, default=None):
            if k == "CLOUDFLARE_ACCOUNT_ID": return "test_acc"
            if k == "CLOUDFLARE_API_TOKEN": return "test_tok"
            return default
        mock_env.side_effect = env_side_effect
        
        provider = CloudflareWorkersAIProvider()
        
        provider.usage_file.parent.mkdir(parents=True, exist_ok=True)
        from datetime import date
        with open(provider.usage_file, "w") as f:
            json.dump({
                "date": date.today().isoformat(),
                "local_estimated_neurons": 8000
            }, f)
            
        self.assertFalse(provider.is_available())
        
        res = provider.generate("test", "", 1280, 720, 1, Path("test.png"))
        self.assertFalse(res["success"])
        self.assertEqual(res["error"], "FREE_LIMIT_REACHED")
        
    @patch('requests.post')
    @patch('os.environ.get')
    def test_paid_plan_error(self, mock_env, mock_post):
        def env_side_effect(k, default=None):
            if k == "CLOUDFLARE_ACCOUNT_ID": return "test_acc"
            if k == "CLOUDFLARE_API_TOKEN": return "test_tok"
            return default
        mock_env.side_effect = env_side_effect
        
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Workers Paid plan required for this request"
        mock_post.return_value = mock_resp
        
        provider = CloudflareWorkersAIProvider()
        res = provider.generate("test", "", 1280, 720, 1, Path("test.png"))
        self.assertFalse(res["success"])
        self.assertEqual(res["error"], "PAID_PLAN_REQUIRED")

if __name__ == '__main__':
    unittest.main()
