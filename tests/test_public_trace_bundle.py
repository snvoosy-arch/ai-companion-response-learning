import json
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / 'docs/diffusion-trace'


class Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if (tag == 'script' and key == 'src') or (tag == 'link' and key == 'href'):
                self.paths.append(value)


class PublicBundleTests(unittest.TestCase):
    def test_html_local_assets_exist_and_data_script_matches_json(self):
        p = Assets()
        p.feed((VIEWER/'index.html').read_text(encoding='utf8'))
        for name in p.paths:
            self.assertNotIn('://', name)
            self.assertTrue((VIEWER/name).is_file(), name)
        text = (VIEWER/'data/traces.js').read_text(encoding='utf8')
        self.assertTrue(text.startswith('window.RUBRUM_TRACE_DATA = '))
        data = json.loads((VIEWER/'data/traces.json').read_text(encoding='utf8'))
        self.assertEqual(json.loads(text.removeprefix('window.RUBRUM_TRACE_DATA = ').rstrip(' \n;')), data)
        self.assertEqual([len(c['steps'])-1 for c in data['cases']], [14,22])

    def test_export_has_only_public_fields_and_no_private_metadata(self):
        data = json.loads((VIEWER/'data/traces.json').read_text(encoding='utf8'))
        allowed = {'id','title','verdict','label','note','user','context','canvas_length','stop_ids',
                   'vocabulary','steps','final','source'}
        for c in data['cases']:
            self.assertEqual(set(c), allowed)
            self.assertEqual(set(c['source']), {'record_sha256','trace_gzip_sha256'})
            for value in c['source'].values(): self.assertRegex(value, r'^[a-f0-9]{64}$')
            self.assertEqual(set(c['steps'][0]), {'step','ids','changed','text','text32','prefix_text','mean_entropy','eos_index','stop_reason'})
            for s in c['steps'][1:]: self.assertEqual(set(s), set(c['steps'][0]) | {'prediction_eos_index'})
        for p in [*VIEWER.rglob('*'), ROOT/'docs/rubrum-current-architecture.md', ROOT/'evidence/model-training/current-summary.json']:
            if p.is_file():
                text=p.read_text(encoding='utf8')
                self.assertIsNone(re.search(r'[A-Z]:[/\\](?:bot|Users)[/\\]',text,re.I),str(p))
                self.assertNotIn('discord_token',text)

    def test_ui_inserts_model_strings_as_text_and_has_no_model_endpoint(self):
        script=(VIEWER/'app.js').read_text(encoding='utf8')
        self.assertNotIn('innerHTML',script)
        self.assertNotIn('fetch(',script)
        self.assertNotIn('XMLHttpRequest',script)


if __name__ == '__main__': unittest.main()
