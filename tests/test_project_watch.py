"""End-to-end multi-file hot reload, including invalid JSON recovery."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


class ProjectWatchTests(unittest.TestCase):
    def test_dependency_and_config_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / 'project_model'
            shutil.copytree(ROOT / 'examples/project_model', project)
            config = project / 'silicondevine.json'
            config.write_text('{}')
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            env = dict(os.environ, PYTHONPATH=str(ROOT / 'python'), PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
            with (Path(directory) / 'watch.log').open('w', encoding='utf-8') as logs:
                proc = subprocess.Popen([sys.executable, '-m', 'silicondevine.watch', str(project), '--config', str(config), '--settings', str(Path(directory) / 'profile.json'), '--port', str(port), '--static-dir', str(ROOT / 'demo-dist')], env=env, stdout=logs, stderr=logs, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                def get(endpoint):
                    with urlopen(f'http://127.0.0.1:{port}/api/{endpoint}', timeout=2) as response:
                        return json.load(response)
                def wait(predicate):
                    deadline = time.monotonic() + 90
                    last = None
                    while time.monotonic() < deadline:
                        try:
                            last = get('status')
                            if predicate(last):
                                return last
                        except OSError:
                            pass
                        time.sleep(.3)
                    self.fail(f'watch timeout: {last}; {(Path(directory) / "watch.log").read_text(encoding="utf-8")}')
                try:
                    wait(lambda s: s['revision'] >= 1)
                    first = get('model')['model']
                    self.assertTrue(first['discovery']['entry'].endswith('network:VisionModel'))
                    helper = project / 'blocks.py'
                    helper.write_text(helper.read_text().replace('return x + self.layers(x)', 'return x + 2 * self.layers(x)'))
                    wait(lambda s: s['revision'] >= 2)
                    second = get('model')['model']
                    self.assertGreater(len(second['nodes']), len(first['nodes']))
                    config = project / 'silicondevine.json'
                    config.write_text('{broken')
                    wait(lambda s: bool(s['error']) and not s['loading'])
                    self.assertEqual(get('model')['revision'], 2)
                    config.write_text(json.dumps({'init': {'channels': 4, 'classes': 5}}))
                    wait(lambda s: s['revision'] >= 3)
                    third = get('model')['model']
                    self.assertEqual(next(t['shape'] for t in third['tensors'] if t['id'] == third['outputs'][0]), [1, 5])
                finally:
                    if os.name == 'nt':
                        subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                    if proc.poll() is None:
                        proc.terminate()
                    proc.wait(timeout=10)


if __name__ == '__main__':
    unittest.main()
