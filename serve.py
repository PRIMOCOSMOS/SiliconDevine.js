"""Serve the already-built offline demo, without Node dependencies."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse

parser=argparse.ArgumentParser(description='SiliconDevine.js demo')
parser.add_argument('--port',type=int,default=5181)
args=parser.parse_args()
root=Path(__file__).resolve().parent/'demo-dist'
if not (root/'index.html').exists():
    raise SystemExit('Build missing: run npm install and npm run build first.')
print(f'SiliconDevine.js: http://127.0.0.1:{args.port}',flush=True)
with ThreadingHTTPServer(('127.0.0.1',args.port),partial(SimpleHTTPRequestHandler,directory=str(root))) as server:
    try: server.serve_forever()
    except KeyboardInterrupt: pass
