#!/usr/bin/env python3
"""Serve this self-contained workshop on loopback and open the browser."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import webbrowser

ROOT = Path(__file__).resolve().parent
with ThreadingHTTPServer(('127.0.0.1', 0), partial(SimpleHTTPRequestHandler, directory=str(ROOT))) as server:
    url = f'http://127.0.0.1:{server.server_port}/'
    print(f'کارگاه آماده است: {url}', flush=True)
    print('برای پایان، این پنجره را ببندید یا Ctrl+C بزنید.', flush=True)
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nپایان کارگاه.')
