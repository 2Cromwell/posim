"""
POSIM 舆情仿真可视化系统 - 本地服务器

Usage:
    python serve.py [port]

默认端口: 8765
启动后浏览器访问: http://localhost:8765/results_vis/
"""
import http.server
import socketserver
import os
import sys
import webbrowser
import urllib.parse
import json
import hashlib
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(__file__).resolve().parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)


class PosimHandler(http.server.SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SCRIPTS_DIR), **kwargs)

    def translate_path(self, path):
        parsed = urllib.parse.urlparse(path)
        decoded = urllib.parse.unquote(parsed.path)
        decoded = decoded.lstrip('/')
        fs_path = Path(self.directory) / decoded
        return str(fs_path)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        decoded = urllib.parse.unquote(parsed.path)

        if decoded == '/results_vis/api/cache':
            self.handle_cache_save()
            return

        self.send_error(404, "Not found")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        decoded = urllib.parse.unquote(parsed.path)

        if decoded.startswith('/results_vis/api/cache/'):
            self.handle_cache_load(decoded)
            return

        fs_path = Path(self.directory) / decoded.lstrip('/')

        if fs_path.is_dir() and not decoded.endswith('/'):
            self.send_response(301)
            new_url = urllib.parse.quote(decoded) + '/'
            self.send_header('Location', new_url)
            self.end_headers()
            return

        if fs_path.is_dir():
            if (fs_path / 'index.html').exists():
                self.path = decoded.rstrip('/') + '/index.html'
                self.send_file(Path(self.directory) / decoded.lstrip('/').rstrip('/') / 'index.html')
                return
            self.send_directory_listing(fs_path, decoded)
            return

        if fs_path.exists():
            self.send_file(fs_path)
        else:
            self.send_error(404, f"File not found: {decoded}")

    def handle_cache_save(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            data = json.loads(body)
            cache_key = data.get('key', '')
            cache_data = data.get('data', {})

            safe_key = hashlib.md5(cache_key.encode('utf-8')).hexdigest()
            cache_file = CACHE_DIR / f'{safe_key}.json'

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({'key': cache_key, 'data': cache_data}, f, ensure_ascii=False)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'ok', 'file': str(cache_file)}).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    def handle_cache_load(self, path):
        cache_key = path.replace('/results_vis/api/cache/', '')
        cache_key = urllib.parse.unquote(cache_key)
        safe_key = hashlib.md5(cache_key.encode('utf-8')).hexdigest()
        cache_file = CACHE_DIR / f'{safe_key}.json'

        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(data['data'], ensure_ascii=False).encode('utf-8'))
            except Exception:
                self.send_error(500, "Cache read error")
        else:
            self.send_error(404, "Cache miss")

    def send_file(self, file_path):
        try:
            content_type = self.guess_type(str(file_path))
            with open(file_path, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', len(data))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(500, str(e))

    def send_directory_listing(self, dir_path, url_path):
        try:
            entries = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        except PermissionError:
            self.send_error(403, "Permission denied")
            return

        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()

        html = f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>Index of {url_path}</title></head><body>'
        html += f'<h1>Index of {url_path}</h1><ul>'
        if url_path != '/':
            html += '<li><a href="../">..</a></li>'
        for entry in entries:
            name = entry.name + ('/' if entry.is_dir() else '')
            encoded = urllib.parse.quote(name)
            html += f'<li><a href="{encoded}">{name}</a></li>'
        html += '</ul></body></html>'
        self.wfile.write(html.encode('utf-8'))

    def guess_type(self, path):
        ext = os.path.splitext(str(path))[1].lower()
        mime_map = {
            '.json': 'application/json',
            '.html': 'text/html; charset=utf-8',
            '.js': 'application/javascript; charset=utf-8',
            '.css': 'text/css; charset=utf-8',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.svg': 'image/svg+xml',
        }
        return mime_map.get(ext, super().guess_type(path))

    def log_message(self, format, *args):
        msg = format % args
        sys.stderr.write(f"  {msg}\n")


def main():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), PosimHandler) as httpd:
        url = f"http://localhost:{PORT}/results_vis/"
        print(f"\n{'='*60}")
        print(f"  POSIM 舆情仿真可视化系统")
        print(f"  服务已启动: {url}")
        print(f"  缓存目录: {CACHE_DIR}")
        print(f"  按 Ctrl+C 停止服务")
        print(f"{'='*60}\n")

        try:
            webbrowser.open(url)
        except Exception:
            pass

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n服务已停止")


if __name__ == '__main__':
    main()
