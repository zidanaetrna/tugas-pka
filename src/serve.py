"""Entrypoint container: menjalankan eksperimen (bila belum ada hasil), lalu menyajikan hasil di port 8080."""
import functools
import http.server
import os
import subprocess
import sys
import threading
import time

import json

from laporan_html import write_html, write_status

OUT = os.environ.get("OUT_DIR", "/app/hasil")
PORT = int(os.environ.get("PORT", "8080"))
FORCE = os.environ.get("RERUN", "0") == "1"
EXTRA = os.environ.get("EKSPERIMEN_ARGS", "").split()


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve():
    handler = functools.partial(QuietHandler, directory=OUT)
    http.server.ThreadingHTTPServer(("0.0.0.0", PORT), handler).serve_forever()


def main():
    os.makedirs(OUT, exist_ok=True)
    done = os.path.exists(os.path.join(OUT, "hasil.json"))
    if not done or FORCE:
        write_status(OUT)
    threading.Thread(target=serve, daemon=True).start()
    print(f"Hasil dapat dibuka di http://localhost:{PORT}", flush=True)

    if done and not FORCE:
        print("Hasil sebelumnya ditemukan di folder hasil/. Set RERUN=1 untuk menjalankan ulang.", flush=True)
    else:
        proc = subprocess.Popen([sys.executable, "/app/src/eksperimen.py",
                                 "--data", "/app/data/diabetes.csv", "--out", OUT, *EXTRA])
        while proc.poll() is None:
            write_status(OUT)
            time.sleep(5)
        if proc.returncode != 0:
            print("Eksperimen gagal, lihat log di atas.", flush=True)
            sys.exit(proc.returncode)
        with open(os.path.join(OUT, "hasil.json"), encoding="utf-8") as f:
            write_html(json.load(f), OUT)  # pastikan halaman akhir tidak tertimpa halaman status
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
