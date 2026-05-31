"""
OCR Microservice — Chạy PaddleOCR trên CPU trong tiến trình riêng biệt.

Tiến trình này được spawn bởi app.py khi bắt đầu stream video.
Nó hoàn toàn cách ly khỏi PyTorch/CUDA của luồng chính, tránh
xung đột DLL và bộ nhớ GPU.

Tối ưu hiệu năng:
  - cpu_threads trong config OCR giới hạn số luồng CPU của PaddleOCR.
  - ThreadingHTTPServer: xử lý nhiều request đồng thời khi phát hiện
    nhiều biển số trong cùng frame.
"""

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2
import numpy as np

from config.settings import get_settings
from rlvds.ocr.preprocessor import prepare_paddle_ocr_input

settings = get_settings()
ocr_settings = settings.ocr

# Force CPU-only for PaddleOCR to avoid GPU/CUDA conflicts
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("FLAGS_use_mkldnn", "0")

thread_count = str(ocr_settings.cpu_threads)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", thread_count)
os.environ.setdefault("OPENBLAS_NUM_THREADS", thread_count)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", thread_count)
os.environ.setdefault("NUMEXPR_NUM_THREADS", thread_count)

from paddleocr import PaddleOCR

print("Initializing PaddleOCR engine on CPU...")
ocr_kwargs = {
    "lang": ocr_settings.lang,
    "use_gpu": False,
    "show_log": False,
    "use_angle_cls": ocr_settings.use_angle_cls,
    "enable_mkldnn": ocr_settings.enable_mkldnn,
    "cpu_threads": ocr_settings.cpu_threads,
}
if ocr_settings.det_model_dir:
    ocr_kwargs["det_model_dir"] = ocr_settings.det_model_dir
if ocr_settings.rec_model_dir:
    ocr_kwargs["rec_model_dir"] = ocr_settings.rec_model_dir
ocr_engine = PaddleOCR(**ocr_kwargs)
print("PaddleOCR engine loaded successfully!")


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads for concurrent plate OCR."""
    daemon_threads = True


class OCRRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Read image from request body
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)

        # Decode image
        nparr = np.frombuffer(post_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None or img.size == 0:
            self._send_response(None)
            return

        img = prepare_paddle_ocr_input(img)

        try:
            result = ocr_engine.ocr(img, cls=False)
            self._send_response(result)
        except Exception as e:
            print(f"OCR Inference error: {e}")
            self._send_response(None)

    def _send_response(self, result):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response = json.dumps({"raw_result": result})
        self.wfile.write(response.encode("utf-8"))

    def log_message(self, format, *args):
        # Suppress logging request info to keep console clean
        return


def run(port=8502):
    server_address = ('127.0.0.1', port)
    httpd = ThreadingHTTPServer(server_address, OCRRequestHandler)
    print(f"OCR Service running at http://127.0.0.1:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down OCR Service...")
        httpd.server_close()


if __name__ == '__main__':
    run()
