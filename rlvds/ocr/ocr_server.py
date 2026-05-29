"""
OCR Microservice — Chạy PaddleOCR trên CPU trong tiến trình riêng biệt.

Tiến trình này được spawn bởi app.py khi bắt đầu stream video.
Nó hoàn toàn cách ly khỏi PyTorch/CUDA của luồng chính, tránh
xung đột DLL và bộ nhớ GPU.

Tối ưu hiệu năng:
  - OMP_NUM_THREADS=2: cho phép PaddlePaddle dùng 2 luồng CPU cho
    inference nhanh hơn mà không ảnh hưởng luồng chính (chạy GPU).
  - ThreadingHTTPServer: xử lý nhiều request đồng thời khi phát hiện
    nhiều biển số trong cùng frame.
"""

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import numpy as np
import cv2

from rlvds.ocr.preprocessor import prepare_paddle_ocr_input

# Force CPU-only for PaddleOCR to avoid GPU/CUDA conflicts
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Cho phép PaddlePaddle dùng 2 luồng CPU — đủ nhanh cho inference
# mà không ảnh hưởng luồng chính (chạy trên GPU riêng biệt).
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

from paddleocr import PaddleOCR

print("Initializing PaddleOCR engine on CPU...")
ocr_engine = PaddleOCR(use_gpu=False, show_log=False)
print("PaddleOCR engine loaded successfully!")


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads for concurrent plate OCR."""
    daemon_threads = True


class OCRRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Read image from request body
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        # Decode image
        nparr = np.frombuffer(post_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None or img.size == 0:
            self._send_response(None)
            return

        img = prepare_paddle_ocr_input(img)

        try:
            result = ocr_engine.ocr(img)
            self._send_response(result)
        except Exception as e:
            print(f"OCR Inference error: {e}")
            self._send_response(None)

    def _send_response(self, result):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        response = json.dumps({'raw_result': result})
        self.wfile.write(response.encode('utf-8'))

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
