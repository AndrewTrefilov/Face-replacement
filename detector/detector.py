#!/usr/bin/env python

import sys
import json
import http.server
import socketserver
import numpy as np
import logging
import time
import os
from retina_face.detector import FaceDetector


logging.basicConfig(level=logging.INFO)
os.system("/sbin/ldconfig")

PORT = 9000


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class HTTPHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_len = int(self.headers.get("Content-Length"))
        BATCH_SIZE = int(self.headers.get("BATCH_SIZE"))
        try:
            in_bytes = self.rfile.read(content_len)
        except:
            self.send_error(422)
            self.end_headers()
            return
        width, height = 1280, 720
        img = np.frombuffer(in_bytes, np.uint8).reshape([BATCH_SIZE, height, width, 3])
        dets, landms = detector.return_bboxes(img)

        self.send_response(200)
        self.send_header("content-type", "text/html")
        self.end_headers()
        if dets.size == 0:
            dets = [None] * BATCH_SIZE
            response = json.dumps(dets)
        else:
            response = json.dumps(dets, cls=NumpyEncoder)
        self.wfile.write(response.encode("utf-8"))

    def do_GET(self):
        self.send_response(200)
        self.send_header("content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"hello !")


if len(sys.argv) > 1:
    PORT = int(sys.argv[1])

PATH_TO_WEGHT = "../retina_face/checkpoint/shufflenet_v2_fp16.pth"
detector = FaceDetector(
    det_threshold=0.51,
    checkpoint_path=PATH_TO_WEGHT,
    device=f"cuda:{sys.argv[2]}",
    top_k=100,
    nms_threshold=0.2,
)

with socketserver.TCPServer(("", PORT), HTTPHandler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()
