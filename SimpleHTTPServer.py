#!/usr/bin/env python3

from http.server import HTTPServer, CGIHTTPRequestHandler

port = 8030
url =  "localhost"
httpd = HTTPServer((url, port), CGIHTTPRequestHandler)
print("Starting simple_httpd on port: " + str(httpd.server_port))
httpd.serve_forever()

