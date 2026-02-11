import urllib.request
import os
import ssl

# SSL verification skip for download
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = "https://thispersondoesnotexist.com"
out = os.path.join(os.path.dirname(__file__), "assets", "test_face.jpg")

req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, context=ctx) as resp:
    data = resp.read()
    with open(out, "wb") as f:
        f.write(data)

print(f"Downloaded: {len(data)} bytes -> {out}")
