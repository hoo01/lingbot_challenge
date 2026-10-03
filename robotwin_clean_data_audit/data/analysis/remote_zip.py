"""Seekable read-only file over HTTP Range requests (honors HTTP(S)_PROXY via requests)."""
import io, requests
from huggingface_hub import hf_hub_url

REPO = "TianxingChen/RoboTwin2.0"

class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, block=4 << 20):
        self.s = requests.Session()
        r = self.s.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.url = r.url
        self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.block = block
        self.cache = {}  # block index -> bytes
        self.bytes_fetched = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _get_block(self, i):
        if i not in self.cache:
            a = i * self.block
            b = min(self.size, a + self.block) - 1
            for _ in range(5):
                try:
                    r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
                    r.raise_for_status()
                    break
                except requests.RequestException:
                    continue
            else:
                raise IOError(f"range fetch failed {a}-{b}")
            self.cache[i] = r.content
            self.bytes_fetched += len(r.content)
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            i, o = divmod(self.pos, self.block)
            chunk = self._get_block(i)[o:o + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)

def open_remote(path_in_repo, **kw):
    return HTTPRangeFile(hf_hub_url(REPO, path_in_repo, repo_type="dataset"), **kw)
