"""
Turbo-NAFS-128  --  an invertible, keyed instantiation of the Turbo-NAFS design
(Alshammari, "Design of Turbo-NAFS: A Quantum-Resilient Encryption Scheme Based
on Functional Superposition", SSRN 6167526, Jan 2026).

RESEARCH CODE.  This is a new, unanalysed cipher.  Passing the statistical tests
in verify.py is necessary but NOT sufficient for security.  Do not use it to
protect real data; use AES-256-GCM or ChaCha20-Poly1305 for that.

Mapping from the paper to this implementation
---------------------------------------------
Paper concept                       Here
----------------------------------  -----------------------------------------
Functional space F = {f1..fn}       8 keyed bijections on Z_2^64 (FUNCS)
Non-commutativity fi∘fj ≠ fj∘fi     measured empirically in verify.py
Hidden composition graph G          key-derived digraph on F, out-degree 4,
                                    no self loops
Ordered composition chains P^k      key-derived walks of length k on G
Entropy coefficients α_k            key-derived odd 64-bit multipliers
Superposition Ψ = Σ α_k P^k(C)      R ← R + Σ α_k · P^k(L)  (mod 2^64)
                                    -- a lifting/Feistel step, so it is
                                    invertible even though Σ is many-to-one
Turbo round T_r(C, K_r)             superposition → XOR subkey → S-box →
                                    key-dependent half-swapping byte shuffle
Entropy injected from prior state   the R update is a function of the L half
                                    of the previous state
Hash commitment H (public)          key-check value KCV = SHA-256 commitment
Turbo seeds S                       SHAKE-256 key schedule

Why the paper's prototype could not decrypt: it used f3(x) = x^2+9 mod 997,
which is 2-to-1, inside an ordinary composition chain, so information was
destroyed every time f3 ran.  Here every f in F is a bijection AND the
superposition is applied as an additive lifting step, so decryption only has to
re-compute Σ α_k P^k(L) and subtract it -- the paths never need inverting.

Two independent implementations are provided:
  * Scalar (pure Python ints)     -- encrypt_block / decrypt_block
  * Batch  (NumPy, many blocks)   -- encrypt_blocks / decrypt_blocks
verify.py checks that they agree bit-for-bit.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import struct
from dataclasses import dataclass, field

import numpy as np

VERSION = 1
BLOCK = 16                      # bytes (128-bit block)
KEY_BYTES = 32                  # 256-bit master key
M64 = (1 << 64) - 1
M32 = (1 << 32) - 1

# ---------------------------------------------------------------------------
# S-box (AES S-box: well-studied, max differential prob 2^-6, max linear bias 2^-4)
# ---------------------------------------------------------------------------
def _aes_sbox():
    sbox = [0] * 256
    p = q = 1
    while True:
        p = p ^ ((p << 1) & 0xFF) ^ (0x1B if p & 0x80 else 0)
        q ^= q << 1
        q ^= q << 2
        q ^= q << 4
        q &= 0xFF
        if q & 0x80:
            q ^= 0x09
        x = q ^ ((q << 1) | (q >> 7)) ^ ((q << 2) | (q >> 6)) ^ ((q << 3) | (q >> 5)) ^ ((q << 4) | (q >> 4))
        sbox[p] = (x ^ 0x63) & 0xFF
        if p == 1:
            break
    sbox[0] = 0x63
    return sbox

SBOX = _aes_sbox()
SBOX_INV = [0] * 256
for _i, _v in enumerate(SBOX):
    SBOX_INV[_v] = _i
SBOX_NP = np.array(SBOX, dtype=np.uint8)
SBOX_INV_NP = np.array(SBOX_INV, dtype=np.uint8)

# ---------------------------------------------------------------------------
# Functional space F: eight keyed bijections on 64-bit words
# ---------------------------------------------------------------------------
FUNC_NAMES = ["ADD", "XOR", "MUL", "ROT", "XSR", "SIG", "SUB", "ARX"]
N_FUNCS = len(FUNC_NAMES)
NONLINEAR = {2, 6, 7}           # MUL, SUB, ARX  (nonlinear over GF(2))
DOWNWARD = {3, 4, 5}            # ROT, XSR, SIG  (move high bits to low bits)


def _rotl64(x, r):
    return ((x << r) | (x >> (64 - r))) & M64


def _sig_amounts(p):
    a = 1 + (p % 63)
    b = 1 + ((p >> 8) % 63)
    if b == a:
        b = 1 + ((a + 31) % 63)
    return a, b


def f_scalar(idx: int, x: int, p: int) -> int:
    """Apply function F[idx] with key parameter p to a 64-bit int."""
    if idx == 0:                                    # ADD   x + p
        return (x + p) & M64
    if idx == 1:                                    # XOR   x ^ p
        return x ^ p
    if idx == 2:                                    # MUL   x * (p|1)  (odd ⇒ bijective)
        return (x * (p | 1)) & M64
    if idx == 3:                                    # ROT   rotl(x, 1..63) ^ p  (keyed so it does
        return _rotl64(x, 1 + p % 63) ^ p           # not commute with SIG or byte-rotations)
    if idx == 4:                                    # XSR   x ^ (x >> 13..47)
        return x ^ (x >> (13 + p % 35))
    if idx == 5:                                    # SIG   x ^ rotl(x,a) ^ rotl(x,b)
        a, b = _sig_amounts(p)                      # 3-term ⇒ invertible over GF(2)[t]/(t^64+1)
        return x ^ _rotl64(x, a) ^ _rotl64(x, b)
    if idx == 6:                                    # SUB   bytewise S-box
        y = 0
        for i in range(8):
            y |= SBOX[(x >> (8 * i)) & 0xFF] << (8 * i)
        return y
    if idx == 7:                                    # ARX   hi+=lo; lo=rotl32(lo,s)^hi
        hi, lo = x >> 32, x & M32
        hi = (hi + lo) & M32
        s = 1 + p % 31
        lo = (((lo << s) | (lo >> (32 - s))) & M32) ^ hi
        return (hi << 32) | lo
    raise ValueError(idx)


def f_inverse_scalar(idx: int, y: int, p: int) -> int:
    """Inverse of F[idx]; used only by the test-suite to PROVE each f is a bijection."""
    if idx == 0:
        return (y - p) & M64
    if idx == 1:
        return y ^ p
    if idx == 2:
        return (y * pow(p | 1, -1, 1 << 64)) & M64
    if idx == 3:
        r = 1 + p % 63
        y ^= p
        return ((y >> r) | (y << (64 - r))) & M64
    if idx == 4:
        s = 13 + p % 35
        x = y
        for _ in range(64 // s + 1):
            x = y ^ (x >> s)
        return x
    if idx == 5:
        # Solve x ^ rotl(x,a) ^ rotl(x,b) = y by Gaussian elimination over GF(2)
        a, b = _sig_amounts(p)
        rows = []
        for i in range(64):           # output bit i = x_i ^ x_{i-a} ^ x_{i-b}
            m =(1 << i) ^ (1 << ((i - a) % 64)) ^ (1 << ((i - b) % 64))
            rows.append([m, (y >> i) & 1])
        for col in range(64):
            piv = next(r for r in range(col, 64) if (rows[r][0] >> col) & 1)
            rows[col], rows[piv] = rows[piv], rows[col]
            for r in range(64):
                if r != col and (rows[r][0] >> col) & 1:
                    rows[r][0] ^= rows[col][0]
                    rows[r][1] ^= rows[col][1]
        return sum(rows[i][1] << i for i in range(64))
    if idx == 6:
        x = 0
        for i in range(8):
            x |= SBOX_INV[(y >> (8 * i)) & 0xFF] << (8 * i)
        return x
    if idx == 7:
        hi, lo = y >> 32, y & M32
        s = 1 + p % 31
        lo = lo ^ hi
        lo = ((lo >> s) | (lo << (32 - s))) & M32
        hi = (hi - lo) & M32
        return (hi << 32) | lo
    raise ValueError(idx)


def _u(v):
    return np.uint64(v)


def f_batch(idx: int, x: np.ndarray, p: int) -> np.ndarray:
    """Vectorised F[idx] on a uint64 array."""
    if idx == 0:
        return x + _u(p)
    if idx == 1:
        return x ^ _u(p)
    if idx == 2:
        return x * _u(p | 1)
    if idx == 3:
        r = 1 + p % 63
        return ((x << _u(r)) | (x >> _u(64 - r))) ^ _u(p)
    if idx == 4:
        return x ^ (x >> _u(13 + p % 35))
    if idx == 5:
        a, b = _sig_amounts(p)
        return x ^ ((x << _u(a)) | (x >> _u(64 - a))) ^ ((x << _u(b)) | (x >> _u(64 - b)))
    if idx == 6:
        by = np.ascontiguousarray(x).view(np.uint8)
        return SBOX_NP[by].view(np.uint64)
    if idx == 7:
        m32 = _u(M32)
        hi, lo = x >> _u(32), x & m32
        hi = (hi + lo) & m32
        s = 1 + p % 31
        lo = (((lo << _u(s)) | (lo >> _u(32 - s))) & m32) ^ hi
        return (hi << _u(32)) | lo
    raise ValueError(idx)


# ---------------------------------------------------------------------------
# Key schedule ("turbo seeds" + hidden composition graph)
# ---------------------------------------------------------------------------
class _Stream:
    """Deterministic byte stream from SHAKE-256 (the key-derivation 'entropy source H')."""

    def __init__(self, seed: bytes):
        self._buf = hashlib.shake_256(seed).digest(1 << 16)
        self._pos = 0

    def take(self, n):
        out = self._buf[self._pos:self._pos + n]
        if len(out) < n:
            raise RuntimeError("key schedule stream exhausted")
        self._pos += n
        return out

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def below(self, n):
        # unbiased integer in [0, n)
        lim = (1 << 64) - ((1 << 64) % n)
        while True:
            v = self.u64()
            if v < lim:
                return v % n

    def shuffle(self, seq):
        seq = list(seq)
        for i in range(len(seq) - 1, 0, -1):
            j = self.below(i + 1)
            seq[i], seq[j] = seq[j], seq[i]
        return seq


@dataclass
class Params:
    rounds: int = 16          # r : turbo rounds
    paths: int = 3            # number of superposed composition paths per round
    path_len: int = 4         # k : composition length
    out_degree: int = 4       # edges per node in the hidden composition graph G

    def tag(self) -> bytes:
        return struct.pack("<BBBBB", VERSION, self.rounds, self.paths, self.path_len, self.out_degree)


@dataclass
class RoundKey:
    paths: list            # list of [(func_idx, param), ...]   (P^k)
    alphas: list           # odd 64-bit coefficients             (α_k)
    whitening: bytes       # 16 bytes XORed before the S-box     (K_r)
    shuffle: list          # 16-entry byte permutation, swaps halves


@dataclass
class KeySchedule:
    params: Params
    graph: list            # adjacency list of hidden composition graph G
    pre_white: bytes
    post_white: bytes
    rounds: list = field(default_factory=list)
    kcv: str = ""          # public hash commitment to the key


def _walk(st: _Stream, graph, k):
    """Key-derived walk of length k on G, required to contain ≥1 nonlinear and
    ≥1 downward-diffusing function (otherwise a path could be purely linear)."""
    for _ in range(1000):
        node = st.below(N_FUNCS)
        walk = [node]
        for _ in range(k - 1):
            node = graph[node][st.below(len(graph[node]))]
            walk.append(node)
        if set(walk) & NONLINEAR and set(walk) & DOWNWARD:
            return walk
    raise RuntimeError("could not find admissible path; graph too sparse")


def expand_key(key: bytes, params: Params | None = None) -> KeySchedule:
    if len(key) != KEY_BYTES:
        raise ValueError("key must be 32 bytes")
    params = params or Params()
    st = _Stream(b"Turbo-NAFS/key-schedule" + params.tag() + key)

    # Hidden composition graph G: each node gets `out_degree` distinct successors, no self-loops
    graph = []
    for i in range(N_FUNCS):
        succ = st.shuffle([j for j in range(N_FUNCS) if j != i])[:params.out_degree]
        graph.append(sorted(succ))

    ks = KeySchedule(params=params, graph=graph,
                     pre_white=st.take(16), post_white=st.take(16))
    for _ in range(params.rounds):
        paths = []
        for _ in range(params.paths):
            walk = _walk(st, graph, params.path_len)
            paths.append([(f, st.u64()) for f in walk])
        alphas = [st.u64() | 1 for _ in range(params.paths)]
        white = st.take(16)
        lo = st.shuffle(range(8))         # new L bytes come from old R bytes
        hi = st.shuffle(range(8))         # new R bytes come from old L bytes
        shuffle = [8 + j for j in lo] + [j for j in hi]
        ks.rounds.append(RoundKey(paths, alphas, white, shuffle))

    ks.kcv = hashlib.sha256(b"Turbo-NAFS/kcv" + params.tag() + key).hexdigest()[:16]
    return ks


# ---------------------------------------------------------------------------
# Scalar reference implementation (one 16-byte block)
# ---------------------------------------------------------------------------
def _superpose_scalar(rk: RoundKey, L: int) -> int:
    acc = 0
    for alpha, path in zip(rk.alphas, rk.paths):
        v = L
        for f, p in path:
            v = f_scalar(f, v, p)
        acc = (acc + alpha * v) & M64
    return acc


def encrypt_block(ks: KeySchedule, block: bytes) -> bytes:
    s = bytes(a ^ b for a, b in zip(block, ks.pre_white))
    for rk in ks.rounds:
        L, R = struct.unpack("<QQ", s)
        R = (R + _superpose_scalar(rk, L)) & M64             # functional superposition
        s = struct.pack("<QQ", L, R)
        s = bytes(SBOX[a ^ b] for a, b in zip(s, rk.whitening))   # K_r + nonlinear layer
        s = bytes(s[j] for j in rk.shuffle)                    # turbo permutation
    return bytes(a ^ b for a, b in zip(s, ks.post_white))


def decrypt_block(ks: KeySchedule, block: bytes) -> bytes:
    s = bytes(a ^ b for a, b in zip(block, ks.post_white))
    for rk in reversed(ks.rounds):
        t = bytearray(16)
        for i, j in enumerate(rk.shuffle):
            t[j] = s[i]
        s = bytes(SBOX_INV[a] ^ b for a, b in zip(t, rk.whitening))
        L, R = struct.unpack("<QQ", s)
        R = (R - _superpose_scalar(rk, L)) & M64             # undo superposition
        s = struct.pack("<QQ", L, R)
    return bytes(a ^ b for a, b in zip(s, ks.pre_white))


# ---------------------------------------------------------------------------
# Batch implementation (NumPy): array of shape (N, 16) uint8
# ---------------------------------------------------------------------------
class BatchCipher:
    def __init__(self, ks: KeySchedule):
        self.ks = ks
        self.pre = np.frombuffer(ks.pre_white, np.uint8)
        self.post = np.frombuffer(ks.post_white, np.uint8)
        self.white = [np.frombuffer(rk.whitening, np.uint8) for rk in ks.rounds]
        self.perm = [np.array(rk.shuffle) for rk in ks.rounds]
        self.inv_perm = [np.argsort(p) for p in self.perm]

    def _superpose(self, rk: RoundKey, L):
        acc = np.zeros_like(L)
        for alpha, path in zip(rk.alphas, rk.paths):
            v = L
            for f, p in path:
                v = f_batch(f, v, p)
            acc = acc + _u(alpha) * v
        return acc

    def encrypt(self, blocks: np.ndarray, rounds: int | None = None) -> np.ndarray:
        """Encrypt (N,16) uint8.  `rounds` < full is for diffusion experiments only
        (the post-whitening is still applied)."""
        r = len(self.ks.rounds) if rounds is None else rounds
        s = np.ascontiguousarray(blocks ^ self.pre)
        with np.errstate(over="ignore"):
            for i in range(r):
                rk = self.ks.rounds[i]
                w = s.view(np.uint64).reshape(-1, 2)
                L, R = w[:, 0].copy(), w[:, 1] + self._superpose(rk, w[:, 0])
                s = np.stack([L, R], axis=1).view(np.uint8).reshape(-1, 16)
                s = SBOX_NP[s ^ self.white[i]]
                s = np.ascontiguousarray(s[:, self.perm[i]])
        return s ^ self.post

    def decrypt(self, blocks: np.ndarray) -> np.ndarray:
        s = np.ascontiguousarray(blocks ^ self.post)
        with np.errstate(over="ignore"):
            for i in reversed(range(len(self.ks.rounds))):
                rk = self.ks.rounds[i]
                s = np.ascontiguousarray(s[:, self.inv_perm[i]])
                s = np.ascontiguousarray(SBOX_INV_NP[s] ^ self.white[i])
                w = s.view(np.uint64).reshape(-1, 2)
                L, R = w[:, 0].copy(), w[:, 1] - self._superpose(rk, w[:, 0])
                s = np.stack([L, R], axis=1).view(np.uint8).reshape(-1, 16)
        return s ^ self.pre


# ---------------------------------------------------------------------------
# Message mode: CBC + PKCS#7, encrypt-then-MAC with HMAC-SHA256
#   output = version(1) || IV(16) || ciphertext || tag(32)
# ---------------------------------------------------------------------------
class AuthenticationError(Exception):
    pass


def _subkeys(master: bytes, params: Params):
    d = hashlib.shake_256(b"Turbo-NAFS/message-mode" + params.tag() + master).digest(64)
    return d[:32], d[32:]


class TurboNAFS:
    """High-level API.

        k = TurboNAFS.generate_key()
        c = TurboNAFS(k).encrypt(b"hello", aad=b"header")
        m = TurboNAFS(k).decrypt(c, aad=b"header")
    """

    def __init__(self, key: bytes, params: Params | None = None):
        self.params = params or Params()
        enc_key, self.mac_key = _subkeys(key, self.params)
        self.ks = expand_key(enc_key, self.params)
        self.batch = BatchCipher(self.ks)

    @staticmethod
    def generate_key() -> bytes:
        return os.urandom(KEY_BYTES)

    @property
    def kcv(self):
        return self.ks.kcv

    def encrypt(self, plaintext: bytes, aad: bytes = b"", iv: bytes | None = None) -> bytes:
        iv = iv or os.urandom(BLOCK)
        pad = BLOCK - len(plaintext) % BLOCK
        data = plaintext + bytes([pad]) * pad
        prev, out = iv, bytearray()
        for i in range(0, len(data), BLOCK):
            blk = bytes(a ^ b for a, b in zip(data[i:i + BLOCK], prev))
            prev = encrypt_block(self.ks, blk)
            out += prev
        body = bytes([VERSION]) + iv + bytes(out)
        tag = hmac.new(self.mac_key, struct.pack("<Q", len(aad)) + aad + body, hashlib.sha256).digest()
        return body + tag

    def decrypt(self, blob: bytes, aad: bytes = b"") -> bytes:
        if len(blob) < 1 + BLOCK + BLOCK + 32 or (len(blob) - 1 - 32) % BLOCK:
            raise AuthenticationError("malformed ciphertext")
        body, tag = blob[:-32], blob[-32:]
        exp = hmac.new(self.mac_key, struct.pack("<Q", len(aad)) + aad + body, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, exp):
            raise AuthenticationError("authentication failed")
        if body[0] != VERSION:
            raise AuthenticationError("unsupported version")
        iv, ct = body[1:1 + BLOCK], body[1 + BLOCK:]
        c = np.frombuffer(ct, np.uint8).reshape(-1, BLOCK)
        p = self.batch.decrypt(c)                                 # CBC decrypt is parallel
        prev = np.vstack([np.frombuffer(iv, np.uint8)[None, :], c[:-1]])
        data = (p ^ prev).tobytes()
        pad = data[-1]
        if not 1 <= pad <= BLOCK or data[-pad:] != bytes([pad]) * pad:
            raise AuthenticationError("bad padding")              # unreachable after a valid MAC
        return data[:-pad]


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def _main():
    import argparse, json, sys
    ap = argparse.ArgumentParser(description="Turbo-NAFS-128 research cipher")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("keygen"); g.add_argument("keyfile")
    e = sub.add_parser("encrypt"); e.add_argument("keyfile"); e.add_argument("infile"); e.add_argument("outfile")
    d = sub.add_parser("decrypt"); d.add_argument("keyfile"); d.add_argument("infile"); d.add_argument("outfile")
    a = ap.parse_args()

    if a.cmd == "keygen":
        k = TurboNAFS.generate_key()
        with open(a.keyfile, "w") as fh:
            json.dump({"algorithm": "Turbo-NAFS-128-CBC-HMAC-SHA256", "key": k.hex(),
                       "kcv": TurboNAFS(k).kcv}, fh, indent=2)
        print("key written; KCV", TurboNAFS(k).kcv)
        return
    kd = json.load(open(a.keyfile))
    c = TurboNAFS(bytes.fromhex(kd["key"]))
    if c.kcv != kd.get("kcv", c.kcv):
        sys.exit("key check value mismatch")
    data = open(a.infile, "rb").read()
    out = c.encrypt(data) if a.cmd == "encrypt" else c.decrypt(data)
    open(a.outfile, "wb").write(out)
    print(f"{a.cmd}ed {len(data)} -> {len(out)} bytes")


if __name__ == "__main__":
    _main()
