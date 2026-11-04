"""
Quantitative verification suite for Turbo-NAFS-128.

Every test prints a measured value, the value an ideal cipher would give, and
PASS/FAIL against an explicit statistical threshold.  Results go to
results.json.  Run:   python3 verify.py          (full, ~2-4 min)
                      python3 verify.py --quick  (smaller samples, ~30 s)

Sections
  1  Correctness       round-trips, two independent implementations agree, KATs
  2  Bijectivity       every f in F inverts; no ciphertext collisions
  3  Non-commutativity fraction of x with fi(fj(x)) != fj(fi(x))
  4  Diffusion         plaintext avalanche, SAC matrix, per-round diffusion curve
  5  Key sensitivity   key avalanche, wrong-key decryption
  6  Randomness        NIST SP 800-22 subset on ciphertext of a counter input
  7  Bias              linear correlations, differential multiplicities
  8  Integrity         tamper / truncation / wrong-AAD rejection
  9  Performance       throughput, round scaling, key setup
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
from scipy import special, stats

import turbo_nafs as tn

RNG = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
RESULTS: dict = {}
VERDICTS: list = []


def rand_blocks(n):
    return RNG.integers(0, 256, size=(n, 16), dtype=np.uint8)


def rand_key():
    return bytes(RNG.integers(0, 256, 32, dtype=np.uint8))


def bits(a: np.ndarray) -> np.ndarray:
    return np.unpackbits(a, axis=-1)


def hamming(a, b):
    return bits(a ^ b).sum(axis=1)


def verdict(section, name, measured, ideal, ok, detail=""):
    VERDICTS.append(dict(section=section, test=name, measured=measured, ideal=ideal,
                         result="PASS" if ok else "FAIL", detail=detail))
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {name}: {measured}   (ideal {ideal}) {detail}")


# ---------------------------------------------------------------------------
def s1_correctness(Q):
    print("\n1. Correctness")
    r = {}
    # (a) batch round-trip, many keys
    nk, nb = Q(1000, 100), Q(1000, 200)
    bad = 0
    for _ in range(nk):
        bc = tn.BatchCipher(tn.expand_key(rand_key()))
        X = rand_blocks(nb)
        bad += int((bc.decrypt(bc.encrypt(X)) != X).any(axis=1).sum())
    r["batch_roundtrips"] = nk * nb
    r["batch_failures"] = bad
    verdict(1, "Block round-trip D(E(x))=x", f"{nk*nb - bad}/{nk*nb}", "100%", bad == 0, f"{nk} keys")

    # (b) scalar reference == batch implementation
    agree = total = 0
    for _ in range(Q(40, 10)):
        ks = tn.expand_key(rand_key()); bc = tn.BatchCipher(ks)
        X = rand_blocks(50); C = bc.encrypt(X)
        for i in range(50):
            c = tn.encrypt_block(ks, X[i].tobytes())
            agree += (c == C[i].tobytes()) and tn.decrypt_block(ks, c) == X[i].tobytes()
            total += 1
    r["impl_agreement"] = [agree, total]
    verdict(1, "Scalar vs NumPy implementations agree", f"{agree}/{total}", "100%", agree == total)

    # (c) message mode with random lengths, AAD
    nm = Q(2000, 300)
    ok = 0
    lengths = [0, 1, 15, 16, 17, 31, 32, 33] + list(RNG.integers(0, 4096, nm - 8))
    for L in lengths:
        c = tn.TurboNAFS(rand_key())
        m = os.urandom(int(L)); aad = os.urandom(int(RNG.integers(0, 64)))
        ok += c.decrypt(c.encrypt(m, aad), aad) == m
    r["message_roundtrips"] = [ok, len(lengths)]
    verdict(1, "Message round-trip (CBC+HMAC, 0-4096 B)", f"{ok}/{len(lengths)}", "100%", ok == len(lengths))

    # (d) randomised encryption: same message twice -> different ciphertexts
    c = tn.TurboNAFS(rand_key()); m = b"A" * 64
    distinct = len({c.encrypt(m) for _ in range(1000)})
    verdict(1, "Same plaintext, 1000 encryptions -> distinct ciphertexts", distinct, 1000, distinct == 1000)

    # (e) Known-answer test vectors (deterministic, for other implementations to match)
    key = bytes(range(32)); pt = bytes(range(16))
    ks = tn.expand_key(key)
    kat = {"key": key.hex(), "plaintext": pt.hex(), "ciphertext": tn.encrypt_block(ks, pt).hex(),
           "kcv": ks.kcv,
           "zero_key_zero_block": tn.encrypt_block(tn.expand_key(bytes(32)), bytes(16)).hex()}
    msg = tn.TurboNAFS(key).encrypt(b"Turbo-NAFS test vector", b"aad", iv=bytes(16))
    kat["message_mode"] = {"plaintext": "Turbo-NAFS test vector", "aad": "aad", "iv": "00" * 16, "output": msg.hex()}
    r["kat"] = kat
    print(f"  KAT  E(00..1f, 00..0f) = {kat['ciphertext']}")
    RESULTS["correctness"] = r


# ---------------------------------------------------------------------------
def s2_bijectivity(Q):
    print("\n2. Bijectivity (decryption is only possible if nothing is lost)")
    r = {}
    fails = 0; trials = 0
    for i in range(tn.N_FUNCS):
        n = Q(300, 50) if i == 5 else Q(5000, 500)
        for _ in range(n):
            x = int(RNG.integers(0, 2**63)) * 2 + int(RNG.integers(0, 2))
            p = int(RNG.integers(0, 2**63)) * 2 + int(RNG.integers(0, 2))
            fails += tn.f_inverse_scalar(i, tn.f_scalar(i, x, p), p) != x
            trials += 1
    r["function_inverse_trials"] = trials; r["function_inverse_failures"] = int(fails)
    verdict(2, "Each f in F has an exact inverse", f"{trials - fails}/{trials}", "100%", fails == 0)

    n = Q(1 << 21, 1 << 17)
    bc = tn.BatchCipher(tn.expand_key(rand_key()))
    ctr = np.zeros((n, 16), np.uint8)
    ctr[:, :8] = np.arange(n, dtype=np.uint64).view(np.uint8).reshape(-1, 8)
    C = bc.encrypt(ctr).view(np.uint64).reshape(-1, 2)
    dup = n - len(np.unique(C, axis=0))
    r["collision_inputs"] = n; r["collision_duplicates"] = int(dup)
    verdict(2, f"Ciphertext collisions over {n:,} distinct inputs", dup, 0, dup == 0,
            "(paper prototype: 693 of 997 lost at r=5)")
    RESULTS["bijectivity"] = r


# ---------------------------------------------------------------------------
def s3_noncommutativity(Q):
    print("\n3. Non-commutativity of the functional space")
    n_params, n_x = Q(100, 30), Q(500, 200)
    M = np.zeros((tn.N_FUNCS, tn.N_FUNCS))
    for i in range(tn.N_FUNCS):
        for j in range(tn.N_FUNCS):
            diff = 0
            for _ in range(n_params):
                pi, pj = (int(v) for v in RNG.integers(0, 2**63, 2, dtype=np.int64))
                x = RNG.integers(0, 2**63, n_x, dtype=np.int64).astype(np.uint64) * np.uint64(2)
                with np.errstate(over="ignore"):
                    a = tn.f_batch(i, tn.f_batch(j, x, pj), pi)
                    b = tn.f_batch(j, tn.f_batch(i, x, pi), pj)
                diff += int((a != b).sum())
            M[i, j] = diff / (n_params * n_x)
    off = M[~np.eye(tn.N_FUNCS, dtype=bool)]
    RESULTS["noncommutativity"] = {"names": tn.FUNC_NAMES, "matrix": M.round(4).tolist(),
                                   "offdiag_min": float(off.min()), "offdiag_mean": float(off.mean())}
    verdict(3, "Distinct pairs fi∘fj≠fj∘fi (min over pairs)", f"{off.min():.4f}", "> 0.5",
            off.min() > 0.5, f"mean {off.mean():.4f}; paper's f1,f2 commute 100%")


# ---------------------------------------------------------------------------
def s4_diffusion(Q):
    print("\n4. Diffusion / avalanche")
    r = {}
    bc = tn.BatchCipher(tn.expand_key(rand_key()))
    n = Q(100000, 20000)
    X = rand_blocks(n)
    pos = RNG.integers(0, 128, n)
    Y = X.copy(); Y[np.arange(n), pos // 8] ^= (0x80 >> (pos % 8)).astype(np.uint8)
    hd = hamming(bc.encrypt(X), bc.encrypt(Y))
    mean, sd = float(hd.mean()), float(hd.std())
    # goodness of fit to Binomial(128, 1/2)
    obs = np.bincount(hd, minlength=129)
    exp = stats.binom.pmf(np.arange(129), 128, 0.5) * n
    keep = exp >= 5
    chi = float(((obs[keep] - exp[keep]) ** 2 / exp[keep]).sum()
                + ((obs[~keep].sum() - exp[~keep].sum()) ** 2 / exp[~keep].sum()))
    df = int(keep.sum())
    p = float(stats.chi2.sf(chi, df))
    r["plaintext_avalanche"] = dict(n=n, mean_bits=mean, sd_bits=sd, mean_frac=mean / 128,
                                    hist=obs.tolist(), chi2=chi, df=df, p=p, min=int(hd.min()), max=int(hd.max()))
    verdict(4, "Plaintext avalanche (1-bit flip)", f"{mean:.3f} bits ({mean/128:.2%}), sd {sd:.3f}",
            "64.000 (50%), sd 5.657", abs(mean / 128 - 0.5) < 0.005 and p > 0.001,
            f"fit to Binomial(128,½) p={p:.3f}")

    # Strict Avalanche Criterion matrix, full rounds
    def sac(rounds, N):
        cnt = np.zeros((128, 128))
        Xb = rand_blocks(N)
        base = bits(bc.encrypt(Xb, rounds)).astype(np.int32)
        for i in range(128):
            Yb = Xb.copy(); Yb[:, i // 8] ^= np.uint8(0x80 >> (i % 8))
            cnt[i] = (bits(bc.encrypt(Yb, rounds)).astype(np.int32) ^ base).sum(axis=0)
        P = cnt / N
        chi = float((((cnt - N / 2) ** 2) / (N / 4)).sum())
        return P, chi, float(stats.chi2.sf(chi, 128 * 128))

    N = Q(4096, 1024)
    P, chi, p = sac(None, N)
    dev = np.abs(P - 0.5)
    exp_max = math.sqrt(0.25 / N) * stats.norm.isf(0.5 * (1 - 0.95 ** (1 / 16384)))
    r["sac"] = dict(N=N, max_dev=float(dev.max()), mean_dev=float(dev.mean()), chi2=chi, p=p,
                    expected_max_dev_95=exp_max, matrix=P.round(4).tolist())
    verdict(4, "Strict Avalanche Criterion (128×128 cells)", f"max|p-½|={dev.max():.4f}",
            f"≤{exp_max:.4f} (95% bound for N={N})", p > 0.001, f"χ²(16384) p={p:.3f}")

    # Per-round diffusion curve
    curve = []
    Nr = Q(1024, 256)
    for rr in range(1, bc.ks.params.rounds + 1):
        Pr, chir, pr = sac(rr, Nr)
        curve.append(dict(rounds=rr, mean_flip=float(Pr.mean()), min_cell=float(Pr.min()),
                          max_cell=float(Pr.max()), sac_p=pr,
                          zero_cells=int((Pr == 0).sum())))
    full = next((c["rounds"] for c in curve if c["sac_p"] > 0.001), None)
    r["round_curve"] = curve; r["full_diffusion_round"] = full
    print("  rounds  mean-flip  min-cell  max-cell  SAC-p")
    for c in curve:
        print(f"  {c['rounds']:>6}  {c['mean_flip']:.4f}    {c['min_cell']:.4f}    {c['max_cell']:.4f}    {c['sac_p']:.3g}")
    verdict(4, "First round passing SAC (security margin)", f"round {full} of {bc.ks.params.rounds}",
            "well below r", full is not None and full <= bc.ks.params.rounds // 2,
            f"margin {bc.ks.params.rounds - full if full else 0} rounds")
    RESULTS["diffusion"] = r


# ---------------------------------------------------------------------------
def s5_key_sensitivity(Q):
    print("\n5. Key sensitivity")
    r = {}
    trials, nb = Q(2000, 300), 32
    fr = []; wrong = []
    for _ in range(trials):
        k = bytearray(rand_key()); X = rand_blocks(nb)
        b1 = tn.BatchCipher(tn.expand_key(bytes(k)))
        i = int(RNG.integers(0, 256)); k[i // 8] ^= 1 << (i % 8)
        b2 = tn.BatchCipher(tn.expand_key(bytes(k)))
        C1 = b1.encrypt(X)
        fr.append(hamming(C1, b2.encrypt(X)) / 128)
        wrong.append(1 - hamming(b2.decrypt(C1), X) / 128)        # fraction of plaintext bits "recovered"
    fr = np.concatenate(fr); wrong = np.concatenate(wrong)
    r["key_avalanche"] = dict(n=int(fr.size), mean=float(fr.mean()), sd=float(fr.std()),
                              min=float(fr.min()), max=float(fr.max()))
    verdict(5, "Key avalanche (1 of 256 key bits flipped)", f"{fr.mean():.4f} of ct bits flip",
            "0.5000", abs(fr.mean() - 0.5) < 0.005, f"n={fr.size:,} blocks")
    r["wrong_key_decrypt"] = dict(mean_bits_correct=float(wrong.mean()), exact_recoveries=int((wrong == 1).sum()))
    verdict(5, "Decrypt with 1-bit-wrong key: plaintext bits correct", f"{wrong.mean():.4f}",
            "0.5000 (chance)", abs(wrong.mean() - 0.5) < 0.005 and (wrong == 1).sum() == 0)

    # also: no two keys among many share a key-check value / first ciphertext
    ks = [tn.expand_key(rand_key()) for _ in range(Q(5000, 500))]
    c0 = {tn.encrypt_block(k, bytes(16)) for k in ks}
    verdict(5, f"Distinct keys -> distinct E_k(0) over {len(ks)} keys", len(c0), len(ks), len(c0) == len(ks))
    RESULTS["key_sensitivity"] = r


# ---------------------------------------------------------------------------
# NIST SP 800-22 subset
def nist_monobit(b):
    n = b.size; s = 2 * int(b.sum()) - n
    return special.erfc(abs(s) / math.sqrt(n) / math.sqrt(2))


def nist_block_freq(b, M=128):
    N = b.size // M; pi = b[:N * M].reshape(N, M).mean(axis=1)
    return special.gammaincc(N / 2, 4 * M * ((pi - 0.5) ** 2).sum() / 2)


def nist_runs(b):
    n = b.size; pi = b.mean()
    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        return 0.0
    V = 1 + int((b[1:] != b[:-1]).sum())
    return special.erfc(abs(V - 2 * n * pi * (1 - pi)) / (2 * math.sqrt(2 * n) * pi * (1 - pi)))


def nist_longest_run(b, M=10000):
    N = b.size // M
    blocks = b[:N * M].reshape(N, M).astype(np.int8)
    longest = []
    for row in blocks:
        d = np.diff(np.concatenate(([0], row, [0])))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        longest.append((ends - starts).max() if starts.size else 0)
    longest = np.array(longest)
    v = np.array([(longest <= 10).sum()] + [(longest == k).sum() for k in range(11, 16)] + [(longest >= 16).sum()])
    pi = np.array([0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727])
    chi = ((v - N * pi) ** 2 / (N * pi)).sum()
    return special.gammaincc(3, chi / 2)


def nist_cusum(b):
    n = b.size; x = 2 * b.astype(np.int64) - 1
    z = int(np.abs(np.cumsum(x)).max())
    sq = math.sqrt(n)
    k1 = np.arange(int((-n / z + 1) // 4), int((n / z - 1) // 4) + 1)
    k2 = np.arange(int((-n / z - 3) // 4), int((n / z - 1) // 4) + 1)
    s1 = (stats.norm.cdf((4 * k1 + 1) * z / sq) - stats.norm.cdf((4 * k1 - 1) * z / sq)).sum()
    s2 = (stats.norm.cdf((4 * k2 + 3) * z / sq) - stats.norm.cdf((4 * k2 + 1) * z / sq)).sum()
    return 1 - s1 + s2


def nist_approx_entropy(b, m=8):
    n = b.size

    def phi(mm):
        ext = np.concatenate((b, b[:mm - 1])).astype(np.int64)
        idx = np.zeros(n, np.int64)
        for k in range(mm):
            idx = (idx << 1) | ext[k:k + n]
        c = np.bincount(idx, minlength=1 << mm) / n
        c = c[c > 0]
        return (c * np.log(c)).sum()
    ap = phi(m) - phi(m + 1)
    chi = 2 * n * (math.log(2) - ap)
    return special.gammaincc(2 ** (m - 1), chi / 2)


def nist_dft(b):
    n = b.size; x = 2 * b.astype(np.float64) - 1
    S = np.abs(np.fft.rfft(x))[: n // 2]
    T = math.sqrt(math.log(1 / 0.05) * n)
    N0 = 0.95 * n / 2; N1 = (S < T).sum()
    d = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4)
    return special.erfc(abs(d) / math.sqrt(2))


NIST = {"Frequency (monobit)": nist_monobit, "Block frequency": nist_block_freq, "Runs": nist_runs,
        "Longest run of ones": nist_longest_run, "Cumulative sums": nist_cusum,
        "Approximate entropy (m=8)": nist_approx_entropy, "Spectral (DFT)": nist_dft}


def s6_randomness(Q):
    print("\n6. Randomness of ciphertext (input = counter 0,1,2,... : maximally structured)")
    r = {}
    m_seq, nbits = Q(200, 50), 1_000_000
    nblk = nbits // 128 + 1
    ctr = np.zeros((nblk, 16), np.uint8)
    ctr[:, :8] = np.arange(nblk, dtype=np.uint64).view(np.uint8).reshape(-1, 8)

    def run(source):
        pv = {k: [] for k in NIST}
        for _ in range(m_seq):
            b = source()
            for k, f in NIST.items():
                pv[k].append(float(f(b)))
        out = {}
        lo = 0.99 - 3 * math.sqrt(0.99 * 0.01 / m_seq)
        for k, v in pv.items():
            v = np.array(v); prop = float((v >= 0.01).mean())
            hist = np.histogram(v, bins=10, range=(0, 1))[0]
            pu = float(stats.chisquare(hist).pvalue)
            out[k] = dict(pass_rate=prop, uniformity_p=pu, ok=bool(prop >= lo and pu >= 0.0001))
        return out, lo

    def turbo_src():
        bc = tn.BatchCipher(tn.expand_key(rand_key()))
        return bits(bc.encrypt(ctr)).ravel()[:nbits]

    def urandom_src():
        return bits(np.frombuffer(os.urandom(nbits // 8), np.uint8)[None, :]).ravel()

    t_out, lo = run(turbo_src)
    u_out, _ = run(urandom_src)
    r["nist"] = dict(sequences=m_seq, bits_per_sequence=nbits, min_pass_rate=lo,
                     turbo_nafs=t_out, os_urandom_control=u_out)
    print(f"  {m_seq} sequences x 1,000,000 bits, each under a fresh key; pass rate must be ≥ {lo:.4f}")
    print(f"  {'test':30s} {'Turbo-NAFS':>22s} {'os.urandom control':>22s}")
    for k in NIST:
        t, u = t_out[k], u_out[k]
        print(f"  {k:30s} {t['pass_rate']:.3f} (unif p {t['uniformity_p']:.3f})   {u['pass_rate']:.3f} (unif p {u['uniformity_p']:.3f})")
    allok = all(v["ok"] for v in t_out.values())
    verdict(6, "NIST SP 800-22 subset (7 tests)", f"{sum(v['ok'] for v in t_out.values())}/7 pass",
            "7/7", allok)

    # byte-level statistics on one large ciphertext
    nbig = Q(1 << 20, 1 << 17)
    big_ctr = np.zeros((nbig, 16), np.uint8)
    big_ctr[:, :8] = np.arange(nbig, dtype=np.uint64).view(np.uint8).reshape(-1, 8)
    c = tn.BatchCipher(tn.expand_key(rand_key())).encrypt(big_ctr).ravel()
    freq = np.bincount(c, minlength=256)
    chi = float(stats.chisquare(freq).statistic); pchi = float(stats.chisquare(freq).pvalue)
    pr = freq / c.size; H = float(-(pr * np.log2(pr)).sum())
    corr = float(np.corrcoef(c[:-1].astype(float), c[1:].astype(float))[0, 1])
    r["bytes"] = dict(n_bytes=int(c.size), chi2=chi, chi2_p=pchi, entropy_bits_per_byte=H, serial_corr=corr,
                      freq=freq.tolist())
    verdict(6, f"Byte entropy over {c.size/2**20:.0f} MiB", f"{H:.6f} bits/byte", "8.000000", H > 7.9998)
    verdict(6, "Byte-frequency χ²(255)", f"{chi:.1f}, p={pchi:.3f}", "≈255, p>0.001", pchi > 0.001)
    verdict(6, "Serial correlation of adjacent bytes", f"{corr:+.5f}", f"0 ± {3/math.sqrt(c.size):.5f}",
            abs(corr) < 3 / math.sqrt(c.size))
    RESULTS["randomness"] = r


# ---------------------------------------------------------------------------
def s7_bias(Q):
    print("\n7. Linear and differential bias")
    r = {}
    bc = tn.BatchCipher(tn.expand_key(rand_key()))
    chunks, cs = Q(16, 4), 1 << 16
    N = chunks * cs
    acc = np.zeros((128, 128))
    for _ in range(chunks):
        X = rand_blocks(cs)
        P = bits(X).astype(np.float32) * 2 - 1
        C = bits(bc.encrypt(X)).astype(np.float32) * 2 - 1
        acc += P.T @ C
    corr = acc / N
    chi = float((corr ** 2 * N).sum()); p = float(stats.chi2.sf(chi, 128 * 128))
    exp_max = stats.norm.isf(0.5 * (1 - 0.95 ** (1 / 16384))) / math.sqrt(N)
    r["linear"] = dict(N=N, max_abs_corr=float(np.abs(corr).max()), chi2=chi, p=p, expected_max_95=exp_max)
    verdict(7, f"Max |corr(pt bit, ct bit)| over 16,384 pairs, N={N:,}", f"{np.abs(corr).max():.5f}",
            f"≤{exp_max:.5f} (95% bound)", p > 0.001, f"χ² p={p:.3f}")

    nd = Q(1 << 20, 1 << 16)
    X = rand_blocks(nd); Y = X.copy(); Y[:, 0] ^= 0x80          # fixed 1-bit input difference
    D = (bc.encrypt(X) ^ bc.encrypt(Y)).view(np.uint64).reshape(-1, 2)
    _, counts = np.unique(D, axis=0, return_counts=True)
    r["differential"] = dict(N=nd, distinct=int(counts.size), max_multiplicity=int(counts.max()))
    verdict(7, f"Fixed 1-bit Δin over {nd:,} pairs: most frequent Δout", f"seen {counts.max()}×",
            "1× (no repeats expected)", counts.max() <= 2)
    # per-byte differential uniformity
    Db = D.view(np.uint8).reshape(-1, 16)
    ps = [float(stats.chisquare(np.bincount(Db[:, j], minlength=256)).pvalue) for j in range(16)]
    r["differential"]["byte_chi2_min_p"] = min(ps)
    verdict(7, "Output-difference bytes uniform (16 χ² tests)", f"min p={min(ps):.4f}", "> 0.001/16",
            min(ps) > 0.001 / 16)
    RESULTS["bias"] = r


# ---------------------------------------------------------------------------
def s8_integrity(Q):
    print("\n8. Integrity (message mode)")
    r = {}
    c = tn.TurboNAFS(rand_key())
    m = os.urandom(256); blob = c.encrypt(m, b"hdr")
    n = Q(5000, 1000); rejected = 0
    for _ in range(n):
        t = bytearray(blob); i = int(RNG.integers(0, len(t) * 8)); t[i // 8] ^= 1 << (i % 8)
        try:
            c.decrypt(bytes(t), b"hdr")
        except tn.AuthenticationError:
            rejected += 1
    r["bitflip_rejected"] = [rejected, n]
    verdict(8, "Single-bit tampering rejected", f"{rejected}/{n}", "100%", rejected == n)
    tests = {"truncated": blob[:-1], "wrong AAD": None, "wrong key": None, "empty": b""}
    ok = 0
    for name, t in tests.items():
        try:
            if name == "wrong AAD":
                c.decrypt(blob, b"HDR")
            elif name == "wrong key":
                tn.TurboNAFS(rand_key()).decrypt(blob, b"hdr")
            else:
                c.decrypt(t, b"hdr")
        except tn.AuthenticationError:
            ok += 1
    r["other_rejections"] = [ok, len(tests)]
    verdict(8, "Truncation / wrong AAD / wrong key / empty rejected", f"{ok}/{len(tests)}", "4/4", ok == len(tests))
    RESULTS["integrity"] = r


# ---------------------------------------------------------------------------
def s9_performance(Q):
    print("\n9. Performance (Python/NumPy; not representative of a C implementation)")
    r = {}
    ks = tn.expand_key(rand_key()); bc = tn.BatchCipher(ks)
    X = rand_blocks(Q(200000, 50000)); mb = X.size / 2**20
    t = time.perf_counter(); C = bc.encrypt(X); te = time.perf_counter() - t
    t = time.perf_counter(); bc.decrypt(C); td = time.perf_counter() - t
    r["batch_MiB_s"] = dict(encrypt=mb / te, decrypt=mb / td)
    blk = os.urandom(16); n = Q(3000, 500)
    t = time.perf_counter()
    for _ in range(n):
        blk = tn.encrypt_block(ks, blk)
    r["scalar_us_per_block"] = (time.perf_counter() - t) / n * 1e6
    t = time.perf_counter()
    for _ in range(Q(1000, 200)):
        tn.expand_key(rand_key())
    r["key_setup_ms"] = (time.perf_counter() - t) / Q(1000, 200) * 1e3
    scaling = []
    Xs = rand_blocks(50000)
    for rr in [1, 2, 4, 6, 8, 10, 12, 14, 16]:
        t = time.perf_counter(); bc.encrypt(Xs, rr); dt = time.perf_counter() - t
        scaling.append(dict(rounds=rr, us_per_block=dt / 50000 * 1e6))
    rs = np.array([s["rounds"] for s in scaling]); us = np.array([s["us_per_block"] for s in scaling])
    lin = stats.linregress(rs, us)
    r["round_scaling"] = scaling; r["round_scaling_r2"] = float(lin.rvalue ** 2)
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        e = Cipher(algorithms.AES(os.urandom(32)), modes.ECB()).encryptor()
        data = X.tobytes(); t = time.perf_counter(); e.update(data); ta = time.perf_counter() - t
        r["aes256_reference_MiB_s"] = mb / ta
    except Exception:
        r["aes256_reference_MiB_s"] = None
    print(f"  batch encrypt {r['batch_MiB_s']['encrypt']:.2f} MiB/s, decrypt {r['batch_MiB_s']['decrypt']:.2f} MiB/s")
    print(f"  scalar {r['scalar_us_per_block']:.1f} µs/block, key setup {r['key_setup_ms']:.2f} ms")
    print(f"  runtime vs rounds linear fit R² = {r['round_scaling_r2']:.4f}")
    if r["aes256_reference_MiB_s"]:
        print(f"  (reference: AES-256 in OpenSSL {r['aes256_reference_MiB_s']:.0f} MiB/s)")
    RESULTS["performance"] = r


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()
    Q = (lambda full, quick: quick) if a.quick else (lambda full, quick: full)
    p = tn.Params()
    print(f"Turbo-NAFS-128  rounds={p.rounds} paths={p.paths} k={p.path_len} |F|={tn.N_FUNCS}  "
          f"mode={'quick' if a.quick else 'full'}")
    t0 = time.time()
    for s in (s1_correctness, s2_bijectivity, s3_noncommutativity, s4_diffusion, s5_key_sensitivity,
              s6_randomness, s7_bias, s8_integrity, s9_performance):
        s(Q)
    RESULTS["verdicts"] = VERDICTS
    RESULTS["params"] = p.__dict__
    RESULTS["runtime_s"] = time.time() - t0
    passed = sum(v["result"] == "PASS" for v in VERDICTS)
    RESULTS["summary"] = f"{passed}/{len(VERDICTS)} checks passed"
    with open(a.out, "w") as fh:
        json.dump(RESULTS, fh, indent=1)
    print(f"\n{RESULTS['summary']}  ({RESULTS['runtime_s']:.0f} s)  -> {a.out}")
    sys.exit(0 if passed == len(VERDICTS) else 1)


if __name__ == "__main__":
    main()
