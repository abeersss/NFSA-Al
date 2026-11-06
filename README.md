# Turbo-NAFS-128: an invertible block cipher based on non-abelian functional superposition

Reference implementation, verification suite and data for the manuscript

> A. Alshammari, "An invertible block cipher based on non-abelian functional superposition," submitted to the *International Journal of Informatics and Communication Technology (IJ-ICT)*.

It builds on the preliminary design in A. S. Alshammari, "Design of Turbo-NAFS: A quantum-resilient encryption scheme based on functional superposition," SSRN preprint [6167526](https://ssrn.com/abstract=6167526), 2026.

> **Research code.** Turbo-NAFS-128 is a new cipher with no third-party cryptanalysis. The statistical results below are necessary but not sufficient for security. Do not use it to protect real data; use AES-256-GCM or ChaCha20-Poly1305 instead.

## What it is

Turbo-NAFS-128 has a 128-bit block, a 256-bit key and 16 rounds. Each round does two things:

1. It adds a key-weighted superposition of three secret composition paths to one half of the state: `R ← R ⊞ Σ αⱼ·Pⱼ(L)`. The paths are walks of length 4 on a key-derived graph over 8 non-commuting keyed bijections.
2. It applies a round key, the AES S-box to all 16 bytes and a byte permutation that swaps the two halves.

Because the superposition is applied as an additive lifting (Feistel) step, decryption simply recomputes it and subtracts it. The cipher is therefore a permutation for any choice of paths (Theorem 1 in the paper). The preliminary prototype composed a 2-to-1 map (x² + 9) directly, so it could not decrypt. `code/audit_paper_prototype.py` reproduces that failure.

Messages use CBC with PKCS#7 padding and HMAC-SHA256 in encrypt-then-MAC order, with optional associated data.

## Repository layout

| Path | Contents |
| --- | --- |
| `code/turbo_nafs.py` | The cipher: key schedule, scalar and NumPy batch implementations, message mode, command-line tool |
| `code/verify.py` | The 22-check quantitative verification suite |
| `code/audit_paper_prototype.py` | Re-runs the preliminary prototype and measures its invertibility |
| `results/results.json` | Full results of the run reported in the paper (22/22 checks passed) |
| `results/results_audit.json` | Invertibility audit of the preliminary prototype |
| `results/verify_full.log` | Console log of the full verification run |
| `paper/Alshammari_IJICT_manuscript.docx` | Manuscript in the IJ-ICT template |
| `paper/figures/` | Figures 1 and 2 and the script that draws them |

## Running it

Requires Python 3.9 or later, NumPy and SciPy.

```bash
pip install -r requirements.txt
cd code
python3 verify.py --quick        # about 20 seconds
python3 verify.py                # full suite, about 2 minutes; writes results.json
python3 audit_paper_prototype.py

python3 turbo_nafs.py keygen my.key
python3 turbo_nafs.py encrypt my.key input.bin output.enc
python3 turbo_nafs.py decrypt my.key output.enc input.out
```

From Python:

```python
from turbo_nafs import TurboNAFS
k = TurboNAFS.generate_key()
c = TurboNAFS(k).encrypt(b"hello", aad=b"header")
assert TurboNAFS(k).decrypt(c, aad=b"header") == b"hello"
```

## Main results

| Check | Measured | Ideal |
| --- | --- | --- |
| Block round trips over 1,000 keys | 1,000,000 / 1,000,000 | 100% |
| Collisions over 2²¹ distinct inputs | 0 | 0 |
| Minimum non-commutativity over all function pairs | 0.9997 | > 0.5 |
| Bits flipped by a 1-bit plaintext change | 63.996 of 128 | 64 |
| First round passing the strict avalanche criterion | 3 of 16 | well below 16 |
| NIST SP 800-22 subset (200 × 10⁶ bits) | 7 / 7 tests pass | 7 / 7 |
| Single-bit tampering rejected | 5,000 / 5,000 | 100% |

See `results/results.json` for all 22 checks and their thresholds.

## Known-answer test vectors

These vectors use the block cipher with a raw key (`expand_key`). Bytes are hex, and the 128-bit state is read as two little-endian 64-bit words.

| Key | Plaintext | Ciphertext |
| --- | --- | --- |
| `000102…1e1f` | `000102…0e0f` | `259aa298196741a96a5f59ac23d9c900` |
| `00` × 32 | `00` × 16 | `b80493328e692e8af527c92110d3e597` |

The key-check value for key `000102…1f` is `df0344cde4c30cd5`.

## Open problems

- A lower bound on the number of active S-boxes over several rounds, which would give a differential bound (and its linear counterpart)
- Reduced-round differential, linear, integral and algebraic attacks
- Whether quantum superposition-query (Simon-type) attacks on Feistel networks extend to reduced-round Turbo-NAFS
- A constant-time, optimized C implementation

## Author

Dr. Abeer Alshammari · Research Practitioner, Kuwait · ORCID [0009-0003-9379-2667](https://orcid.org/0009-0003-9379-2667)
