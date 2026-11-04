"""
Audit of the Section 8 prototype from the Turbo-NAFS preprint (SSRN 6167526).

Reproduces the paper's code verbatim and measures the property an encryption
scheme must have before anything else: can every ciphertext be decrypted back
to exactly one plaintext?  It also checks whether the output depends on a key.
"""
import random
import json
import math
from collections import Counter

# ---- paper code, verbatim (Section 8.2) ------------------------------------
def f1(x): return (x * 7 + 3) % 997
def f2(x): return (x * 11 + 5) % 997
def f3(x): return (x ** 2 + 9) % 997
functions = [f1, f2, f3]

def turbo_round(x, seed):
    random.seed(seed)
    for _ in range(3):
        f = random.choice(functions)
        x = f(x)
    return x

def encrypt(message, rounds=5):
    state = message
    for i in range(rounds):
        state = turbo_round(state, seed=i * 31 + 7)
    return state
# -----------------------------------------------------------------------------


def entropy(values):
    c = Counter(values)
    n = len(values)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def main():
    out = {}

    # 1. Reproduce the paper's headline numbers
    out["paper_example"] = {"c(123)": encrypt(123), "c(124)": encrypt(124)}

    # 2. Invertibility: how many plaintexts in Z_997 can be uniquely decrypted?
    per_round = []
    for r in range(1, 9):
        cts = [encrypt(m, r) for m in range(997)]
        c = Counter(cts)
        uniquely_decryptable = sum(1 for m in range(997) if c[cts[m]] == 1)
        largest_preimage = max(c.values())
        per_round.append({
            "rounds": r,
            "distinct_ciphertexts": len(c),
            "uniquely_decryptable_plaintexts": uniquely_decryptable,
            "pct_lost": round(100 * (1 - uniquely_decryptable / 997), 1),
            "largest_collision_class": largest_preimage,
            "entropy_bits": round(entropy(cts), 4),
        })
    out["invertibility"] = per_round

    # 3. f3 alone is not a bijection (squaring mod an odd prime is 2-to-1)
    img = {f3(x) for x in range(997)}
    out["f3_image_size"] = len(img)

    # 4. Is there a key?  The turbo seeds are fixed constants (i*31+7), so the
    #    "ciphertext" is identical for everyone - it is a public hash, not a cipher.
    out["path_per_round_r5"] = []
    for i in range(5):
        random.seed(i * 31 + 7)
        out["path_per_round_r5"].append([random.choice(["f1", "f2", "f3"]) for _ in range(3)])

    # 5. Non-commutativity of the paper's F on Z_997
    pairs = {}
    for a, fa in zip(["f1", "f2", "f3"], functions):
        for b, fb in zip(["f1", "f2", "f3"], functions):
            if a < b:
                diff = sum(1 for x in range(997) if fa(fb(x)) != fb(fa(x)))
                pairs[f"{a},{b}"] = round(diff / 997, 4)
    out["noncommuting_fraction"] = pairs

    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    res = main()
    with open("results_audit.json", "w") as fh:
        json.dump(res, fh, indent=2)
