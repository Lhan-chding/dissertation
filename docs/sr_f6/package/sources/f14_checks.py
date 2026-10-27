"""Re-derive the counterexamples in GPT's round-2 package (SR_Semantic_Feedback_Discussion_20261027).
Pure arithmetic; no data needed. Run: python3 -I f14_checks.py
"""
import math, statistics

print("== 1. residual-only relation credit rewards consistent error ==")
# model answers each of a pair correctly w.p. p (independent); residual credit = 1 iff both wrong but consistent
# (worst case: wrong answers are always consistent) -> E[credit] = (1-p)^2, d/dp = -2(1-p) < 0
for p in (0.1, 0.5, 0.9):
    print(f"  p={p}: E[residual]={(1-p)**2:.3f}  dE/dp={-2*(1-p):+.3f}")
# with J anchor: objective p + lam*(1-p)^2, derivative 1 - 2 lam (1-p)
for lam, p in ((0.25, 0.1), (1.0, 0.1), (1.0, 0.6)):
    print(f"  J-anchored lam={lam}, p={p}: d/dp = {1 - 2*lam*(1-p):+.3f}")

print("\n== 2. joint-objective gradient weights vs (1/2,1/2) ==")
# maximize P(J_i=1 and J_j=1) = p1*p0 (independent) -> grad = (p0, p1): the weaker item gets the LARGER weight
for p1, p0 in ((0.9, 0.3), (0.5, 0.5)):
    print(f"  p1={p1}, p0={p0}: grad=(dP/dp1, dP/dp0)=({p0}, {p1}); sum-of-marginals grad=(1,1)")

print("\n== 3. LOCAL per-token coefficient A/T + c/m (SR-F5 numbers) ==")
A, T, c, m = -0.378, 220, 0.25 * (0 - 7/8), 10   # lone J=0 in a 7/8 group, S=FAIL; mask of 10 tokens
print(f"  A/T={A/T:+.5f}  c/m={c/m:+.5f}  per-token={A/T + c/m:+.5f}  flip when T/m > {abs(A/c):.2f}")

print("\n== 4. 32 failures with 0 observed -> 95% one-sided upper bound ==")
n = 32
print(f"  rule of three: {3/n:.3f};  exact (1-0.05^(1/n)): {1 - 0.05**(1/n):.3f}")

print("\n== 5. prompt-scalar cancellation under group standardization ==")
j = [1, 1, 1, 1, 1, 1, 1, 0]
k = 0.37   # any per-prompt constant added to every reward
def std(x):
    mu, sd = statistics.mean(x), statistics.pstdev(x)
    return [(v - mu) / sd for v in x]
print("  A(j)      :", [round(v, 3) for v in std(j)])
print("  A(j+k)    :", [round(v, 3) for v in std([v + k for v in j])])
print("  identical ->", std(j) == std([v + k for v in j]) or max(abs(a-b) for a,b in zip(std(j), std([v+k for v in j]))) < 1e-12)

print("\n== 6. J-stratum-centred credit (my a_R proposal) in the SR-F5 7/8 group ==")
s = [1, 1, 1, 1, 1, 1, 0, 0]   # index 6: J=1 but S=FAIL (lucky success); index 7: J=0, S=FAIL
lam = 0.25
c = []
for i, (ji, si) in enumerate(zip(j, s)):
    stratum = [sk for jk, sk in zip(j, s) if jk == ji]
    c.append(lam * (si - statistics.mean(stratum)))
print("  c_i:", [round(v, 3) for v in c], " (lucky success penalised, lone failure untouched, no cross-stratum leak)")
