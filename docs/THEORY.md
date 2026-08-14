# Theory map

The v2 theory is implemented in `src/compbias/theory`,
`src/compbias/identification`, and `src/compbias/estimation`.

The verified components include:

1. natural-trajectory KL selection law;
2. covariance direction law and pairwise odds;
3. repeated-selection and coordination identities;
4. Bregman and scaling identities;
5. natural-versus-synthetic transport bounds;
6. task-induced behavioral pseudometrics;
7. crossed-risk compensation interaction;
8. factorization non-identifiability and multi-interface partial identification;
9. optimizer-independent checkpoint density-ratio identity.

The legacy seven families of finite-table property checks contain 7,000 random
instances and a recorded maximum absolute error of approximately `5.17e-11`,
below the `1e-8` tolerance. The v2 additions have separate property-test
evidence under `artifacts/theory_v2/property_tests.json`.

For assumptions, equations, and claim limits, see `paper/theorem_notes_v2.md`
and `paper/identification_contract.md`.
