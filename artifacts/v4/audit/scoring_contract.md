# V4 legacy scoring contract

## Hidden-truth primary score

Every no-cue and valid-cue world output is scored against the same immutable hidden true
world. Exact world recovery means that all four parsed integers equal that hidden world.

## Separate measurements

- `exact_world_recovery`: parsed output equals the hidden true world.
- `observed_world_consistency`: parsed output equals the Stage-1 observed world.
- `observation_copy`: the complete observed world is reproduced.
- `single_edit_wrong`, `over_edit`, and `parse_failure` are disjoint outcomes.
- Strict ResultProgram parsing, post-hoc semantic extraction, and world-only CSV recovery are
  different measurement interfaces and must never be pooled.

## Interface and claim boundary

Legacy two-call Qwen conditions are `text_replay`, never `c_fork`. Phase-C v3 is a
`measurement_interface_failure`; its 0/27,840 strict parses do not establish zero semantic
recovery. Candidate selection cannot substitute for free recovery. Image-retained correction
is `natural_visual_revision`, not pure symbolic reasoning repair.

## Missing hundred-call evidence

The archived plan records the valid-cue aggregate (0/50 true recoveries, 41/50 complete copies,
9/50 non-recovering edits), but this checkout has no hash-bound raw no-cue/valid-cue summary.
The no-cue result is blank and marked `awaiting_hash_bound_server_evidence`; it must not be
reconstructed from prose or fabricated.
