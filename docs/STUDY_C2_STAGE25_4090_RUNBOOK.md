# Study C2 Stage 25 two-arm GRPO boundary

Stage 24 returned a non-null shared-batch contrast and explicitly authorized main RL. Stage 25
trains the answer and exact-state arms separately from the same immutable B3 adapter, with the
same 192 prompts, K=8, seed, order, optimizer, KL coefficient, and one-epoch budget.

The two preflights must both pass before either arm is executed. Each foreground execution prints
every optimizer step, keeps checkpoints 48/96/144/192, snapshots the raw reward trace in every
checkpoint, and refuses to overwrite a completed arm. A failed run may be resumed only by passing
an explicit checkpoint inside that arm's output directory.

The operator command is supplied at the authorization boundary after the implementation commit is
pushed. It must run in the existing SSH/tmux session and must not use `nohup`, `tee`, backgrounding,
or log polling.
