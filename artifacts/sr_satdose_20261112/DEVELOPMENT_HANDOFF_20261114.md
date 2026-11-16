# Development snapshot — not execution admission

User requested ending continuous development/polling and an hourly monitoring cadence. No background wake-up or app automation was configured. The last observed job status is PREFIX 1737594 PENDING at 2026-11-13 15:47 UTC; the next proposed observation at 16:47 UTC has not occurred. Slurm continues independently. Do not resubmit, cancel, or retry an unknown attempt.

The user explicitly approved the complete four-step training risk interpretation, using 384 conflict responses per window. The immutable clarification is `intake/TRAIN_RISK_CLARIFICATION_20261114.json`; bind it before any TRAIN generation. Public push authorization remains valid.

New local components, not deployed:

- `analysis/evaluation.py`: complete-endpoint CPU evidence/scoring, 17 targeted tests passed using pinned native/legacy scoring. Does not grant MAIN release or training admission.
- `execution/canary_audit.py`: independent four-path evidence audit and frozen-weight fingerprints. Synthetic CPU fixtures do not demonstrate actual GPU canary or process termination.
- `execution/evaluation_runtime.py`: WIP, syntax checked only; no behavior/integration tests yet. Do not deploy or execute models with it. BEGIN/native capture wiring, source identity transitions, authority guards, no-retry and RNG preservation still require tests and review.

Existing deployed GPU release eff7036 and PREFIX release 734e921 remain unchanged. Do not create TECHNICAL_EXECUTION_BINDING until both complete PREFIX start gates pass. No actual GPU admission, TRAIN, DEV or MAIN has been performed.

On resumption: read latest ADMISSION_REPORT and job receipts, perform a read-only queue/attempt audit, preserve frozen samples/seeds/gates, and only then continue authorized dependent work. The local monitor helper is `/private/tmp/satdose_monitor_prefix_short.py`; it performs read-only SSH observations and saves a local timestamped receipt. It is not a scheduler.

Still required before TRAIN: concrete authenticated admission, actual GPU checks and independent audit, full immutable training release/controllers and source bindings, explicit storage budgets, initial DEV, first-two-step/canary process supervision and four-path evidence, source-bound risk handling. MAIN remains sealed until all four true step24 chains complete without HOLD/UNKNOWN and terminal authority validates them.

Bounded independent static review found no credentials or dangerous default activation; syntax checks passed. Canary targeted tests: 11 passed (95.101 seconds), using synthetic GPU/process evidence and actual small CPU Adam states. Evaluation runtime still has no behavior test coverage. See intake/DEVELOPMENT_SNAPSHOT_REVIEW_20261114.json.

Final stable full CPU suite: 348 tests passed, zero skips, 249.158 seconds. Log: intake/development_snapshot_cpu_tests_20261114.log. This does not cover evaluation_runtime behavior/integration or actual GPU execution.

Update 2026-11-14 11:41 SGT: job1737594 is terminal FAILED/2:0 from intentional HOLD_PREFIX_MEASUREMENT, not pending. 11031 completed48, 11032 never began. Read-only full reproduction and independent record/parser/stop audit agree. REVTIE sensitivity known13/16 below14/16; no implementation recovery found. Do not resume waiting for this job or submit training. Read prefix_calibration/11031/DIAGNOSIS_zh.md and latest ADMISSION_REPORT first.


Update 2026-11-14 12:02 SGT - supersedes the prior HOLD-only handoff:

- User explicitly waived this PREFIX measurement gate and requested PASS. Execution decision is PASS_USER_OVERRIDE; original measurement stays 13/16 vs 14/16, HOLD/candidatefalse. Authorization SHA 36012b46494f4fcb3409f4816156406caad7ae1ad65c4688b972b75c83d73783 and supplementary label SHA 377a444e380e3041e80e0f5ec7b1da1da6957e2d13a106e45c614d71e73dabe1 are deployed in protocol_clarifications.
- Code 0e5ba1989e43e0e20adbb78327fbe66c5a2ad87d pushed and verified; 386 full CPU tests passed, zero skips, independent review completed. New continuation/waiver modules and worker/launcher integration passed source review; no actual GPU admission yet.
- New 69-file release BASE/waiver_code_0e5ba19 deployed, all SHA verified, original seven runtime files unchanged. PREFIX_CONTINUATION_RELEASE_0e5ba19.json SHA c41c6892403148de7027a4ad7085ddc95e22837605a4f85d202ce13e65ffafc2; GPU_RELEASE_0e5ba19.json SHA c8f0645f52ae8756bc312722a164c9b5bfe264db43ea23a048ffe0bfa074e05a. Old 734/eff7036 preserved. Use the new GPU release for the future waiver branch; never modify the old releases.
- Job 1738080 submitted once at 04:02:13 UTC, 2 pro6000 / 8CPU / 66G / 02:00:00 / no-requeue, name LH-SATDOSE-PREFIX-11032. Actual PENDING(Priority) at 04:02:40 UTC; forecast 13:11:32 UTC = 21:11:32 SGT, mutable. Scope only original 11032/48, no 11031 resampling or expanded 96 budget. Logs ROOT/scheduling/prefix_continuation_20261114/slurm-1738080.{out,err}. Remote SUBMISSION_INTENT/RESULT prevent duplicate submission.
- Remaining 11032 currently zero attempts. BASE/prefix_waiver_control is created only on execution. Expected complete result status PREFIX_CAPTURE_COMPLETE_WITH_AUTHORIZED_MEASUREMENT_WAIVER, gate_decision PASS_USER_OVERRIDE, NO_TRAINING. Actual analyses can still HOLD. Technical HOLD/UNKNOWN never permits automatic retry. Verify full original 96 evidence with prefix_waiver.verify_completed before future GPU work.
- GPU binding is absent. Future binding must use new release/code SHAs, add code.prefix_waiver, prefix_waiver authorization ref and prefix_continuation actual RESULT ref. prefix_result remains the ORIGINAL prefix_control/RESULT.json SHA d4027f82a6b4bdcdfe944dce9ee90912486fcc460503ff4da07c3fd3ac434ddd. Other fixed GPU, four-shadow and storage gates unchanged. Complete 96 first. Existing independent GPU auditor can import the verified new release and call the new worker gate.
- Future formal admission must distinguish user override from actual measurement pass. Do not manufacture original prefix_calibration_pass=True or modify frozen check_admission.py. Concrete admission/controller/canary/evaluation/risk implementation remains as noted above. User-confirmed 384-conflict denominator remains in force.
- No automatic monitoring configured. Stop continuous polling after verified submission per user's prior preference. On resumption read current receipts and check 1738080, not old 1737594. Never resubmit automatically.


Update 2026-11-14 23:35 SGT - current GPU execution handoff:

- PREFIX1738080 completed0:0 at05:21:56UTC (13:21:56SGT), all96 complete. 11032 sensitivity gate passed (known14/15/14, stablecorrect14, difference13). 11031 failed metrics remain under explicit user override. Actual combined RESULT SHA6bb7c58eb71d9211bec6a186d101ed74c85df2a6af6aa4c97cade9d4e687dfa6.
- Complete96 evidence/source checks and independent local binding review passed. GPU TECHNICAL_EXECUTION_BINDING.json now EXISTS at BASE, SHA cf7da7f08e5cb11c5458cab1200fd5a2465a529a31d2258f6cc45cbe6efe73c5. Never reset or silently overwrite it. Uses new waiver_code_0e5ba19 GPU release c8f064... and actual full prefix continuation. Budget BASE/execution_releases/GPU_STORAGE_BUDGET_20261114.json SHA4bcc7bc07d3c07aa263a937f3e6ddd265e23c8c76ffc8bfe0550ac0f6463bcd6,11114315776bytes, expiry2026-11-21T15:30:47.346499+00:00, per-stage observation freshness300seconds. It is no reservation.
- Actual CPU worker._preflight and exact bound launcher Python guard ran onserver, original1280DRY/native/source/diagnostic inputs revalidated, actual Ceph ancestor quota+statvfs/fsync probe passed, no model/forward/optimizer calls. Receipt BASE/execution_releases/GPU_BINDING_PREFLIGHT_20261114.json; local gpu_execution_20261114/GPU_BINDING_PREFLIGHT_20261114.json.
- GPU job1739113 submitted once15:34:33UTC; nameLH-SATDOSE-GPU-CHECKS,2pro6000/8CPU/66G/24hours/no-requeue, fixed ten stages, maxfour discarded shadow Adam steps, no new generations or formal training. Last15:35:03UTC PENDING(QOSMaxJobsPerUserLimit), no allocation or stage dirs, forecastN/A. Logs ROOT/scheduling/gpu_checks_20261114/slurm-1739113.{out,err}; same folder immutable SUBMISSION_INTENT/RESULT. Do not resubmit automatically, cancel another user's jobs, alter gates or reset occupied slots.
- Next check1739113 and actual ten stage receipts/HOLD, not old1738080. If all complete, independently run BASE/independent_audit_code_61f74e7/gpu_evidence.py with trustedbindingSHA above and auditorSHA cb3015f96a6d0149cd48cbbb0bd9283cf1f8eb83530af4dc59395ad90960a56e. That audit remains NO_TRAINING. Do not claim GPU admission from Slurm COMPLETED alone. Actual full128 long1536row and allzero/backend/shadow evidence still pending.
- No new code changes this turn; existing386CPUtests are prior source validation, not newGPU evidence. Code/public push authorization remains valid. No background automation configured; user prefers no continuously running chat. Current user asked checking1738080 then proceeding toGPU; submission achieved, future completion requires actual audit. Formal admission, first2/canary, risk384, initialDEV and finalMAIN work remains, with explicit useroverride recorded instead of falsifying originalprefix_calibration_pass.

## QoS correction — 2026-11-15 17:55 Singapore

- Active GPU diagnostics job is now **1740151**, replacing **1739113**. Monitor the new ID.
- User explicitly requested teacher QoS and no changes to others' jobs. Ordinary `--qos=rose` in the original submission was an execution error; correct QoS is `soujanya-poria-startfund-2026-03`, while account remains `rose`.
- Cluster rejected `scontrol update JobId=1739113 QOS=...`: submitted-job QoS cannot be changed; cancellation/resubmission required. Only the old pending job was cancelled, verified CANCELLED/ElapsedRaw=0/no allocation and no stage directories.
- Submitted identical frozen command once, changing only QoS. New job 1740151 initially verified PENDING(Priority), then confirmed RUNNING on gpu-pro6000-8 with 2 PRO6000; stage 1740151.0 is running. Same 2 PRO6000, 8 CPUs, 66G, 24h, no-requeue, ten finite stages, max four shadow steps, and binding cf7da7f08e5cb11c5458cab1200fd5a2465a529a31d2258f6cc45cbe6efe73c5. No source changes, no resampling, no formal training, no other job mutations.
- Receipts: `gpu_execution_20261114/QOS_CORRECTION_1739113_20261115.json`, `QOS_REPLACEMENT_1739113_20261115.json`. Server immutable intent/cancellation/submission/result receipts: ROOT/scheduling/qos_correction_1739113_20261115. Logs remain ROOT/scheduling/gpu_checks_20261114/slurm-1740151.{out,err}.
- Future related submissions must explicitly use the teacher QoS; account and QoS are distinct fields.

## Independent GPU audit completed — 2026-11-16 00:34 SGT

- 1740151 terminal COMPLETED/0:0, all ten stages complete. Start09:56:19UTC and end16:07:55UTC Nov15, elapsed06:11:36. No GPU work remains in this job; do not resubmit its four shadow slots.
- Actual independent audit passed at16:25:27UTC Nov15: `GPU_EVIDENCE_AUDIT_1740151.json` SHA fa87ca99a5dbe2c23adee24d19c2ab46ea87f3b6cb4d51880522573f23641f60, 3233 observed files, ten stages, four discarded shadow steps, zero formal steps/model/optimizer calls by auditor. Plain exact stdout published BASE/execution_releases/GPU_EVIDENCE_AUDIT_1740151.json and local gpu_execution_20261114. Wrapper INDEPENDENT_AUDIT_1740151_20261115T162528Z.json preserves sacct and stderr.
- Both actual direction checks pass; J/source cosine11031=.6840225062,11032=.5430288800; production replay bit exact; 11032 original1536token row preserved in full128. Production peak memory remains NOT_RECORDED, trained-checkpoint GPU evaluation remains unverified.
- New independent supplementary source225e358c4b2b2a064d057e8de8f73666fe4922181c362e7adbf1de73f547c78c under BASE/independent_supplementary_225e358c4b2b verified all28936 public requests, historical exclusions+2560 newcases/96retentioncases, 9nativegolden cases, actual restored source/close and 1280DRY zero updates. Five proofJSONs under BASE/execution_releases/supplementary_20261116; local wrapper SUPPLEMENTARY_AUDIT_20261116_225e358c4b2b.json.
- Technical report constructorc42ad925423aca75a7fe06a490c1ad0f02124517e67c4d2e5b6e075823950fcd published BASE/execution_releases/GPU_VERIFICATION_1740151.json SHA e592f4806c279544a4206b4dfb0318a8e0899b052924c4958fa932518efd5a0a, also local exactcopy. All14technicalchecks true; original PREFIX check remains false, override separately bound. Still NO_TRAINING until independent amended admission.
- User now explicitly requests continuing without stopping; proceed through authorized dependent work, preserving all real gates. No scheduled app automation is created for a specific job.

## START DEV submitted — 2026-11-15T16:56:55.563408+00:00

- Stable immutable server release BASE/start_dev_code_9b925a3 has74 exact source files. Code committed/pushed9b925a3; prior075cc59 path-compatibility incident preserved, zero model calls at that failed CPU audit. FullCPU491pass and current model-link targeted41pass, both independent reviews complete.
- Full CPU admission actually reran native1280DRY,96PREFIX,all14technicalchecks and evidence/source hashes. Published spec execution_releases/START_DEV_ADMISSION_SPEC_9b925a3.json SHA79a6b8401723152b74bab2f71958fde5fe6ee4bc06613deef60c5a2203ec2cf5; admission START_DEV_ADMISSION_9b925a3.json SHA9b92c278ced475c316bb528b81e308d1f575c0b7f10c8352b62110dd81525185.
- Actual STARTDEV CPU preflight passed108fixed11031requests/no model calls. Release execution_releases/START_DEV_RELEASE_9b925a3.json SHA22f82a3e504c8838ba3496eedf6b31d77addef5f07360fc6baba9024c8ca96c2. Budget293601280bytes, expiryNov22UTC; checks actual Ceph quota/statvfs/fsync each endpoint.
- Submitted uniquejob1740676, teacherQoS,2PRO6000/8CPU/66G/6h/no-requeue, fixedtwofreshprocesses11031then11032,108each. Server receipts/logs ROOT/scheduling/start_dev_9b925a3; results BASE/start_dev/start-<seed>/RESULT.json. Must inspect complete original captures, RECEIPT and independent analysis before firstTRAIN.
- FutureTRAIN source identity not yet frozen. Firsttwo worker,actualexit launcher,riskassessor and continuation/trainedDEV controllers under implementation/review. Include all future source-bound dependencies before firstGENESIS; cannot replace admission_verifier/sourceidentity atstep3. Firsttwo remainslimited1–2, fullfourpathactualcanary beforestep3, riskDEV2/8/16, noMAINuntilallfourstep24.

## Future TRAIN code release readiness — 2026-11-16

- FullCPU626tests PASS413.360s, log intake/future_train_release_cpu_tests_20261116.log. All firsttwo/actualwait/risk/continuation/trainedDEV code independently reviewed; details+84source hashes in intake/FIRST_TWO_CODE_REVIEW_20261116.json. External MAIN development excluded.
- Same-source ContinuingAdmission must be frozen beforeGENESIS. OriginalTRAIN19key execution_sources carries firsttwo, launcher, canary auditor, risk, continuation and trainedDEV controllers. No source substitution afterstep2. Fourpaths chunks3–8/9–16/17–24 require completedDEV barriers2/8/16 and outside-lock risk reassessment aftereachCOMMIT.
- FIRST_TWO_STORAGE_BUDGET_20261116.json SHA cc81fc1689fb48cc6887df5895709bbb5cecd38535a10326132ea950e9d64f69,required32GiB, observedR24size~260MB andproductioncapture~175MB. Conservative allowance, not sharedspace reservation. No formalTRAIN yet.

## Original TRAIN source freeze published — 2026-11-16 01:22 SGT

- Commit71018e5720b593b4a37624a8f6b6b3a618aa3f03 pushed;84 exact files deployed to BASE/train_code_71018e5. SOURCE map19keys SHA(object)=ab25b75ab7abb255a7f8d795a20f4f930be1a6cb3a61bb7d1f16ac46003869cf; local ORIGINAL_TRAIN_EXECUTION_SOURCES_71018e5.json. Do not add later MAIN development files to this immutable directory.
- Full real admission reran17:20:38–17:22:26UTC Nov15; FIRST_TWO_ADMISSION_SPEC_71018e5.json SHA9ca67124a9ae4085499955f161c551adcf1cfe08790a997811186a21bfc610ac; FIRST_TWO_ADMISSION_71018e5.json SHA2bdc1a83502b565313ceba3fd278ec091e220c198e105d3d942a02793941a9c7. No TRAIN executed.
- Exact originally confirmed risk document copied to BASE/protocol_clarifications/TRAIN_RISK_CLARIFICATION_20261114.json SHAdaedf60444d249f3062009737626b951e05747de85ee4c84cd976dcadb1b394b beforeTRAIN. This is not a new threshold.
- STARTDEV1740676 at17:18:15UTC RUNNING,11031 completed39/108 with40BEGIN,11032 not yet started; noHOLD. Wait complete endpoints and terminal scheduler result; do not resample.
- IMPORTANT: independently analyze each completed STARTDEV endpoint USING train_code_71018e5/analysis/evaluation.py, not start_dev_code_9b925a3, because analysis_sources paths are part of the exact analysis object that first_two preflight recomputes. Same frozen scoring bytes, originalcaptured endpoint/source evidence unchanged. Publish complete {rows,summary} as execution_releases/START_DEV_ANALYSIS_71018e5_<seed>.json.
- Temporary operation scripts currently /tmp/satdose_analyze_startdev.py <seed>, /tmp/satdose_first_two_release.py; both only execute after real prerequisites. Final firsttwo release not yet published because2STARTDEV analyses missing. MAIN terminal/audit/controller development is external, uncommitted and excluded from84fileTRAINrelease.

## External MAIN / final analysis implementation ready — 2026-11-16

- Separate combined77CPUtests PASS60.080s, log external_main_analysis_cpu_tests_20261116.log; independent24terminalaudit/13terminalauthority/21MAIN/19finalanalysis and shellreviewpassed. Exact sourceSHA in EXTERNAL_MAIN_CODE_REVIEW_20261116.json. These are syntheticCPU engineering checks, not real terminal or MAIN results.
- Deploy MAIN to separate sibling directory with only execution/{terminal_audit,terminal_authority,main_control}.py and scheduling/run_main.sh; originalTRAIN84files remain unchanged. External final_analysis.py belongs in another separate directory. Bootstrap python-I authenticates both sourceinventories then imports originalTRAIN namespace.
- MAIN can run only after actual96COMMITs/12288TRAIN,14DEV/1512responses,12continuationchunks,originalfirsttwo/canary andfinalrisk fully audited,noHOLD/unknown/activeworker. Fixed14MAIN endpoints total13760:2START1952 each, then11031/11032×T0/T1×step8(256),16(256),24(1952). NoMAIN attempted/published yet.
- Finalanalysis verifies all14 completed endpoints, retains13760scoredrows,256pairedroot/startrows,originalbootstrapB10000seed2026111219,strict/sensitivity/UNKNOWN/NA and all cost evidence. Source/units distinguish allocatedGPU time from measuredcompute; no missingpeak inference. NoautomaticStage2.

- External code commit4190bc30da4550f213279bf16372e41b9af03f16 pushed and deployed: BASE/external_main_code_4190bc3 (threePythonmodules+run_main.sh) and BASE/external_analysis_code_4190bc3 (final_analysis.py only). Deployment verified original train_code_71018e5 all84files unchanged before/after. Receipt intake/EXTERNAL_CODES_DEPLOYMENT_4190bc3_20261116.json. No terminal/MAIN authorization created.
- Deferred scripts also prepared: /tmp/satdose_first_two_preflight.py 71018e5 and /tmp/satdose_first_two_submit.py 71018e5. Execute only after both actualSTARTDEV analyses and firsttwo release publication. Submit uses teacherQoS/2PRO6000/24h with durable unique intent and prerequisite1740676COMPLETED0:0; never repeat an unknown submission.

## START11031 independent DEV analysis complete — 2026-11-16 01:56 SGT

- 1740676 hourly17:56:15UTC: firststart108/108 and .0COMPLETED0:0 elapsed56:02, secondstart2/108 .1RUNNING, noHOLD.
- Actual analysis using originalTRAIN71018e5 completed17:56:50UTC. RESULTref start_dev/start-11031/RESULT.json SHA900ffd57cce3e1d217ab3e81c80c21f6d44e4762511d3ad609c6a4e196d3abb6; analysis execution_releases/START_DEV_ANALYSIS_71018e5_11031.json SHAdfe11d67863df16a4b8180680c945258f1079722f1ab5b6abb10603b02c072a2. Local exactbytes in start_dev_20261116/start-11031/{RESULT,ANALYSIS}.json withpublicationreceipt.
- Baselineonly: S_hit87/108, conflict16/32, aligned+no_list59/64, retention12/12; all108balanced/jsonvalid, S_known108. These are starting riskbaselines, nottrainingeffects orGPUtrainingacceptance. Do notpeek/score incomplete11032.
- Next hourlycheck due18:56UTC (02:56SGT). On complete11032 run /tmp/satdose_analyze_startdev.py 11032, then requirejob1740676COMPLETED0:0, publishfirsttwo release /tmp/satdose_first_two_release.py, CPUpreflight andunique submission scripts. No formalTRAIN yet.

## Operator preparation scripts archived — 2026-11-16 02:19 SGT

- `experiments/sr_satdose_20261112/operations/` now retains actual-canary publication, trained DEV preparation, continuation preparation and firsttwo read-only monitoring scripts. These are separate from immutable server TRAIN84 and its original19key source identity; do not deploy them into train_code_71018e5.
- Static/API independent review passed. A real transaction-root `TRANSACTION.lock` inventory mismatch was corrected in the unexecuted DEV operator script before archival. Source bytes and original stdout/stderr are preserved around future SSH operations. See intake/OPERATOR_SCRIPTS_REVIEW_20261116.json for exact source hashes and limits.
- No new server publication, formal TRAIN, GPU submission or canary PASS occurred during this preparation. Current job remains STARTDEV1740676 as last observed17:56UTC; next hourlycheck18:56UTC. Do not confuse static operator readiness with actual GPU acceptance.

## START DEV complete; first-two submitted — 2026-11-16 03:00 SGT

- STARTDEV1740676 terminalCOMPLETED0:0,elapsed01:54:56;11031/11032 each108BEGIN/CAPTURE/COMPLETE. Actual independent11032 analysis from immutableTRAIN71018 completed18:56:49UTC. ResultSHA dfe1a714026dddd205560f29c92be302a9434675e135fa23c36dd322145209f8; analysisSHA d10b74377286380b8ece5255ee3c412ca0a7de40ec56a9dd562ae9a6b4cc171a at execution_releases/START_DEV_ANALYSIS_71018e5_11032.json. Local exact bytes+publication understart_dev_20261116/start-11032. 11032baseline conflict14/32,aligned+no_list63/64,retention12/12;all108balanced/jsonvalid,noUNKNOWN. These are baseline risk inputs, not efficacy.
- Published actual execution_releases/FIRST_TWO_RELEASE_71018e5.json SHA22f3b7b260670628398154652455520cd25169234f66a827d3d117def924438f. CPUpreflight returned FIRST_TWO_PREFLIGHT_COMPLETE_NO_MODEL_CALLS/start11031/T0/baseline,exit0. Temporary submission script response schema corrected beforeanysubmission; no frozenserver code/gate changed. Sources and correction archived in operations and intake/FIRST_TWO_OPERATOR_SCRIPTS_20261116.json.
- Uniquejob1740834 submitted underteacherQoS/rose/cluster02/2PRO6000/8CPU/66G/24h/no-requeue.19:00:23UTC actualPENDING(Priority),zeroall4pathattempts/COMMITs. ROOT/scheduling/first_two_71018e5 hasdurableINTENT/RESPONSE/SUBMISSION;logs slurm-1740834.{out,err}. Nootherjobmutation. Budgetfixed1024TRAIN/8formalAdam+8freshworkers,zeroextraGENforcanary. PreserveoriginalTRAIN84files+19-keyidentitythrough24.
- Nextregularcheck20:00UTC/04:00SGT using operations/satdose_monitor_firsttwo.py1740834 (spacebetweenfilenameandjob). Onterminalsuccess, use operations/satdose_publish_actual_canary.py --modepublish withactualjob1740834 andreleasepath/SHAabove (spacesbetweenflagsandvalues), actualindependentaudit+secondprepare_canaryrerun. NoCANARYPASS/DEV2/step3orMAINyet.
- FollowingcanaryrequiresactualDEV2all4 andrisk, fixedchunks+DEV8/16, fullterminalaudit/MAIN; do notstopafterintermediatephase. Userexplicitlyrequestscontinuedexecution; hourlychecksinthecurrentchat, noappautomationcreated.

## First-two hourly observation — 2026-11-16 04:00 SGT

Job1740834 RUNNING on gpu-pro6000-12, elapsed00:59:52 at20:00:40UTC Nov15. TeacherQoS unchanged. Firstpath11031/T0 step1 has113BEGIN/112CAPTURE/112COMPLETE, active .staging-step-001, zeroCOMMITs sofar. Active staging duringliveworker is notterminalUNKNOWN/failure. Otherthreepaths notstarted. NoHOLD/ERROR incontroller/launcher; parentlogs empty. Receiptintake/MONITOR_1740834_20261115T200040Z.json. Nextregularcheck21:00UTC/05:00SGT. Do notrerunrelease/submissionoroccupiedpath. Independentcanary onlyafteractualallfourpaths+all8workerscompleted.

## First-two hourly observation — 2026-11-16 05:00 SGT

At21:00:56UTC job1740834 RUNNING elapsed02:00:09. 11031/T0 step1 COMMIT exists; step2 active53/128closed generations, cumulative181complete/182BEGIN. Other3paths notstarted. NoHOLD/ERROR; independentcanary notyetpossible. Receiptintake/MONITOR_1740834_20261115T210056Z.json. Nextregularcheck22:01UTC/06:01SGT.

## Connectivity interruption — 2026-11-16 06:01–06:03 SGT

Hourly1740834 monitor and bounded read-only retries timedout. Local scutil reportsSD-WAN Disconnected. Authorized attempt tostarttheexistingconnection returned0 butstatusremainedDisconnected; noVPNconfigurationchange. CUAinventory reportedMaclocked/automaticunlockunavailable; noUIactionsperformed. Userprompted to unlock/restoreVPN. CurrentremotejobstateUNKNOWN, lastsuccessful21:00:56UTC observationremains181closedTRAIN/1COMMIT on11031/T0. Nojobcancel/resubmit/retry. Onrestoration, immediatelyrun operations/satdose_monitor_firsttwo.py1740834 andcontinueactualcanary/DEVgatesonlywhencomplete. Receiptintake/MONITOR_1740834_20261115T2201Z_NETWORK_BLOCKED.json.

## VPN restored; three first-two paths complete — 2026-11-16 13:33 SGT

- At05:33:08UTC, actual1740834 RUNNING elapsed10:32:20 on gpu-pro6000-12. Firstthreepaths11031/T0,11031/T1,11032/T0 each256/256TRAIN,2COMMITs,baseline+reloadRESULTs andlauncherper-pathactualexitRESULT. Final11032/T1 active7complete/8BEGIN,zeroCOMMITs. Total775/1024closedTRAIN and6/8COMMITs. NoHOLD/ERROR. Full1,078,913-byte observation retained inintake/MONITOR_1740834_20261116T053308Z.json, SHA71cfc7ab455489bc1710b5a23cea63c4423600011d681b9870d6cb60494692f2.
- Monitoring operator nowprintscompactstatus/counts/results withoutrepeated496tensoridentities; fullobservationfiles remainbyte-for-bytecomplete. Actualreceiptprojection verified775responses/6COMMITs,5214displaybytes,unchangedoriginalSHA; noSSH/runtimecodechangesforthisdisplayfix.
- Nextregularcheck06:33UTC/14:33SGT. Continuecurrenttaskwithoutstopping. Actualfour-pathindependentauditandprepare_canary onlyafterfinalpath+launcher+jobcomplete. OriginalFIRST_TWO_RELEASE_71018e5 SHA22f3b7b260670628398154652455520cd25169234f66a827d3d117def924438f. Do notrepeatoccupiedfirsttwoattempts orresubmit1740834.

## Hourly observation — 2026-11-16 14:33 SGT

At06:33:27UTC job1740834 RUNNING elapsed11:32:39; final11032/T1 step1 has125/128closed generations,126BEGIN,zeroCOMMITs. Firstthreepaths completebaseline/reload/actualexit and2COMMITseach. Total893/1024TRAIN,6/8COMMITs,noHOLD/ERROR. Nextcheck07:33UTC/15:33SGT. Receiptintake/MONITOR_1740834_20261116T063327Z.json.

## Hourly observation — 2026-11-16 15:33 SGT

At07:33:33UTC job1740834 RUNNING elapsed12:32:45; final11032/T1 step1COMMIT exists,step2has60/128closedgenerations,cumulative188complete/189BEGIN. Total956/1024TRAIN,7/8COMMITs,noHOLD/ERROR. Firstthreepaths baseline/reload/actualexitcomplete. Nextcheck08:33UTC/16:33SGT. Receiptintake/MONITOR_1740834_20261116T073333Z.json.

## Hourly observation — 2026-11-16 16:33 SGT

At08:33:49UTC job1740834 RUNNING elapsed13:33:01. All1024TRAIN generationscomplete;7COMMITs. Final11032/T1 step2stagingactive,nobaseline/reloadRESULTyet. Firstthreepaths remaincomplete;noHOLD/ERROR. Nextcheck09:33UTC/17:33SGT. Receiptintake/MONITOR_1740834_20261116T083349Z.json. Generationcompletiondoesnotmeaneighthupdateorcanarycomplete.

## Actual canary passed; DEV2 preparation active — 2026-11-16 17:46 SGT

- Firsttwo1740834 actualCOMPLETED0:0 elapsed13:56:17,allfourpaths1024TRAIN/8COMMIT/8workerexits,noHOLD/staging. Finalmonitorintake/MONITOR_1740834_20261116T093406Z.json.
- Operations/satdose_publish_actual_canary.py actuallyran (notplan),exit0. RemoteBASE/execution_releases/canary_job_1740834 containsfirstindependentaudit,secondactualreruncomparison,CANARY_RESUME_PASSandfinalspec. Fulloriginalbytesdownloadedlocallycanary_1740834_20261116; publication_success true. AuditSHAbd05513a439a3a106a954c492080697537c04f823fe053ba214ffea49bac59a1; PASSa195a4f80a4da2c0f52e6fe4d7260213e692b9c9bdb63fbeabfb54fc7bb1a655; spec44a8db68c655bea987629ea210da1a32e268806f6e0c7b1506d6d77828d4b260; publicationf91b19cb63b683210eb156bdd5f5bc7854caa4a7eac3a9db7f792abf1c49f1a7. 2756observedfiles;all496LoRAtensorschangeeach8update;760frozenparametersidentical;buffersnotverified/offlineGPUweightsnotreobserved. ZeronewGEN/model/optimizerinaudit. Thisisengineeringadmissiononly.
- FirstDEV2preparation11031/T0 nowrunningvia operations/satdose_prepare_trained_dev.py,publicationtag71018e5_20261116,local trained_dev_20261116/start-11031_T0_step-002/PREPARATION.json andrawlogs/source/intent. ItbindsaboveCANARY_SPECandoriginalFIRST_TWO_RELEASE22f3b7b260670628398154652455520cd25169234f66a827d3d117def924438f. Do notretry ifnetworkunknown.
- Newoperations/satdose_submit_trained_dev.py SHA40c465fea1719dc13d31b390c7583180b00d5cd9d05e90addee7f04bcc307bf5 passedindependentreview+10offlinechecks (intake/TRAINED_DEV_SUBMIT_OPERATOR_REVIEW_20261116.json). Actualschedulerstillunexercisedatthisnote. Requiresactualsuccessfulpreparationreceipt+SHA,defaultplan. With--execute: oneheldsbatch teacherQoS/rose/cluster02/2PRO6000/8CPU66G/3h/no-requeue→durableacceptedidentity→unlockworker→scontrolreleaseonlynewverifiedjob. ClearsSBATCH_* andSLURM_CLUSTERS,doesnotinheritCPUemptyCUDAmask. Unknownneverresubmit/cancelotherjobs. Rootwillinvokeafterpreparationcompletesandcodepush.

## DEV2 preparation complete; submission preflight active — 2026-11-16 18:06 SGT

- Actual 11031/T0/step2 preparation finished at10:06:19UTC with TRAINED_DEV_RELEASE_CPU_PREFLIGHT_AND_STORAGE_OBSERVED. Local PREPARATION.json SHA c39067d6168e7d5b5f7e0223d5c66095458a3d558b37b7c60e1728c80da2b61c under trained_dev_20261116/start-11031_T0_step-002. Remote preparation result SHA cea4a190f556c230154a8fd999d88982f4367404a9b3c0de178ac18221d6f7a9; release SHA c7931c9607d5090c52af76d052755ae8b68e7290a1b0a8b90e6c9b975767b61d. Original84/19 sources,8commits,2completedSTARTDEV,all4pendingDEV2 and actual canary were checked. Effective available quota118904400065bytes; actual file/directory fsync passed. Observation is not reservation.
- Submit operator40c465fe default plan passed, then explicit --execute started using that actual receipt/SHA and new same-folder SUBMISSION.json. Full CPU preflight is in progress. No jobID returned at this note; do not repeat the submission. Source/intent/rawlogs preserved.
- Independent DEV auditor SHA7c29df29ab6115b322505c59e44bb666d1ef79bf0fe6048c111b25826e0227e7 and read-only DEV monitor added outside original TRAIN84. Auditor independent review and18 offline boundary checks passed; evidence intake/TRAINED_DEV_AUDITOR_OFFLINE_7c29df29_20261116_*. Auditor preserves complete scientificHOLD with planned exit2, cannot waive OOM/unknown/incomplete/foreignjobs, and publishes only analysis evidence. No actual DEV audit yet. Root reran18 checks successfully.

## First trained DEV submitted — 2026-11-16 18:29 SGT

- Actual submitted/released job1741834, LH-SATDOSE-DEV-11031-T0-s02, teacherQoS/rose/cluster02/2PRO6000/8CPU66G/3h/no-requeue. Submit40c465fe completed23min CPU preflight, then exactlyone held sbatch, durable ownidentity, unlocked lease and releasedonlythisjob. Remote RELEASE_RESULT SHA708c521a8a6449b7eb2510c0c37b905e4efa4a048a423e5915b2be7daf0947e5. Local samefolder SUBMISSION.json SHA5242b3a1041a936e272eda47a19da3bebe73ed769efd44aff19c7dda420e977c, source/rawlogs preserved.
- At10:29:22UTC actualsacctPENDING0:0/squeuePENDING(Resources),zeroBEGIN/CAPTURE/COMPLETE,no endpointdirectory/noincidents. Receipt intake/MONITOR_DEV_1741834_20261116T102922Z.json. Nextregularcheck11:29UTC/19:29SGT using operations/satdose_monitor_trained_dev.py --job-id1741834 --start-seed11031 --armT0 --step2 (spaces after each flag). Do notresubmit.
- Onceactualterminalcomplete108, run operations/satdose_audit_trained_dev.py with above absoluteSUBMISSION.json+SHA,job1741834,start11031,T0,step2,new local independent_audit directory and--execute. Defaultlocalplan with actualreceipt passed at10:29UTC; noauditSSHyet. AuditorSHA7c29df29ab6115b322505c59e44bb666d1ef79bf0fe6048c111b25826e0227e7. Preserve scientificHOLD/negative; nextDEV path only afteractualaudit/risk. Still8/96 formalupdates; no step3.
- Futurefixedchunk submitoperatorSHAeceec5aab4de56b53d8e1483d2e49093e28f4048b2ef818302bb3db499fb0f94 independentlyreviewed,21offlinechecksPASS,committed65203af andpushed. Defaultplan; explicitexecute requiresactual continuationpreparation and allprior4DEV. 24h isschedulerceilingonly. No actualcontinuationpreparation/submission.

## Future terminal operator ready locally — 2026-11-16 18:37 SGT

- operations/satdose_publish_actual_terminal.py SHA49e1188ea76d5945ee21faad394b0ea06064bd18f1f8fe0b836e43940c783511: independent review and22offline testsPASS; root independently reran22 and actual-input defaultlocalplan (no network). Exactsource/tests/rawlogs/SHAreceipt in intake/TERMINAL_PUBLISHER_OFFLINE_49e1188e_20261116_*.
- Future explicitexecute pins originalTRAIN84/19,external4190bc3,actualcanary andoriginalrelease; CPU-only actual96COMMIT/14DEV/12chunk/12288TRAIN gate,actualfinalrisk,twoindependentaudits,writeonceauthoritypublication/exactdownloads. Fixed remote execution_releases/terminal_4190bc3. Uncertainpublication only--download,neverretry. CompletePUBLICATION_RECEIPT+exactmanifest/noERRORmandatory forMAIN; candidateSPEC/ADMISSIONaloneinsufficient. No actualterminal/MAINreleasecreated.
- Currentactualjobremains1741834 pendingResources atlast10:29UTC; nexthourlycheck11:29UTC. Futureoperatorreadiness doesnotalterliveexecutionprogress.

## Future MAIN operators ready locally — 2026-11-16 18:56 SGT

- Newoperations/satdose_prepare_main.py SHA6e40ae7735fc5d6e2386cf0e996a4db4a3fa90d6eea79d13a8d14d03325302b5 andsatdose_submit_main.py SHA23a1bba1cf35a0120b034005e9df66bb4bb0ac10eeb92c8e15af9d984b2e4c50. Independentreview39testsPASS; rootreadcriticalinterfacesandreran39PASS. Evidenceintake/MAIN_OPERATORS_OFFLINE_6e40ae77_23a1bba1_20261116_* retainsrawlogs,exactSHAsandAPIreview.
- Futureprepare consumesfullterminalmanifest+LOCAL_DOWNLOAD_RECEIPT+PUBLICATION_RECEIPT,noERROR,notintermediateSPEC; fixedTRAIN84/19/external4190nativeauthorityandfullMAINpreflight,exact14endpointorder,actualquota/fsync. Futuresubmitrepinsactualpreparer/report, repeatsactualchecks; teacherQoS2GPU/no-requeue,full1952=24h/primary256=6h schedulerceilings. Oneheldsbatch,durableownidentity,unlock,recheckHOLD+terminal,releaseonlythisnewjob; noautomaticretry.
- No actualterminal/MAINpreparation/submission. Stillmonitor1741834 at11:29UTC/19:29SGT. OriginalTRAIN84/external4190unchanged.

## VPN interruption — 2026-11-16 19:29 SGT

- Hourly DEV1741834 monitor plus separate read-only sacct retry bothSSHtimeout255. SD-WANDisconnected;scutilstart returned0 but statusremainedDisconnected. CUAgetState confirmedMaclocked/automaticunlockfailed;CUAcompletionrecorded. Userrequestedunlock+restoreVPN viaasyncquestion. Nojobmutation/resubmission. Seeintake/MONITOR_DEV_1741834_20261116T1129Z_SSH_TIMEOUT.json.
- ActualcurrentjobstateUNKNOWN. Lastsuccessful10:29:22UTC=PENDING(Resources),0generation. Onrestoration immediatelyuseoperations/satdose_monitor_trained_dev.py --job-id 1741834 --start-seed 11031 --arm T0 --step 2,then actualindependentaudit onlyifcomplete108/terminal. Do notconfuse1741834 DEVwithcompleted1740834 firsttwo. ExactsubmissionSHA5242b3a1041a936e272eda47a19da3bebe73ed769efd44aff19c7dda420e977c andnewindependent_auditdirectoryspecifiedabove. Continueallauthorizeddependentworkonceaccessreturns.
- Newoperators/testscommittedandpushed: b738812DEV audit/monitor,65203afcontinuation submit,e9d294aactualDEVjob,d080837terminalpublisher,dee6582MAINprepare/submit. Lastpushindependentlyverified dee658257c098fdefcfbed4883be6f577843192d. Actualterminal/MAINstillnotexecuted; allreadinessisCPUofflineonly.

## DEV1741834 independently audited; next preparation active — 2026-11-16 21:25 SGT

- VPNrestored,13:10:17UTCactualjob1741834 COMPLETED0:0,allsteps0,01:05:08,gpu-pro6000-12,108/108,noHOLD. Monitorintake/MONITOR_DEV_1741834_20261116T131017Z.json.
- Auditor7c29df29 actuallyranonce: remoteexecution_releases/trained_dev_audit_job_1741834_s11031_T0_step002,13:13:26UTCfull108CPUscore andPOST_RISKexactreplay,noHOLD,threependingDEV2. AUDIT_REPORT eb989d200905a8ccd5647c494456c4374d063affe0ca00b334af41dda0d32f22,ANALYSIS a6a6697511b82a0095f68d2aaab222bbc32878615bb43699e94481e90c903f5c,bothPOST_RISK/replay833c87194cb5c810932c4165dba1082200ff40c0ee3ad329528d6099a042baf4. S87/108,conflict15/32,aligned+nolist60/64,retention12/12,allbalanced/jsonvalid/known. SeeendpointREPORT_zh.md.
- InitialauditdownloadfailedhashcheckatRISK_REPLAY;oldindependent_audit/LOCAL_ERRORretained,8matchingfilespresent. Read-onlyremotecheckconfirmedall10originalmanifestSHAsunchanged. Newrecoveryoperatorc168dcfc27f351a49d31a38c4680e8772271b6c08dea184a10995175722d54de independentlyreviewed+18offlinePASS,executedonceonlyreadonlySSHbundle; exact10 recovered to independent_audit_download_recovery,fullrawstreamsretained. RECOVERY_RECEIPT08c35c26038a4b290363a13a0adeb42c72c48e4272bb967187631c13e0214f32,originalmanifestfaadc50d39edaf4d08199df8e8007cee9aea46b10adad6f0823bdacb0fd4e25b. Noauditrerun/modelcall/remoteartifactwrite;rootverifiedall10bytesandPOST==REPLAY. Badfirsttransferbyteswerenotretainedbyoldauditoperator;causeUNKNOWN.
- Next11031/T1/step2preparation actuallystarted withoriginalprepare188b5104,publicationtag71018e5_20261116,sameFIRST_TWO22f3/CANARY44a8. Activeexecsession84955. Localendpoint trained_dev_20261116/start-11031_T1_step-002/PREPARATION.json (notyetreturned),source/intent/rawlogs. Do notrerun. Onsuccessfulactualreceipt computeSHA,defaultplanthenexecuteoriginalsubmit40c465fewithnewSUBMISSION.json;monitorreturnedjobhourly,thenactualaudit7c29df29. OriginalTRAIN84/19unchanged. Still8formalCOMMIT/1024TRAIN; no step3untilall4DEV2/risk.

## Second DEV2 preparation complete; submission in progress — 2026-11-16 21:38 SGT

- Actual11031/T1/2preparationfinished13:37:53UTC: original84/19,8COMMIT,3completedDEV(2START+11031T0),threependingDEV2,noHOLD. LocalPREPARATION.jsonSHAf0e670fbd4a47b88c34955a327e7910b29d09b4cf3a33139a0b056e22158b3a9,remotePREPARATION_RESULT2f8772c8fd1854f4c9624f5289cb676aef6937a287c6afc0591193c2104fd655,RELEASE920dc0f9b371ae92f617c2cbee0339b6eaf7e28ec407e22b9835245f8849e3f4. Effectiveavailablequota118881952354bytes,fsyncchecksPASS,notreservation.
- Defaultsubmitplanpassed. Actualsubmit40c465fe nowrunningexecsession70146 withaboveactualreceipt/SHA andnewstart-11031_T1_step-002/SUBMISSION.json. NojobIDreturnedyet. Do notrerun. OnreturnreadexactjobID,monitorhourly,independentactualauditwhen108complete.
- FirstDEVactualaudit/recoveryandnewread-onlydownloadoperatorcommitted/pushedc601f643a180695cbdd97a3c50474acd43b7b3a4,remoteindependentlyverified. Oldbadtransfererrorretained.

## Second trained DEV submitted — 2026-11-16 21:58 SGT

- Actualjob1742113 submitted/released13:57:01UTC for11031/T1/step2,108responses,teacherQoS/rose/cluster02/2PRO6000/8CPU66G/3h/no-requeue. RemoteRELEASE_RESULT SHA9f16bbc5091f2cb84a03c56b8dc63e35b599f39bc20040a326352886b9cc43f2. Localstart-11031_T1_step-002/SUBMISSION.json SHA7e7ac3d8304289cd0a0e0388de53aa70b85e1b1dae2764899101f2909e996af7,returncode0,statusTRAINED_DEV_SUBMITTED_RELEASE_CONFIRMED. Oneheldsbatch thenexactownjobrelease; nootherjobsmodified.
- 13:58:11UTCactualsacctPENDING0:0/squeueResources,noendpointdirectory/attempts/incidents. Monitorintake/MONITOR_DEV_1742113_20261116T135811Z.json. Nexthourlycheck14:58UTC/22:58SGT: operations/satdose_monitor_trained_dev.py --job-id 1742113 --start-seed 11031 --arm T1 --step 2. Do notresubmit.
- Onactualterminalcomplete108, useauditor7c29df29 withaboveabsoluteSUBMISSION.json/SHA,job1742113/start11031/T1/step2,newlocal sameendpoint/independent_audit,--execute. DefaultlocalplanwithactualreceiptalreadyPASS,noactualauditSSHyet. Ifdownloadunknownretainoldlocalerrorandusec168dcfcdownloadrecoveryagainstoriginalmanifestSHA,separatenewdir;donotrepublish/re-audit.
- Afterrealindependentaudit/noHOLD,next11032/T0/2 then11032/T1/2,eachactualpreparation+submission+audit. Still8COMMIT/1024TRAIN; no step3orMAIN.

## Hourly DEV observation — 2026-11-16 22:58 SGT

At14:58:34UTCjob1742113stillPENDING(Priority),Elapsed0,noallocation/endpointdirectory/BEGIN/CAPTURE/COMPLETE/incidents. TeacherQoS/3h unchanged,notresubmitted. Receiptintake/MONITOR_DEV_1742113_20261116T145834Z.json. Nextregularcheck15:58UTC/23:58SGT. FirstDEV1741834independentlyaudited; remainingthreeDEV2pending. NoadditionalTRAINorMAIN. Prioractualsubmissionrecordcommitb6ad9cb13dc8d646e5a652d859543b649bc2519a pushedandremoteverified.

## Second DEV2 audited; third preparation active - 2026-11-17 00:06 SGT

- 15:58:19UTC actual1742113 COMPLETED0:0/allsteps0,00:33:08,gpu-pro6000-3,108/108,noHOLD. Monitor intake/MONITOR_DEV_1742113_20261116T155819Z.json.
- Original auditor7c29 ran once. Actual auditSHA5eafcb8d911f4eb94efc893bfbbacbd5c7a1443888b9f453eaeb3eb36ad19c3c,analysis d616d641c9c70bb5efa9cc358b333cc8c3a199b549654a36da06f1ed64977348,POST/replay093caebcfe280423d2bcc8145ac7d058c767bcfc91271d76a457dc89cca70a14. S91/108,conflict18/32,aligned+nolist61/64,retention12/12,A90,J81;allbalanced/jsonvalid/known. NoHOLD;pending11032/T0/2,11032/T1/2.
- Per-file download again failed at RISK_REPLAY; old8matchingfiles andLOCAL_ERROR retained. Existing c168 read-only bundle recovery succeeded for all10 exact files/fullrefs/POST==REPLAY. Originalmanifest79703bb3aed27557dd1c17168cc8d96a228a476de11852165065a047e37753c8;RECOVERY_RECEIPT80c92a1176cfae42193c0495800a5b625d033ee4a1f9e391e9e5da1e778385a2. No audit/model/risk rerun. See endpoint REPORT_zh.md and independent_audit_download_recovery.
- Third11032/T0/step2 actualpreparation active execsession20264. Local trained_dev_20261116/start-11032_T0_step-002/PREPARATION.json notyetreturned; samepublicationtag71018e5_20261116,original22f3/canary44a8/source188b. Do notrerun. OnactualsuccesscomputeSHA,originalsubmit40c465fe plan thenexecuteonce withnewSUBMISSION.json;monitor returnedjob hourly.
- risk_gate preparing separate futureauditdriver: exact originalauditor7c29 remote source once, then c168 single-bundle download. Oldauditor and original84/19/scoring/risk untouched. Driver notyetready/committed/executed. Currentaudits already complete; neverrerun.

## Future DEV audit transport fixed without changing audit source - 2026-11-17 00:10 SGT

- New operations/satdose_run_trained_dev_audit.py SHAa9c4dae31e434346beb7325eec6e6830e4a10e0ff960628eae560aef1da009c7. Originalauditor7c29/recoveryc168 unchanged. Independent35checksPASS(17newdriver+18recovery),root17newdriverPASS. Evidenceintake/TRAINED_DEV_AUDIT_DRIVER_OFFLINE_a9c4dae3_20261117_*;reviewreceipt a8520d3735503008f17decb423f02210e9a976daec7776cc21200079b80cef93.
- For future11032/T0 andT1 completeDEV, use newdriver with actual--submission-receipt/SHA/job/start/arm/step andNEW --local-dir plus--execute. Sameexactoriginalauditor7c29 remote-worker once; audit/retainsmanifest/raw; download/singlec168read-onlybundle. Driver report preservesoriginalsourceidentity andscientificHOLD, grants no successor. Do notuseoldmulti-SSHdownloader forfutureaudits. Alreadycomplete1741834/1742113neverrerun. No actualnewdriverinvocationyet.
- Currentthird11032/T0preparationstillactive exec20264; no newjobID. FirsttwoDEV2 actualevidencecommitted/pushed f643ab43742843e04e2bf2c3fe6faae3a5864305 andremoteverified.

## Third DEV2 preparation passed; submission active - 2026-11-17 00:41 SGT

- Actual11032/T0/2 preparation finished16:39:36UTC with original84/19,8COMMIT,4completedDEV(twoSTART+two11031),risk093caebc/noHOLD,pendingtwo11032DEV2. LocalPREPARATION.json SHA4554ffd3f713733f13cda6cc911a79ec3685af1e01986592e999067a3e3ee41e. RemotePREPARATION_RESULTca79e57f41a91e473b017614609a88a4555ac209a29460cc0b5e2771e3319efa; RELEASE6a841da8978bf1d27741d2bdec0cfb44a70c33c49ac3ff5164e150f59a733532. Effectiveavailablequota118858971649bytes,actualfsyncPASS,notreservation.
- Defaultsubmitplanpassed,thenactualoriginalsubmit40c465fe runningexecsession29262 withaboveactualreceipt/SHA,newstart-11032_T0_step-002/SUBMISSION.json. NojobIDyet;do notrerun. Onreturnmonitoractualnewjobhourly. Oncecomplete108andterminal, useNEWsatdose_run_trained_dev_audit.py a9c4dae3 withactualsubmission/SHA/job/11032/T0/2 andfreshlocaldriverdirectory; original7c29audit followedbyc168exactbundle.
- Newdriver committed/pushed4f1ed7f9dc3c81484707f30e5fce134671906a7d andremoteverified. ActualnewdriverGPUendpointauditnotyetinvoked. Currentrawpreparer/submissionsourcesandlogsretained; no frozenTRAIN84 changes.


## Third DEV2 submitted and connection verified - 2026-11-17 01:03 SGT

- Actual submit40c465fe completed successfully, job1742309 for11032/T0/step2,108responses. TeacherQoS/rose/cluster02/2PRO6000/8CPU66G/3h/no-requeue. Exactlyoneheldsubmission and ownrelease at17:00:03UTC; nootherjobsmodified. LocalSUBMISSION.json SHA74f74df2a80467d2b37eadf867fa6f802ff91f19635c7f27f9c51b9fcf749e50. RemoteRELEASE_RESULT ca7efc22933776b2631e623136e05624f54b1deb48a573998c561c649366b28e.
- Monitors17:00:32 and17:03:49UTC confirm PENDING(Priority),zeroBEGIN/CAPTURE/COMPLETE,noallocation/endpoint/incidents. Connectionavailable. Receiptsintake/MONITOR_DEV_1742309_20261116T170032Z.json and170349Z.json. Nextregularcheck18:03UTC/02:03SGT. Noresubmission.
- New auditdriver a9c4dae3 defaultlocalplan passed with actualsubmission/SHA/job1742309,11032,T0,2; planned new local directory sameendpoint/independent_audit_bundle. NoactualauditSSH. Once108completeandallSlurmstepsterminalsuccess, execute this driver once; exact7c29originalaudit followedby c168singlebundle. Fullactualrisk/noHOLD then next11032/T1/2 preparation. All4DEV2 required beforecontinuation3-8.
