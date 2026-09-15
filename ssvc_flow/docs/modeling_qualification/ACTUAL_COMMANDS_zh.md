# 实际执行命令与阶段回执

以下来自实际运行回执；失败及重跑原件保留。所有科学阶段复用同一M2，不因报告修订重跑实验。

## M0_1789452859429342000

exit=0; wall_seconds=0.46406691696029156; peak_rss_gib=0.0950164794921875

```bash
/Users/louis/Documents/ChatGPT/dissertation-ntu/.venv/bin/python -m src.modeling_qualification.cli math --out runs/modeling_qualification/qualification_20260915/M0
```

回执：`runs/modeling_qualification/qualification_20260915/.receipts/M0_1789452859429342000/result.json`；SHA-256 `b173e2a27e08bd254ee8e9d05ca0f03e291d7d0db813b58b310fb374b30225a1`。

## M1_1789452859989846000

exit=0; wall_seconds=1.2095477909315377; peak_rss_gib=0.3149871826171875

```bash
/Users/louis/Documents/ChatGPT/dissertation-ntu/.venv/bin/python -m src.modeling_qualification.cli witnesses --out runs/modeling_qualification/qualification_20260915/M1
```

回执：`runs/modeling_qualification/qualification_20260915/.receipts/M1_1789452859989846000/result.json`；SHA-256 `8f3bcc9fafc451417c60a04ef7e030590ec9363e313a4fb183ca89ebf5dad5ea`。

## M1_portable_1789455142548367000

exit=0; wall_seconds=1.5256814999738708; peak_rss_gib=0.314971923828125

```bash
/Users/louis/Documents/ChatGPT/dissertation-ntu/.venv/bin/python -m src.modeling_qualification.cli witnesses --config configs/modeling_qualification/protocol.json --out runs/modeling_qualification/qualification_20260915/M1_portable
```

回执：`runs/modeling_qualification/qualification_20260915/.receipts/M1_portable_1789455142548367000/result.json`；SHA-256 `92021d4e69d9dfb468a0826e9754bec71dd298afbe992ee95914654e2cd8019c`。

## M2_1789452996207387000

exit=0; wall_seconds=26.334821458091028; peak_rss_gib=0.3299713134765625

```bash
/Users/louis/Documents/ChatGPT/dissertation-ntu/.venv/bin/python -m src.modeling_qualification.cli collect --profile core --out runs/modeling_qualification/qualification_20260915/M2
```

回执：`runs/modeling_qualification/qualification_20260915/.receipts/M2_1789452996207387000/result.json`；SHA-256 `987c06d6402059b0abad5272a5aa97b21afc76f185d1437de4b717b6b0f248bc`。

## M3_1789453342425778000

exit=0; wall_seconds=74.99786529201083; peak_rss_gib=1.4338531494140625

```bash
/Users/louis/Documents/ChatGPT/dissertation-ntu/.venv/bin/python -m src.modeling_qualification.cli fit-evaluate --input runs/modeling_qualification/qualification_20260915/M2 --out runs/modeling_qualification/qualification_20260915/M3
```

回执：`runs/modeling_qualification/qualification_20260915/.receipts/M3_1789453342425778000/result.json`；SHA-256 `62b2876d227754e0c7e4d1f0bcbc2a799dc9411e2052cb392a62d9c2fe6c3444`。

## M3_verified_1789453528962991000

exit=0; wall_seconds=77.17399912502151; peak_rss_gib=1.433563232421875

```bash
/Users/louis/Documents/ChatGPT/dissertation-ntu/.venv/bin/python -m src.modeling_qualification.cli fit-evaluate --input runs/modeling_qualification/qualification_20260915/M2 --out runs/modeling_qualification/qualification_20260915/M3_verified
```

回执：`runs/modeling_qualification/qualification_20260915/.receipts/M3_verified_1789453528962991000/result.json`；SHA-256 `29697295404183183c5bc3226c36732b590653717d53c3fec0e6ecc2a1cc27b8`。

## M4_1789453774932420000

exit=0; wall_seconds=71.69586904195603; peak_rss_gib=1.199951171875

```bash
/Users/louis/Documents/ChatGPT/dissertation-ntu/.venv/bin/python -m src.modeling_qualification.cli track --input runs/modeling_qualification/qualification_20260915/M2 --models runs/modeling_qualification/qualification_20260915/M3_verified --out runs/modeling_qualification/qualification_20260915/M4
```

回执：`runs/modeling_qualification/qualification_20260915/.receipts/M4_1789453774932420000/result.json`；SHA-256 `5989920677085052071c20dede166ef78151e11c9ff670f13eaf035ab61f644a`。

## smoke_1789452812864046000

exit=0; wall_seconds=1.0538706670049578; peak_rss_gib=0.27392578125

```bash
/Users/louis/Documents/ChatGPT/dissertation-ntu/.venv/bin/python -m src.modeling_qualification.cli collect --profile smoke --config configs/modeling_qualification/protocol.json --out runs/modeling_qualification/qualification_20260915/smoke
```

回执：`runs/modeling_qualification/qualification_20260915/.receipts/smoke_1789452812864046000/result.json`；SHA-256 `f475968a9d45c1e07b32a06151c25058f4082d84670c33bfd173d09be20a9ad6`。

## results_20260915_1789455808171283000

exit=0; wall_seconds=15.182803833042271; peak_rss_gib=0.159210205078125

```bash
/Users/louis/Documents/ChatGPT/dissertation-ntu/.venv/bin/python -m src.modeling_qualification.cli report --input runs/modeling_qualification/qualification_20260915 --out docs/modeling_qualification/results_20260915
```

回执：`docs/modeling_qualification/.receipts/results_20260915_1789455808171283000/result.json`；SHA-256 `5d1fd031f0caa6c29e5cbbadf8a4a2f546646a9518ca6f448c052c854cc07500`。

## results_20260915_1789456022037787000

exit=0; wall_seconds=14.85950712498743; peak_rss_gib=0.158447265625

```bash
/Users/louis/Documents/ChatGPT/dissertation-ntu/.venv/bin/python -m src.modeling_qualification.cli report --input runs/modeling_qualification/qualification_20260915 --out docs/modeling_qualification/results_20260915
```

回执：`docs/modeling_qualification/.receipts/results_20260915_1789456022037787000/result.json`；SHA-256 `3d65f4abf008a63924ba1ffadbab609f50d8229db6a3d8268a2fd02739071cc7`。

## results_20260915_1789456101182072000

exit=0; wall_seconds=6.692488916916773; peak_rss_gib=0.1571197509765625

```bash
/Users/louis/Documents/ChatGPT/dissertation-ntu/.venv/bin/python -m src.modeling_qualification.cli report --input runs/modeling_qualification/qualification_20260915 --out docs/modeling_qualification/results_20260915
```

回执：`docs/modeling_qualification/.receipts/results_20260915_1789456101182072000/result.json`；SHA-256 `c72ea4bdf8ffc4ea0a46adcbcbddd4722ad28929fa9814923765b15fd72881f5`。
