# 固定提示词实例

本文件由附件原题和reference编译器生成；不是模型输出。O0保持原system/user字节。

## O0

基础场景：`2587437e7468366b845916e243e020b6`；输出顺序：`[0, 1, 2, 3]`

```text
SYSTEM
You are repairing a four-integer chart record. Return only one JSON array
containing four integers from 0 to 99. Do not include an explanation.

USER
The record order is [a,b,c,d]. For the chart, a and b are the two series
at the first x position; c and d are the same two series at the second position.
The observed record is [35,27,11,48].
Exactly one value in this observed record is wrong.
The following relationships are reliable:
a + d = 83
b + d = 75
c + d = 54
The downstream calculation is: max(a,b,c,d)-min(a,b,c,d).
Recover the correct record. Return [a,b,c,d], not the downstream answer.
```

## A1

基础场景：`2587437e7468366b845916e243e020b6`；输出顺序：`[3, 2, 1, 0]`

```text
SYSTEM
You are repairing a four-integer chart record. Return only one JSON array
containing four integers from 0 to 99. Do not include an explanation.

USER
The record order is [a,b,c,d]. For the chart, a and b are the two series
at the first x position; c and d are the same two series at the second position.
The observed record is [35,27,11,48].
Exactly one value in this observed record is wrong.
The following relationships are reliable:
a + d = 83
b + d = 75
c + d = 54
The downstream calculation is: max(a,b,c,d)-min(a,b,c,d).
Recover the correct record. Return [d,c,b,a], not the downstream answer.
```

## B0

基础场景：`2587437e7468366b845916e243e020b6`；输出顺序：`[0, 1, 2, 3]`

```text
SYSTEM
You are repairing a four-integer chart record. Return only one JSON array
containing four integers from 0 to 99. Do not include an explanation.

USER
The record order is [a,b,c,d]. For the chart, a and b are the two series
at the first x position; c and d are the same two series at the second position.
The observed record is [35,27,11,48].
Exactly one value in this observed record is wrong.
The following relationships are reliable:
a + d = 83
b + d = 75
c + d = 54
The downstream calculation is: max(a,b,c,d)-min(a,b,c,d).
Recover the correct record. Return [a,b,c,d], not the downstream answer.
```

## B1

基础场景：`2587437e7468366b845916e243e020b6`；输出顺序：`[0, 1, 2, 3]`

```text
SYSTEM
You are repairing a four-integer chart record. Return only one JSON array
containing four integers from 0 to 99. Do not include an explanation.

USER
The record order is [a,b,c,d]. For the chart, a and b are the two series
at the first x position; c and d are the same two series at the second position.
The observed record is [35,27,11,48].
Exactly one value in this observed record is wrong.
The following relationships are reliable:
a + d = 83
b - a = -8
c - a = -29
The downstream calculation is: max(a,b,c,d)-min(a,b,c,d).
Recover the correct record. Return [a,b,c,d], not the downstream answer.
```

## BR

基础场景：`2587437e7468366b845916e243e020b6`；输出顺序：`[0, 1, 2, 3]`

```text
SYSTEM
You are repairing a four-integer chart record. Return only one JSON array
containing four integers from 0 to 99. Do not include an explanation.

USER
The record order is [a,b,c,d]. For the chart, a and b are the two series
at the first x position; c and d are the same two series at the second position.
The observed record is [35,27,11,48].
Exactly one value in this observed record is wrong.
The following relationships are reliable:
c + d = 54
b + d = 75
a + d = 83
The downstream calculation is: max(a,b,c,d)-min(a,b,c,d).
Recover the correct record. Return [a,b,c,d], not the downstream answer.
```

## L00

基础场景：`2587437e7468366b845916e243e020b6`；输出顺序：`[0, 1, 2, 3]`

```text
SYSTEM
You are repairing a four-integer chart record. Return only one JSON array
containing four integers from 0 to 99. Do not include an explanation.

USER
The record order is [a,b,c,d]. For the chart, a and b are the two series
at the first x position; c and d are the same two series at the second position.
The observed record is a=35, b=27, c=11, d=48.
Exactly one value in this observed record is wrong.
The following relationships are reliable:
a + d = 83
b + d = 75
c + d = 54
The downstream calculation is: max(a,b,c,d)-min(a,b,c,d).
Recover the correct record. Return [a,b,c,d], not the downstream answer.
```

## L01

基础场景：`2587437e7468366b845916e243e020b6`；输出顺序：`[3, 2, 1, 0]`

```text
SYSTEM
You are repairing a four-integer chart record. Return only one JSON array
containing four integers from 0 to 99. Do not include an explanation.

USER
The record order is [a,b,c,d]. For the chart, a and b are the two series
at the first x position; c and d are the same two series at the second position.
The observed record is a=35, b=27, c=11, d=48.
Exactly one value in this observed record is wrong.
The following relationships are reliable:
a + d = 83
b + d = 75
c + d = 54
The downstream calculation is: max(a,b,c,d)-min(a,b,c,d).
Recover the correct record. Return [d,c,b,a], not the downstream answer.
```

## L10

基础场景：`2587437e7468366b845916e243e020b6`；输出顺序：`[0, 1, 2, 3]`

```text
SYSTEM
You are repairing a four-integer chart record. Return only one JSON array
containing four integers from 0 to 99. Do not include an explanation.

USER
The record order is [a,b,c,d]. For the chart, a and b are the two series
at the first x position; c and d are the same two series at the second position.
The observed record is d=48, c=11, b=27, a=35.
Exactly one value in this observed record is wrong.
The following relationships are reliable:
a + d = 83
b + d = 75
c + d = 54
The downstream calculation is: max(a,b,c,d)-min(a,b,c,d).
Recover the correct record. Return [a,b,c,d], not the downstream answer.
```

## L11

基础场景：`2587437e7468366b845916e243e020b6`；输出顺序：`[3, 2, 1, 0]`

```text
SYSTEM
You are repairing a four-integer chart record. Return only one JSON array
containing four integers from 0 to 99. Do not include an explanation.

USER
The record order is [a,b,c,d]. For the chart, a and b are the two series
at the first x position; c and d are the same two series at the second position.
The observed record is d=48, c=11, b=27, a=35.
Exactly one value in this observed record is wrong.
The following relationships are reliable:
a + d = 83
b + d = 75
c + d = 54
The downstream calculation is: max(a,b,c,d)-min(a,b,c,d).
Recover the correct record. Return [d,c,b,a], not the downstream answer.
```
