# 一条题组—证据—评分—奖励的完整例子

以下是程序算例，不是模型输出。

| 类别 | Alpha | Beta |
|---|---:|---:|
| January | 30 | 20 |
| February | 42 | 50 |
| March | 26 | 14 |
| April | 17 | 21 |

问题：Alpha严格大于Beta的类别中，求Alpha数值之和。正确答案56。

核验这个条件需要两系列四类别的8个值，不能只报January/Alpha=30和March/Alpha=26。允许不同证据顺序，但不接受漏掉最后被排除的类别。

```json
{"evidence":[{"series":"Alpha","category":"January","value":30},{"series":"Beta","category":"January","value":20},{"series":"Alpha","category":"February","value":42},{"series":"Beta","category":"February","value":50},{"series":"Alpha","category":"March","value":26},{"series":"Beta","category":"March","value":14},{"series":"Alpha","category":"April","value":17},{"series":"Beta","category":"April","value":21}],"answer":56}
```

这个输出 E=P=A=J=C=1。把answer改57，则E=P=1，A=J=C=0。把Alpha/January读为31、答案保持56，则E=1、p_read=7/8、P=0、A=1、J=0；C按错误报告值执行原查询判断，不是内部推理标注。

把February/Alpha改60，答案116；把它改50，再问严格大于/至少同样大，答案分别56/106；求均值时原题答案28。它们分别对应相关证据变化、语言条件边界、运算变化。

## 一组8个回答

- 2条完整正确：J=1,E=1,p=1。
- 3条未完整正确但证据对象正确、半数值正确：J=0,E=1,p=.5。
- 3条证据范围错误：J=0,E=0,p=0。

纯J的组优势约为 `(1.732,1.732,-.577,-.577,-.577,-.577,-.577,-.577)`。

GATE有界精化保持两条成功的优势不变，使中间3条约为−.433，末3条约为−.722。失败仍为负，组和为0。

这只说明新反馈提供了不同信用，**不证明参数更新一定更好**。测试套件含一个同样组构成的单参数反例：纯J的一阶效果为正，精化的一阶效果为负。这是保留真实学习对照的理由，而不是要求删除候选。

本目录的36张例图来自已生成清单的4个根；联系图展示8个原题图。它们还没有经过Qwen的native processor，不代表全部2,890张不同原图已被渲染或核验。
