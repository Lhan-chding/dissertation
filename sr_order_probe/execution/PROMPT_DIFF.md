# Frozen public prompt differences
Example qid: SMOKE_ORDER_0000_view1

## C0 -> C_NOLIST
```diff
--- C0
+++ C_NOLIST
@@ -10,6 +10,6 @@
 "answer": the final exact result: an integer, an exact finite decimal, or a quoted fraction such as "17/3". Do not round repeating decimals or write unquoted fractions. Put answer last. It is the sole final answer; numbers in work do not replace it.
 The examples illustrate types only. Use actual labels and values from THIS chart, never the Example_A/Example_B placeholders. Do not add extra fields. Evidence and bindings serve different roles: a number written only in bindings does not count as an evidence entry. Do not use external tools.
 EVIDENCE CHECKLIST FOR THIS QUESTION:
-- Ranking: report the original Alpha value for EVERY category in this public list: ["Frame", "Watch", "Clock", "Bowl", "Vase", "Stool", "Shelf", "Spoon", "Desk", "Book"].
+- Ranking: report the original Alpha value for EVERY category shown on the chart.
 - Operand: report the original Beta value of each finally selected category in evidence.
 - A cell already reported for ranking or filtering is reported only once. Values written only in bindings or work do not count as evidence.
```

## C0 -> C_ENUM
```diff
--- C0
+++ C_ENUM
@@ -10,6 +10,6 @@
 "answer": the final exact result: an integer, an exact finite decimal, or a quoted fraction such as "17/3". Do not round repeating decimals or write unquoted fractions. Put answer last. It is the sole final answer; numbers in work do not replace it.
 The examples illustrate types only. Use actual labels and values from THIS chart, never the Example_A/Example_B placeholders. Do not add extra fields. Evidence and bindings serve different roles: a number written only in bindings does not count as an evidence entry. Do not use external tools.
 EVIDENCE CHECKLIST FOR THIS QUESTION:
-- Ranking: report the original Alpha value for EVERY category in this public list: ["Frame", "Watch", "Clock", "Bowl", "Vase", "Stool", "Shelf", "Spoon", "Desk", "Book"].
+- Ranking: report the original Alpha value for EVERY category in this public list: ["Frame", "Book", "Stool", "Spoon", "Bowl", "Watch", "Vase", "Clock", "Shelf", "Desk"].
 - Operand: report the original Beta value of each finally selected category in evidence.
 - A cell already reported for ranking or filtering is reported only once. Values written only in bindings or work do not count as evidence.
```

## C0 -> C_REF
```diff
--- C0
+++ C_REF
@@ -1,4 +1,4 @@
-Select the 3 categories with the largest Alpha values. Compute the mean of Beta for the final selected categories. All chart quantities are integer counts in [10,90]. Break ranking ties by printed left-to-right order.
+Select the 3 categories with the largest Alpha values. Compute the mean of Beta for the final selected categories. All chart quantities are integer counts in [10,90]. Break ranking ties by the category positions on the chart horizontal axis, from left to right. The order in which you write evidence does not change these chart positions.
 Return exactly one JSON object, with no Markdown fences or text outside it.
 Write every key exactly once in each object.
 Use these fields in the following suggested order; work is optional:

```

## C0 -> C_PRINT
```diff
--- C0
+++ C_PRINT
@@ -12,4 +12,5 @@
 EVIDENCE CHECKLIST FOR THIS QUESTION:
 - Ranking: report the original Alpha value for EVERY category in this public list: ["Frame", "Watch", "Clock", "Bowl", "Vase", "Stool", "Shelf", "Spoon", "Desk", "Book"].
 - Operand: report the original Beta value of each finally selected category in evidence.
-- A cell already reported for ranking or filtering is reported only once. Values written only in bindings or work do not count as evidence.+- A cell already reported for ranking or filtering is reported only once. Values written only in bindings or work do not count as evidence.
+For the Alpha evidence entries, list categories in printed left-to-right order. Keep the question's tie-breaking rule unchanged.
```

## C0 -> C_ALT
```diff
--- C0
+++ C_ALT
@@ -12,4 +12,5 @@
 EVIDENCE CHECKLIST FOR THIS QUESTION:
 - Ranking: report the original Alpha value for EVERY category in this public list: ["Frame", "Watch", "Clock", "Bowl", "Vase", "Stool", "Shelf", "Spoon", "Desk", "Book"].
 - Operand: report the original Beta value of each finally selected category in evidence.
-- A cell already reported for ranking or filtering is reported only once. Values written only in bindings or work do not count as evidence.+- A cell already reported for ranking or filtering is reported only once. Values written only in bindings or work do not count as evidence.
+For the Alpha evidence entries, list categories in alphabetical order by category label. Keep the question's tie-breaking rule unchanged.
```

Presentation trims trailing whitespace; canonical prompt bytes remain in frozen data.
