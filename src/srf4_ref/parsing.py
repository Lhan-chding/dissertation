"""Safe diagnostic JSON extraction and arithmetic checking. NOT a complete work parser."""
from __future__ import annotations
import ast
import json
import math
from fractions import Fraction

def _pairs(pairs):
    d={}
    for k,v in pairs:
        if k in d: raise ValueError('Duplicate JSON key')
        d[k]=v
    return d

def extract_object(text: str) -> dict:
    # May recover prose-wrapped JSON diagnostically; never upgrades official protocol_valid.
    decoder=json.JSONDecoder(object_pairs_hook=_pairs,
          parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    starts=[i for i,c in enumerate(text) if c=='{']
    for start in starts:
        try:
            obj,n=decoder.raw_decode(text,start)
        except (ValueError,json.JSONDecodeError): continue
        if isinstance(obj,dict) and 'answer' in obj:
            return {'object':obj,'start':start,'end':n,'prefix':text[:start],
                    'suffix':text[n:],'strict_document':not text[:start].strip() and not text[n:].strip()}
    return {'object':None,'start':None,'end':None,'strict_document':False,'reason':'NO_SAFE_OBJECT'}

def rational(value) -> Fraction:
    if isinstance(value,bool) or value is None: raise ValueError('Not a number')
    if isinstance(value,float) and not math.isfinite(value): raise ValueError('Nonfinite')
    return Fraction(str(value))

def arithmetic(expression: str) -> Fraction:
    if len(expression)>512: raise ValueError('Expression too long')
    tree=ast.parse(expression,mode='eval')
    def visit(n,depth=0):
        if depth>20: raise ValueError('Expression too deep')
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):
            v=rational(n.value)
            if abs(v.numerator).bit_length()>256 or v.denominator.bit_length()>256: raise ValueError('Number too large')
            return v
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd)):
            v=visit(n.operand,depth+1); return -v if isinstance(n.op,ast.USub) else v
        if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
            a,b=visit(n.left,depth+1),visit(n.right,depth+1)
            if isinstance(n.op,ast.Add): return a+b
            if isinstance(n.op,ast.Sub): return a-b
            if isinstance(n.op,ast.Mult): return a*b
            if b==0: raise ValueError('Division by zero')
            return a/b
        raise ValueError('Unsupported expression: no eval, names, calls or powers')
    return visit(tree.body)

def equation_holds(lhs: str, rhs: str) -> bool:
    return arithmetic(lhs)==arithmetic(rhs)
