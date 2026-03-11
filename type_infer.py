#!/usr/bin/env python3
"""Type inference engine — Hindley-Milner style type inference."""
import sys

class TypeVar:
    _counter = 0
    def __init__(self, name=None):
        if name is None: TypeVar._counter += 1; name = f"t{TypeVar._counter}"
        self.name = name; self.instance = None
    def __repr__(self): return self.resolve().__repr__() if self.instance else self.name
    def resolve(self):
        if self.instance: return self.instance.resolve() if isinstance(self.instance, TypeVar) else self.instance
        return self

class FuncType:
    def __init__(self, arg, ret): self.arg = arg; self.ret = ret
    def __repr__(self): return f"({self.arg} -> {self.ret})"

class ConcreteType:
    def __init__(self, name): self.name = name
    def __repr__(self): return self.name

INT = ConcreteType("Int"); BOOL = ConcreteType("Bool"); STRING = ConcreteType("String")

def unify(t1, t2):
    t1 = t1.resolve() if isinstance(t1, TypeVar) else t1
    t2 = t2.resolve() if isinstance(t2, TypeVar) else t2
    if isinstance(t1, TypeVar): t1.instance = t2; return
    if isinstance(t2, TypeVar): t2.instance = t1; return
    if isinstance(t1, ConcreteType) and isinstance(t2, ConcreteType):
        if t1.name != t2.name: raise TypeError(f"Cannot unify {t1} and {t2}")
        return
    if isinstance(t1, FuncType) and isinstance(t2, FuncType):
        unify(t1.arg, t2.arg); unify(t1.ret, t2.ret); return
    raise TypeError(f"Cannot unify {t1} and {t2}")

def infer(expr, env=None):
    if env is None: env = {}
    if isinstance(expr, int): return INT
    if isinstance(expr, bool): return BOOL
    if isinstance(expr, str):
        if expr in env: return env[expr]
        raise NameError(f"Undefined: {expr}")
    if isinstance(expr, tuple):
        if expr[0] == 'lambda':
            param, body = expr[1], expr[2]
            param_type = TypeVar()
            new_env = {**env, param: param_type}
            body_type = infer(body, new_env)
            return FuncType(param_type, body_type)
        if expr[0] == 'apply':
            func_type = infer(expr[1], env)
            arg_type = infer(expr[2], env)
            ret_type = TypeVar()
            unify(func_type, FuncType(arg_type, ret_type))
            return ret_type.resolve() if isinstance(ret_type, TypeVar) else ret_type
    raise TypeError(f"Cannot infer type of {expr}")

if __name__ == "__main__":
    env = {"add": FuncType(INT, FuncType(INT, INT)), "not": FuncType(BOOL, BOOL),
           "eq": FuncType(INT, FuncType(INT, BOOL)), "x": INT, "flag": BOOL}
    tests = [42, True, "x", ("lambda", "y", "y"), ("apply", "not", True),
             ("apply", ("apply", "add", 1), 2)]
    print("Type Inference:")
    for expr in tests:
        try:
            t = infer(expr, env.copy())
            print(f"  {str(expr):>40s} : {t}")
        except (TypeError, NameError) as e:
            print(f"  {str(expr):>40s} : ERROR: {e}")
