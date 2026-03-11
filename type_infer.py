#!/usr/bin/env python3
"""type_infer.py — Hindley-Milner type inference with Algorithm W.

Implements complete HM type inference: type variables, function types,
unification, generalization, instantiation, and let-polymorphism.

One file. Zero deps. Does one thing well.
"""

import sys
from dataclasses import dataclass


# ─── Types ───

@dataclass(frozen=True)
class TVar:
    name: str
    def __repr__(self): return self.name

@dataclass(frozen=True)
class TCon:
    name: str
    def __repr__(self): return self.name

@dataclass(frozen=True)
class TFun:
    arg: object
    ret: object
    def __repr__(self): return f"({self.arg} → {self.ret})"

@dataclass(frozen=True)
class TList:
    elem: object
    def __repr__(self): return f"[{self.elem}]"

@dataclass(frozen=True)
class TTuple:
    elems: tuple
    def __repr__(self): return f"({', '.join(repr(e) for e in self.elems)})"


# ─── AST ───

@dataclass
class Var:
    name: str

@dataclass
class Lit:
    value: object
    type: str = 'int'

@dataclass
class App:
    func: object
    arg: object

@dataclass
class Lam:
    param: str
    body: object

@dataclass
class Let:
    name: str
    value: object
    body: object

@dataclass
class If:
    cond: object
    then: object
    else_: object


# ─── Type Inference ───

class TypeInfer:
    def __init__(self):
        self.supply = 0
        self.subst: dict[str, object] = {}

    def fresh(self) -> TVar:
        self.supply += 1
        return TVar(f"t{self.supply}")

    def apply(self, t):
        if isinstance(t, TVar):
            if t.name in self.subst:
                return self.apply(self.subst[t.name])
            return t
        if isinstance(t, TFun):
            return TFun(self.apply(t.arg), self.apply(t.ret))
        if isinstance(t, TList):
            return TList(self.apply(t.elem))
        if isinstance(t, TTuple):
            return TTuple(tuple(self.apply(e) for e in t.elems))
        return t

    def occurs(self, name: str, t) -> bool:
        t = self.apply(t)
        if isinstance(t, TVar):
            return t.name == name
        if isinstance(t, TFun):
            return self.occurs(name, t.arg) or self.occurs(name, t.ret)
        if isinstance(t, TList):
            return self.occurs(name, t.elem)
        if isinstance(t, TTuple):
            return any(self.occurs(name, e) for e in t.elems)
        return False

    def unify(self, t1, t2):
        t1, t2 = self.apply(t1), self.apply(t2)
        if isinstance(t1, TVar):
            if t1 == t2:
                return
            if self.occurs(t1.name, t2):
                raise TypeError(f"Infinite type: {t1} ~ {t2}")
            self.subst[t1.name] = t2
        elif isinstance(t2, TVar):
            self.unify(t2, t1)
        elif isinstance(t1, TCon) and isinstance(t2, TCon):
            if t1.name != t2.name:
                raise TypeError(f"Type mismatch: {t1} vs {t2}")
        elif isinstance(t1, TFun) and isinstance(t2, TFun):
            self.unify(t1.arg, t2.arg)
            self.unify(t1.ret, t2.ret)
        elif isinstance(t1, TList) and isinstance(t2, TList):
            self.unify(t1.elem, t2.elem)
        else:
            raise TypeError(f"Cannot unify {t1} with {t2}")

    def free_vars(self, t) -> set[str]:
        t = self.apply(t)
        if isinstance(t, TVar):
            return {t.name}
        if isinstance(t, TFun):
            return self.free_vars(t.arg) | self.free_vars(t.ret)
        if isinstance(t, TList):
            return self.free_vars(t.elem)
        if isinstance(t, TTuple):
            return set().union(*(self.free_vars(e) for e in t.elems))
        return set()

    def env_free_vars(self, env: dict) -> set[str]:
        result = set()
        for _, (t, _) in env.items():
            result |= self.free_vars(t)
        return result

    def generalize(self, env: dict, t) -> tuple:
        t = self.apply(t)
        free = self.free_vars(t) - self.env_free_vars(env)
        return (t, frozenset(free))

    def instantiate(self, scheme: tuple):
        t, quantified = scheme
        mapping = {v: self.fresh() for v in quantified}
        return self._subst_scheme(t, mapping)

    def _subst_scheme(self, t, mapping):
        if isinstance(t, TVar):
            return mapping.get(t.name, t)
        if isinstance(t, TFun):
            return TFun(self._subst_scheme(t.arg, mapping), self._subst_scheme(t.ret, mapping))
        if isinstance(t, TList):
            return TList(self._subst_scheme(t.elem, mapping))
        return t

    def infer(self, expr, env: dict):
        if isinstance(expr, Lit):
            return TCon({'int': 'Int', 'float': 'Float', 'str': 'String', 'bool': 'Bool'}[expr.type])

        if isinstance(expr, Var):
            if expr.name not in env:
                raise TypeError(f"Unbound variable: {expr.name}")
            return self.instantiate(env[expr.name])

        if isinstance(expr, Lam):
            tv = self.fresh()
            new_env = {**env, expr.param: (tv, frozenset())}
            ret = self.infer(expr.body, new_env)
            return TFun(self.apply(tv), ret)

        if isinstance(expr, App):
            fun_t = self.infer(expr.func, env)
            arg_t = self.infer(expr.arg, env)
            ret = self.fresh()
            self.unify(fun_t, TFun(arg_t, ret))
            return self.apply(ret)

        if isinstance(expr, Let):
            val_t = self.infer(expr.value, env)
            scheme = self.generalize(env, val_t)
            new_env = {**env, expr.name: scheme}
            return self.infer(expr.body, new_env)

        if isinstance(expr, If):
            cond_t = self.infer(expr.cond, env)
            self.unify(cond_t, TCon('Bool'))
            then_t = self.infer(expr.then, env)
            else_t = self.infer(expr.else_, env)
            self.unify(then_t, else_t)
            return self.apply(then_t)

        raise TypeError(f"Unknown expression: {type(expr)}")

    def typeof(self, expr, env=None) -> str:
        if env is None:
            env = {}
        t = self.infer(expr, env)
        return repr(self.apply(t))


def demo():
    print("=== Hindley-Milner Type Inference ===\n")
    ti = TypeInfer()

    examples = [
        ("literal int", Lit(42)),
        ("identity λx.x", Lam('x', Var('x'))),
        ("const λx.λy.x", Lam('x', Lam('y', Var('x')))),
        ("apply id to 5", App(Lam('x', Var('x')), Lit(5))),
        ("let-poly", Let('id', Lam('x', Var('x')),
                         Let('a', App(Var('id'), Lit(1)),
                             App(Var('id'), Lit(True, 'bool'))))),
    ]

    for name, expr in examples:
        ti2 = TypeInfer()
        try:
            t = ti2.typeof(expr)
            print(f"  {name:20s} : {t}")
        except TypeError as e:
            print(f"  {name:20s} : ERROR: {e}")

    # Error cases
    print("\nError cases:")
    ti3 = TypeInfer()
    try:
        ti3.typeof(App(Lit(5), Lit(3)))
    except TypeError as e:
        print(f"  apply int to int   : {e}")

    ti4 = TypeInfer()
    try:
        ti4.typeof(Var('unbound'))
    except TypeError as e:
        print(f"  unbound variable   : {e}")


if __name__ == '__main__':
    if '--test' in sys.argv:
        ti = TypeInfer()
        # Int literal
        assert ti.typeof(Lit(1)) == 'Int'
        # Identity
        ti = TypeInfer()
        t = ti.typeof(Lam('x', Var('x')))
        assert '→' in t  # (t1 → t1)
        # Application
        ti = TypeInfer()
        assert ti.typeof(App(Lam('x', Var('x')), Lit(5))) == 'Int'
        # Let polymorphism
        ti = TypeInfer()
        t = ti.typeof(Let('id', Lam('x', Var('x')),
                          Let('a', App(Var('id'), Lit(1)),
                              App(Var('id'), Lit(True, 'bool')))))
        assert t == 'Bool'
        # Type error
        ti = TypeInfer()
        try:
            ti.typeof(App(Lit(5), Lit(3)))
            assert False
        except TypeError:
            pass
        print("All tests passed ✓")
    else:
        demo()
