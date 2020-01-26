import torch
from functools import reduce

##########################################
# Primitives -- Basic Fuzzy Logic (BL)
##########################################

def neg(f):
    val = 1.0 - f
    return val


def prod_tnorm(fs):
    return reduce(lambda f1, f2: f1*f2, fs)


def prod_tconorm(fs):
    return reduce(lambda f1, f2: f1 + f2 - f1*f2, fs)


def iff(f1, f2):
    return prod_tconorm((prod_tnorm((f1, f2)), prod_tnorm((neg(f1), neg(f2)))))


def ite(c1, f1, f2):
    return (f1*c1) + f2*(1-c1)


def implies(f1, f2):
    return prod_tconorm((neg(f1), f2))


##########################################
# Primitives -- LIRA
##########################################
DEFAULT_B = 3.0
DEFAULT_EPS = 0.5


def gt(x, B=DEFAULT_B, eps=DEFAULT_EPS):
    return torch.sigmoid(B*(x - eps))


def ge(x, B=DEFAULT_B, eps=DEFAULT_EPS):
    return torch.sigmoid(B*(x + eps))


def lt(x, B=None, eps=DEFAULT_EPS):
    return neg(ge(x, B, eps))


def le(x, B=None, eps=DEFAULT_EPS):
    return neg(gt(x, B, eps))


def eq(x, B=None, eps=DEFAULT_EPS):
    val = prod_tnorm((ge(x, B, eps), le(x, B, eps)))
    return val


def neq(x, B=None, eps=DEFAULT_EPS):
    return neg(eq(x, B, eps))


##########################################
# CLN Base Class: all operations should inherit
##########################################

class ClnOp(object):#torch.nn.Module):
    def __init__(self):
        super(ClnOp, self).__init__()

    def subclauses(self):
        raise NotImplementedError()

    def print(self, indent=0):
        if getattr(self, 'B', None) is not None:
            B_param = " B(%s, %s)" % (str(self.B.data), str(self.B.grad))
        else:
            B_param = ""
        if isinstance(self.val, Interval):
            print("\t" * indent, type(self).__name__, self.val, self.val.m.grad, B_param)
        else:
            print("\t" * indent, type(self).__name__, self.val.data, self.val.grad, B_param)
        for c in self.subclauses():
            c.print(indent + 1)

    def to_z3(self, variables):
        raise NotImplementedError()

    # TODO optional retain grad function, shouldn't always be applied


##########################################
# Network -- Basic Fuzzy Logic (BL)
##########################################

class Boolean(ClnOp):
    def __init__(self, name):
        super(Boolean, self).__init__()
        self.name = name

    def forward(self, xs):
        reals, bools = xs
        self.val = bools[self.name]
        self.val.retain_grad()
        return self.val

    def subclauses(self):
        return []

    def to_z3(self, variables):
        return variables[self.name]


class Conjunction(ClnOp,torch.nn.Module):
    def __init__(self, fs, tnorm = prod_tnorm):
        super(Conjunction, self).__init__()
        # Allows Pretty Printing of Module and torch.nn.parameters to register
        self.fs = fs
        for idx, i in enumerate(fs):
            setattr(self, "f%s" % (str(idx)), i)
        self.tnorm = tnorm

    def forward(self, xs):
        results = []
        for f in self.fs:
            results.append(f.forward(xs))
        self.val = self.tnorm(results)
        self.val.retain_grad()
        return self.val

    def subclauses(self):
        return self.fs

    def to_z3(self, variables):
        return z3.And([f.to_z3(variables) for f in self.fs])


class Disjunction(ClnOp,torch.nn.Module):
    def __init__(self, fs, tconorm = prod_tconorm):
        super(Disjunction, self).__init__()
        self.fs = fs
        for idx, i in enumerate(fs):
            setattr(self, "f%s" % (str(idx)), i)
        self.tconorm = tconorm

    def forward(self, xs):
        results = []
        for f in self.fs:
            results.append(f.forward(xs))
        self.val = self.tconorm(results)
        self.val.retain_grad()
        return self.val

    def subclauses(self):
        return self.fs

    def to_z3(self, variables):
        return z3.Or([f.to_z3(variables) for f in self.fs])


class Neg(ClnOp,torch.nn.Module):
    def __init__(self, f):
        super(Neg, self).__init__()
        self.f = f

    def forward(self, xs):
        self.val = neg(self.f.forward(xs))
        self.val.retain_grad()
        return self.val

    def subclauses(self):
        return [self.f]

    def to_z3(self, variables):
        return z3.Not(self.f.to_z3(variables))


class Iff(ClnOp,torch.nn.Module):
    def __init__(self, f1, f2):
        super(Iff, self).__init__()
        self.f1 = f1
        self.f2 = f2

    def forward(self, xs):
        self.val = iff(self.f1.forward(xs), self.f2.forward(xs))
        self.val.retain_grad()
        return self.val

    def subclauses(self):
        return self.f1, self.f2

    def to_z3(self, variables):
        f1_z3 = self.f1.to_z3(variables)
        f2_z3 = self.f2.to_z3(variables)
        return z3.Or(z3.And(f1_z3, f2_z3), z3.And(z3.Not(f1_z3), z3.Not(f2_z3)))


class Ite(ClnOp,torch.nn.Module):
    def __init__(self, c, f1, f2):
        super(Ite, self).__init__()
        self.c = c
        self.f1 = f1
        self.f2 = f2

    def forward(self, xs):
        self.val = ite(self.c.forward(xs), self.f1.forward(xs), self.f2.forward(xs))
        self.val.retain_grad()
        return self.val

    def subclauses(self):
        return self.c, self.f1, self.f2

    def to_z3(self, variables):
        c = self.c.to_z3(variables)
        f1 = self.f1.to_z3(variables)
        f2 = self.f2.to_z3(variables)
        return z3.If(c, f1, f2)


class Implies(ClnOp,torch.nn.Module):
    def __init__(self, f1, f2):
        super(Implies, self).__init__()
        self.f1 = f1
        self.f2 = f2

    def forward(self, xs):
        self.val = implies(self.f1.forward(xs), self.f2.forward(xs))
        self.val.retain_grad()
        return self.val

    def subclauses(self):
        return self.f1, self.f2

    def to_z3(self, variables):
        f1_z3 = self.f1.to_z3(variables)
        f2_z3 = self.f2.to_z3(variables)
        return z3.Implies(f1_z3, f2_z3)

##########################################
# Network -- LIRA
##########################################


class Constant(ClnOp):
    def __init__(self, value):
        super(Constant, self).__init__()
        self.value = torch.tensor(value)

    def forward(self, xs):
        self.val = self.value
        self.val.retain_grad()
        return self.val

    def subclauses(self):
        return []

    def to_z3(self, variables):
        return self.value.item()


class BooleanConstant(ClnOp):
    def __init__(self, value):
        super(BooleanConstant, self).__init__()
        if value is True:
            self.value = torch.tensor(1.0)
        elif value is False:
            self.value = torch.tensor(0.0)
        else:
            raise Exception("value not True or False")
        self.bool = value

    def forward(self, xs):
        self.val = self.value
        self.val.retain_grad()
        return self.val

    def subclauses(self):
        return []

    def to_z3(self, variables):
        return self.bool


class Real(ClnOp):
    def __init__(self, name):
        super(Real, self).__init__()
        self.name = name

    def forward(self, xs):
        # type: (Tuple[Dict[str,Tensor], Dict[str, Tensor]]) -> Tensor
        reals, bools = xs 
        self.val = reals[self.name]
        self.val.retain_grad()
        self.local = self.val * 1.0
        self.local.retain_grad()
        return self.local

    def subclauses(self):
        return []
    
    def print(self, indent=0):
        if isinstance(self.val, Interval):
            print("\t" * indent, "LocalVar", self.name, self.local.m.data, self.local.m.grad)
            print("\t" * (indent+1), type(self).__name__, self.name, self.val.m.data, self.val.m.grad)
        else:
            print("\t" * indent, "LocalVar", self.name, self.local.data, self.local.grad)
            print("\t" * (indent+1), type(self).__name__, self.name, self.val.data, self.val.grad)

    def to_z3(self, variables):
        return variables[self.name]


class Linear(ClnOp):
    def __init__(self, ws):
        super(Linear, self).__init__()
        self.ws = ws

    def forward(self, xs):
        self.val = reduce(lambda x1, x2: x1+x2, [w*x for w, x in zip(self.ws, xs)])
        self.val.retain_grad()
        return self.val

    def subclauses(self):
        return []


class Plus(ClnOp):
    def __init__(self, fs):
        super(Plus, self).__init__()
        self.fs = fs
        for idx, i in enumerate(fs):
            setattr(self, "f%s" % (str(idx)), i)

    def forward(self, xs):
        # type: (Tuple[Dict[str,Tensor], Dict[str, Tensor]]) -> Tensor
        self.val = reduce(lambda x1, x2: x1+x2, [f.forward(xs) for f in self.fs])
        self.val.retain_grad()
        return self.val

    def subclauses(self):
        return self.fs

    def to_z3(self, variables):
        return reduce(op.add, [f.to_z3(variables) for f in self.fs])


class Minus(ClnOp):
    def __init__(self, f1, f2):
        super(Minus, self).__init__()
        self.f1 = f1
        self.f2 = f2

    def forward(self, xs):
        # type: (Tuple[Dict[str,Tensor], Dict[str, Tensor]]) -> Tensor
        self.val = self.f1.forward(xs) - self.f2.forward(xs)
        self.val.retain_grad()
        return self.val

    def subclauses(self):
        return self.f1, self.f2

    def to_z3(self, variables):
        f1 = self.f1.to_z3(variables)
        f2 = self.f2.to_z3(variables)
        return f1 - f2


class Mul(ClnOp):
    def __init__(self, fs):
        super(Mul, self).__init__()
        self.fs = fs
        for idx, i in enumerate(fs):
            setattr(self, "f%s" % (str(idx)), i)

    def forward(self, xs):
        # type: (Tuple[Dict[str,Tensor], Dict[str, Tensor]]) -> Tensor
        self.val = reduce(lambda x1, x2: x1*x2, [f.forward(xs) for f in self.fs])
        self.val.retain_grad()
        return self.val

    def subclauses(self):
        return self.fs

    def to_z3(self, variables):
        return reduce(op.mul, [f.to_z3(variables) for f in self.fs])


class Gt(ClnOp,torch.nn.Module):
    def __init__(self, expr, B=None, eps=DEFAULT_EPS):
        super(Gt, self).__init__()
        self.expr = expr
        self.B = B
        if B is not None:
            self.B = torch.nn.Parameter(torch.tensor(B, requires_grad=True), requires_grad=True)
        self.eps = eps

    def forward(self, xs):
        expr = self.expr.forward(xs)
        self.val = gt(expr, self.B, self.eps)
        self.val.retain_grad()
        return self.val

    def subclauses(self):
        return [self.expr]

    def to_z3(self, variables):
        return self.expr.to_z3(variables) > 0


class Ge(ClnOp,torch.nn.Module):
    def __init__(self, expr, B=None, eps=DEFAULT_EPS):
        super(Ge, self).__init__()
        self.expr = expr
        self.B = B
        if B is not None:
            self.B = torch.nn.Parameter(torch.tensor(B, requires_grad=True), requires_grad=True)
        self.eps = eps

    def forward(self, xs):
        expr = self.expr.forward(xs)
        self.val = ge(expr, self.B, self.eps)
        self.val.retain_grad()
        return self.val

    def subclauses(self):
        return [self.expr]

    def to_z3(self, variables):
        return self.expr.to_z3(variables) >= 0


class Lt(ClnOp,torch.nn.Module):
    def __init__(self, expr, B=None, eps=DEFAULT_EPS):
        super(Lt, self).__init__()
        self.expr = expr
        self.B = B
        if B is not None:
            self.B = torch.nn.Parameter(torch.tensor(B, requires_grad=True), requires_grad=True)
        self.eps = eps

    def forward(self, xs):
        expr = self.expr.forward(xs)
        self.val = lt(expr, self.B, self.eps)
        self.val.retain_grad()
        return self.val

    def subclauses(self):
        return [self.expr]

    def to_z3(self, variables):
        return self.expr.to_z3(variables) < 0


class Le(ClnOp,torch.nn.Module):
    def __init__(self, expr, B=None, eps=DEFAULT_EPS):
        super(Le, self).__init__()
        self.expr = expr
        self.B = B
        if B is not None:
            self.B = torch.nn.Parameter(torch.tensor(B, requires_grad=True), requires_grad=True)
        self.eps = eps

    def forward(self, xs):
        expr = self.expr.forward(xs)
        self.val = le(expr, self.B, self.eps)
        self.val.retain_grad()
        return self.val

    def subclauses(self):
        return [self.expr]

    def to_z3(self, variables):
        return self.expr.to_z3(variables) <= 0


class Eq(ClnOp, torch.nn.Module):
    def __init__(self, expr, B=None, eps=DEFAULT_EPS):
        super(Eq, self).__init__()
        self.expr = expr
        self.B = B
        if B is not None:
            self.B = torch.nn.Parameter(torch.tensor(B, requires_grad=True), requires_grad=True)
        self.eps = eps

    def forward(self, xs):
        expr = self.expr.forward(xs)
        self.val = eq(expr, self.B, self.eps)
        self.val.retain_grad()
        return self.val

    def subclauses(self):
        return [self.expr]

    def to_z3(self, variables):
        return self.expr.to_z3(variables) == 0


class Neq(ClnOp, torch.nn.Module):
    def __init__(self, expr, B=None, eps=DEFAULT_EPS):
        super(Neq, self).__init__()
        self.expr = expr
        self.B = B
        if B is not None:
            self.B = torch.nn.Parameter(torch.tensor(B, requires_grad=True), requires_grad=True)
        self.eps = eps

    def forward(self, xs):
        expr = self.expr.forward(xs)
        self.val = neq(expr, self.B, self.eps)
        self.val.retain_grad()
        return self.val

    def subclauses(self):
        return [self.expr]

    def to_z3(self, variables):
        return self.expr.to_z3(variables) != 0


