"""Differential fuzz of the DSL optimizer (cd2_schema.py: ConstantFolder, NonZeroPromoter, RedundantSelectorRemover)
against a reference evaluator written from CD2's semantics (schema/REFERENCE.md). Random trees with opaque leaves are
optimized, then both sides are evaluated in random environments; a difference is printed with the tree.
NaN and fmod at its discontinuity are don't-care (CD2's float32 decides those, not the folder).

    python fuzz_fold.py [SEED]      4000 trees; expect at most the documented reassociation case"""
import copy, math, os, random, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cd2_schema as cs

def raw(x):
    return x.content if isinstance(x, cs.Expression) else x

def ev(x, env):
    x = raw(x)
    if isinstance(x, bool) or isinstance(x, (int, float)):
        return x
    if isinstance(x, list):
        raise ValueError('bare list')
    m = x.get('Mutate')
    if 'Var' in x and m is None:
        return env['vars'][x['Var']]
    ops = {k: v for k, v in x.items() if k != 'Mutate'}
    def many():
        vals = [ev(v, env) for k, v in ops.items() if k != 'Values']
        vals += [ev(v, env) for v in ops.get('Values', [])]
        if not vals: raise ValueError('no operands')
        return vals
    f = lambda k: ev(ops[k], env)
    if m == 'Opaque': return env['opaque'][x['Id']]
    if m == 'Add': return sum(many())
    if m == 'Multiply': return math.prod(many())
    if m == 'Max': return max(many())
    if m == 'Min': return min(many())
    if m == 'Subtract': return f('A') - f('B')
    if m == 'Divide':
        b = f('B'); return 0.0 if b == 0 else f('A') / b
    if m == 'Pow':
        try:
            r = math.pow(f('A'), f('B'))
        except (ValueError, OverflowError):
            raise ValueError('nan')
        if not math.isfinite(r): raise ValueError('nan')
        return r
    if m == 'Modulo':
        b = f('B'); a = f('A')
        if b != 0 and abs(a / b) > 1e6: raise ValueError('chaotic fmod')
        if b == 0: return 0.0
        r = math.fmod(a, b)
        if min(abs(r), abs(abs(r) - abs(b))) < 1e-6 * max(1.0, abs(b)): raise ValueError('fmod at its discontinuity')
        return r
    if m == 'Round': return math.floor(f('Value') + 0.5)
    if m == 'Ceil': return math.ceil(f('Value'))
    if m == 'Floor': return math.floor(f('Value'))
    if m == 'Abs': return abs(f('Value'))
    if m == 'And': return all(many())
    if m == 'Or': return any(many())
    if m == 'Not': return not f('Value')
    if m == 'NonZero': return f('Value') != 0
    if m == 'If': return f('Then') if f('Condition') else f('Else')
    if m == 'IfFloat':
        v = f('Value')
        for op, fn in (('==', lambda a, b: a == b), ('>=', lambda a, b: a >= b), ('>', lambda a, b: a > b),
                       ('<=', lambda a, b: a <= b), ('<', lambda a, b: a < b)):
            if op in ops: return f('Then') if fn(v, f(op)) else f('Else')
        raise ValueError('IfFloat without op')
    if m == 'ByPlayerCount':
        vals = ops['Values']; return ev(vals[min(env['players'] - 1, len(vals) - 1)], env)
    if m == 'ByBiome':
        k = env['biome']; return ev(ops[k], env) if k in ops else f('Default')
    raise ValueError('unknown ' + str(m))

R = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
LEAVES = [0, 1, 2, 0.5, 1.5, 2.5, -1.5, -3, 1e-7, 0.1234567, 3.0]
def num(d):
    if d <= 0 or R.random() < 0.3:
        r = R.random()
        if r < 0.15: return {'Mutate': 'Opaque', 'Id': R.randrange(3)}
        if r < 0.25: return {'Var': 'C'}
        return R.choice(LEAVES)
    op = R.choice(['Add', 'Multiply', 'Max', 'Min', 'Subtract', 'Divide', 'Pow', 'Modulo', 'Round', 'Ceil',
                   'Floor', 'If', 'IfFloat', 'ByPlayerCount', 'ByBiome', 'AddValues'])
    if op in ('Add', 'Multiply', 'Max', 'Min'):
        return dict({'Mutate': op}, **{k: num(d - 1) for k in 'ABC'[:R.randint(1, 3)]})
    if op == 'AddValues':
        return {'Mutate': R.choice(['Add', 'Multiply', 'Max', 'Min']), 'Values': [num(d - 1) for _ in range(R.randint(1, 3))]}
    if op in ('Subtract', 'Divide', 'Pow', 'Modulo'):
        return {'Mutate': op, 'A': num(d - 1), 'B': num(d - 1)}
    if op in ('Round', 'Ceil', 'Floor'):
        return {'Mutate': op, 'Value': num(d - 1)}
    if op == 'If':
        cond = R.choice([True, False, {'Mutate': 'Not', 'Value': R.choice([True, False])},
                         {'Mutate': 'IfFloat', 'Value': num(d - 1), '==': R.choice([0, 1]), 'Then': R.choice([True, False]), 'Else': R.choice([True, False])},
                         {'Mutate': 'NonZero', 'Value': num(d - 1)}])
        return {'Mutate': 'If', 'Condition': cond,
                'Then': num(d - 1), 'Else': num(d - 1)}
    if op == 'IfFloat':
        return {'Mutate': 'IfFloat', 'Value': num(d - 1), R.choice(['==', '>=', '>', '<=', '<']): num(d - 1),
                'Then': num(d - 1), 'Else': num(d - 1)}
    if op == 'ByPlayerCount':
        return {'Mutate': 'ByPlayerCount', 'Values': [num(d - 1) for _ in range(R.randint(1, 4))]}
    return {'Mutate': 'ByBiome', 'Default': num(d - 1), 'MagmaCore': num(d - 1), 'SaltPits': num(d - 1)}

def same(a, b):
    if isinstance(a, float) and math.isnan(a): return isinstance(b, float) and math.isnan(b)
    if isinstance(a, bool) or isinstance(b, bool): return a == b
    return abs(a - b) <= 1e-4 * max(1.0, abs(a), abs(b))

folder = cs.ConstantFolder(); nz = cs.NonZeroPromoter()
remover = cs.RedundantSelectorRemover()
found = {}
for i in range(4000):
    e = num(4)
    try:
        o = remover.optimize(nz.optimize(folder.optimize(copy.deepcopy(e))))
    except Exception as ex:
        found.setdefault('crash ' + type(ex).__name__ + ': ' + str(ex)[:60], (e, None)); continue
    for _ in range(4):
        env = {'vars': {'C': 2.0}, 'opaque': [R.choice([0, 1, 2.5, -2, 7]) for _ in range(3)],
               'players': R.randint(1, 5), 'biome': R.choice(['MagmaCore', 'SaltPits', 'AzureWeald'])}
        try:
            a = ev(e, env)
        except Exception:
            break
        try:
            b = ev(o, env)
        except Exception as ex:
            found.setdefault('optimized invalid: ' + str(ex)[:50], (e, o)); break
        if isinstance(b, complex) or not same(a, b):
            key = 'wrong value'
            found.setdefault(key + ' #%d' % len([k for k in found if k.startswith(key)]), (e, o, a, b)) if len([k for k in found if k.startswith(key)]) < 6 else None
            break
for k, v in found.items():
    print('==', k)
    print('   in :', json.dumps(raw(v[0]), default=lambda z: raw(z))[:300])
    if v[1] is not None: print('   out:', json.dumps(raw(v[1]), default=lambda z: raw(z))[:300])
    if len(v) > 2: print('   CD2 gives', v[2], 'optimized gives', v[3])
