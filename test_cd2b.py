"""Self-check for cd2b.py: the example difficulties load clean, and broken ones give the error CD2 would.
Run: python test_cd2b.py"""

import glob
import json
import os

import cd2b

HERE = os.path.dirname(os.path.abspath(__file__))
S = cd2b.load_schema()


def errors(doc):
    return [i for i in cd2b.validate(doc, S) if i[0] == 'error']


def has(doc, path, text):
    found = [i for i in errors(doc) if i[1] == path and text in i[2]]
    assert found, 'expected an error at %s containing %r, got %r' % (path, text, errors(doc))


def test_examples_clean():
    for p in sorted(glob.glob(os.path.join(HERE, 'examples', '*.json'))):
        with open(p, encoding='utf-8-sig') as f:
            errs = errors(json.load(f))
        assert not errs, '%s: %s' % (os.path.basename(p), errs[:5])


def test_cd2_rules():
    has([], '.', 'Root must be an object')
    has({'Bogus': 1}, '.Bogus', 'Module "Bogus" not found')
    has({'Caps': {'MaxActiveEnemies': 'lots'}}, '.Caps.MaxActiveEnemies', 'string literal')
    has({'Caps': {'MaxEnemies': 60}}, '.Caps.MaxEnemies', 'Unrecognized Field')
    has({'Caps': {'MaxActiveEnemies': {'Mutate': 'NoSuchThing'}}}, '.Caps.MaxActiveEnemies', 'Unrecognized Mutate')
    has({'Caps': {'MaxActiveEnemies': {'Value': 1}}}, '.Caps.MaxActiveEnemies', "no 'Mutate' field")
    has({'Caps': {'MaxActiveEnemies': {'Var': 'Missing'}}}, '.Caps.MaxActiveEnemies', 'not defined in Vars')
    # A per-player array is an implicit ByPlayerCount for a float, and fine.
    assert not errors({'Caps': {'MaxActiveEnemies': [60, 80, 100, 120]}})
    # Inside a mutator there are no per-player arrays, and a Select-family option takes the field's type.
    cost = {'Resupply': {'Cost': {'Mutate': 'ByResuppliesCalled', 'Values': [20, 'x', [1, 2]]}}}
    has(cost, '.Resupply.Cost.Values[1]', 'string literal')
    has(cost, '.Resupply.Cost.Values[2]', 'no per-player array')
    # Keys match case-insensitively, with a warning.
    issues = cd2b.validate({'caps': {'maxactiveenemies': 60}}, S)
    assert not [i for i in issues if i[0] == 'error'] and len(issues) == 2, issues


def test_ecd2a():
    base = {'ElytrasAddon': {'Enemies': {'ED_Spider_Grunt': {}}}}
    grunt = base['ElytrasAddon']['Enemies']['ED_Spider_Grunt']
    p = '.ElytrasAddon.Enemies.ED_Spider_Grunt'
    grunt['Bogus'] = 1
    has(base, p + '.Bogus', 'Unrecognized Field')
    del grunt['Bogus']
    grunt['Teleporter'] = {'Type': 'Swapper', 'TargetPool': 'Damaged - Self'}
    has(base, p + '.Teleporter', 'DestinationPool')
    grunt['Teleporter'] = {'Type': 'Normal', 'TargetPool': 'Damaged -'}
    has(base, p + '.Teleporter.TargetPool', 'Missing operand')
    grunt['Teleporter'] = [{'TargetPool': 'Self'}, {'TargetPool': '(Target & AllDamaged) + Self'}]
    assert not errors(base), errors(base)


def test_optimizer():
    """The DSL's optimizer folds with CD2's arithmetic and never changes what CD2 computes (see cd2_schema.py,
    ConstantFolder). Each case is a bug the old folder had."""
    import cd2_schema as cs
    fold = lambda e: cs._plain(cs.RedundantSelectorRemover().optimize(cs.ConstantFolder().optimize(e)))
    x = {'Mutate': 'DwarfCount'}
    assert fold({'Mutate': 'Ceil', 'Value': 1e-7}) == 1             # was 0: every float rounded to 6 decimals
    assert fold({'Mutate': 'Add', 'A': x, 'B': 1e-7}) == {'Mutate': 'Add', 'A': x, 'B': 1e-7}
    assert fold({'Mutate': 'Round', 'Value': 0.5}) == 1             # was 0: Python's banker's rounding
    assert fold({'Mutate': 'Round', 'Value': -1.5}) == -1           # CD2: floor(x + 0.5)
    assert fold({'Mutate': 'Multiply', 'Values': [x]}) == x          # was the bare list [x]
    assert fold({'Mutate': 'Add', 'Values': [x, 1], 'C': 2}) == {'Mutate': 'Add', 'Values': [x], 'A': 3}
    assert fold({'Mutate': 'Modulo', 'A': -1, 'B': 3}) == -1         # fmod, not Python's %
    assert fold({'Mutate': 'Divide', 'A': 1, 'B': 0}) == 0           # SafeDivide
    assert isinstance(fold({'Mutate': 'Pow', 'A': -8, 'B': 1 / 3}), dict)   # NaN in CD2: not folded (was complex)
    assert fold({'Mutate': 'IfFloat', 'Value': 2, '>': 1, 'Then': x, 'Else': 0}) == x
    assert fold({'Mutate': 'And', 'A': {'Mutate': 'IsSolo'}, 'B': False}) is False
    assert fold({'Mutate': 'ByBiome', 'Default': 1, 'MagmaCore': 1}) == 1
    assert fold({'Mutate': 'ByPlayerCount', 'Values': [x, x]}) == x
    # A NoOptimize subtree is left alone wherever it sits, and a Var it names survives.
    keep = cs.Mul(2, 3).NoOptimize()
    assert fold(cs.Expression({'Mutate': 'Add', 'A': keep, 'B': x})) == {'Mutate': 'Add', 'A': {'Mutate': 'Multiply', 'A': 2, 'B': 3}, 'B': x}
    v = cs.Variable('K', 5)
    w = cs.Variable('Shown', 1, watch=True)
    p = cs.DifficultyProfile(Name='t', Vars=[v, w], Caps=cs.CapSettings(
        MaxActiveEnemies=cs.Expression({'Mutate': 'Add', 'A': cs.Expression({'Var': 'K'}).NoOptimize(), 'B': w})))
    d = p.to_dict(optimize=True)
    assert set(d['Vars']) == {'K', 'Shown'}, d
    # The DSL writes only what CD2 accepts.
    assert cs.Neq(x, 1).to_dict() == {'Mutate': 'IfFloat', 'Value': x, '==': 1, 'Then': False, 'Else': True}
    assert cs.Abs(x).to_dict() == {'Mutate': 'Abs', 'Value': x}


def test_dsl_build_valid():
    """nightmares.py's output, optimized and not, loads clean in CD2's rules."""
    import runpy
    import cd2_schema as cs
    got = {}
    save, cs.DifficultyProfile.save = cs.DifficultyProfile.save, lambda self, *a, **k: got.setdefault('p', self)
    try:
        runpy.run_path(os.path.join(HERE, 'nightmares.py'), run_name='__main__')
    finally:
        cs.DifficultyProfile.save = save
    for opt in (False, True):
        errs = errors(got['p'].to_dict(optimize=opt))
        assert not errs, (opt, errs[:5])


def test_decompile():
    """decompile.py's scripts rebuild every example exactly, and the cases that once broke it."""
    import decompile as dc
    for p in sorted(glob.glob(os.path.join(HERE, 'examples', '*.json'))):
        with open(p, encoding='utf-8-sig') as f:
            doc = json.load(f)
        assert dc.decompile(doc, p)[1], p
    by_dna = {'Mutate': 'ByDNA', 'Default': 10, 'PE': 12, 'Mining,2': 12}
    doc = {'Name': 't',
           'Vars': {'A': {'Type': 'Float', 'Value': {'Var': 'B'}}, 'B': {'Type': 'Float', 'Value': {'Var': 'A'}},
                    'If': {'Type': 'Float', 'Value': 1, 'Watch': True}},
           # A dict on the left of * hands the product to the right operand, which puts itself first.
           'Caps': {'MaxActiveEnemies': {'Mutate': 'Multiply', 'A': by_dna, 'B': {'Mutate': 'DwarfCount'}},
                    'MaxActiveSwarmers': {'Mutate': 'Add', 'A': by_dna, 'B': {'Var': 'If'}},
                    'maxactivecritters': {'Mutate': 'IfFloat', 'Value': 1, '>=': {'Var': 'A'}, 'Then': 2, 'Else': None}},
           'ElytrasAddon': {'Enemies': {}}}
    src, ok = dc.decompile(doc, 't')
    assert ok, src
    assert 'Var("A")' in src or 'Var("B")' in src   # the cycle is cut with a by-name reference
    # CD2.sav: a GVAS file with a Difficulties map of name -> JSON text.
    import struct, tempfile
    fs = lambda t: struct.pack('<i', len(t.encode()) + 1) + t.encode() + b'\0'
    sav = (b'GVAS' + fs('Difficulties') + fs('MapProperty') + struct.pack('<q', 0) + fs('StrProperty') * 2 + b'\0' +
           struct.pack('<ii', 0, 1) + fs('Hazard 5') + fs('{"Name":"Hazard 5"}'))
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, 'CD2.sav'), 'wb') as f:
            f.write(sav)
        assert dc.read_sav(os.path.join(d, 'CD2.sav')) == {'Hazard 5': '{"Name":"Hazard 5"}'}


def test_pool_formulas():
    ok = ['Self', 'All - Self', 'AllTargets - Damaged & AllDamaged', '(Target + Self) & All', ' damaged ']
    bad = ['', 'Target -', '& Self', 'Target Self', '(Target', 'Target)', 'Nobody', 'Self * 2']
    for f in ok:
        assert cd2b.pool_formula_error(f) == '', f
    for f in bad:
        assert cd2b.pool_formula_error(f), f


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
