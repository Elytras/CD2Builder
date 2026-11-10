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
