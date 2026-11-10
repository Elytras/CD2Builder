"""Builds schema/cd2.json from the fragments in schema/parts/, which were read off CD2's decompiled Blueprints
(CD2_SCHEMA.md, "Regenerating"). It normalises the readers' vocabulary into the one cd2b.py checks:

- a mutator's "returns" "any(select)" -> "any", and a parameter typed "any(select)" or "option" -> "same" (it takes
  the type of the value the Select-family mutator stands in);
- "enum": "enums.X" -> "X" (a shared list in "enums");
- an "extra" with an "enum" lists the extra keys allowed -> "keys";
- Vars' per-Type options -> a "switch" on the raw "Type" field;
- two readers' copies of one mutator are merged: the folder owner's entry wins, the other fills missing params.

Run: python schema/merge.py"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PARTS = os.path.join(HERE, 'parts')


def part(name):
    with open(os.path.join(PARTS, name), encoding='utf-8') as f:
        return json.load(f)


def norm(e, in_mutator=False):
    if isinstance(e, list):
        return [norm(x, in_mutator) for x in e]
    if not isinstance(e, dict):
        return e
    out = {k: norm(v, in_mutator) for k, v in e.items()}
    if out.get('type') in ('any(select)', 'option'):
        out['type'] = 'same'
    if isinstance(out.get('enum'), str) and out['enum'].startswith('enums.'):
        out['enum'] = out['enum'][len('enums.'):]
    # Registry names are suggestions, not a closed set: CD2 fills its registries at run time from every mounted pak.
    for k, new in (('enum_ref', 'registry'), ('key_enum_ref', 'key_registry')):
        if isinstance(out.get(k), str):
            out[new] = out.pop(k).split('.', 1)[-1]
    if out.get('inherits'):   # EnemiesNoSync: a header-only subclass of Enemies
        out['notes'] = ('Same schema as %s. ' % out.pop('inherits')) + (out.get('notes') or '')
    return out


def norm_mutator(m):
    m = norm(m, True)
    if m.get('returns') == 'any(select)':
        m['returns'] = 'any'
    ex = m.get('extra')
    if isinstance(ex, dict) and isinstance(ex.get('enum'), list):
        ex['keys'] = ex.pop('enum')
    return m


def main():
    modules, core, state, enemies = part('modules.json'), part('mutators-core.json'), part('mutators-state.json'), \
        part('enemies.json')

    mods = {}
    for src in (modules['modules'], enemies['modules']):
        for k, v in src.items():
            mods[k] = norm(v)

    # Vars: the raw "Type" picks the variable's kind, so a switch reports the right option's errors.
    vars_ = mods['Vars']['values']
    names = ('Float', 'Boolean', 'String', 'Strings')
    vars_['type'] = 'switch'
    vars_['field'] = 'Type'
    vars_['cases'] = dict(zip(names, vars_.pop('options')))

    muts = {}
    for k, v in core['mutators'].items():
        muts[k] = norm_mutator(v)
    for k, v in state['mutators'].items():   # the state reader owns Mission/ etc.: it wins on shared names
        v = norm_mutator(v)
        if k in muts:
            for p, pe in muts[k].get('params', {}).items():
                v.setdefault('params', {}).setdefault(p, pe)
            v['notes'] = (v.get('notes') or '') + ((' | ' + muts[k]['notes']) if muts[k].get('notes') else '')
        muts[k] = v

    values = {k: {'notes': v.get('notes', ''), 'accepts_json': v.get('accepts', []), 'src': v.get('src')}
              for k, v in core['values'].items()}
    accepts = {'float': ['float'], 'bool': ['bool'], 'string': ['string'], 'stringlist': ['stringlist', 'string'],
               'floatarray': ['float'], 'randinterval': ['float']}
    for k, a in accepts.items():
        values.setdefault(k, {})['accepts'] = a

    schema = {
        'cd2': {
            'pak': 'custom-difficulty2.pak',
            'pak_sha1': '70a6123b3206d638d36b937e7a45aefb48a906ee',
            'decompiled_with': 'ue-mods AssetGen/tools/bpdecomp_emit.py --mod /Game/_AssemblyStorm=AssemblyStorm',
            'src_root': 'AssemblyStorm/ (the decompiled tree); every "src" is relative to it',
            'reference': 'CD2_SCHEMA.md',
        },
        'vartypes': {'Float': 'float', 'Boolean': 'bool', 'String': 'string', 'Strings': 'stringlist'},
        'values': values,
        'modules': dict(sorted(mods.items())),
        'mutators': dict(sorted(muts.items())),
        'abstract': core.get('abstract', []),
        'enums': {k: norm(v) for k, v in state.get('enums', {}).items()},
        'registries': dict(enemies.get('registries', {}),
                           ED_mods=sorted(set(enemies['registries_mod_provided'].get('ED_yinnys_addon', []) +
                                              enemies['registries_mod_provided'].get('ED_NewEnemies', [])))),
        'registries_notes': dict(enemies.get('registries_notes', {}),
                                 ED_mods=enemies['registries_mod_provided'].get('notes', '')),
        'defs': {},
    }
    with open(os.path.join(HERE, 'cd2.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(schema, f, indent=1, ensure_ascii=False)
        f.write('\n')
    write_reference(schema)
    print('cd2.json: %d modules, %d mutators, %d enums, registries %s' % (
        len(mods), len(muts), len(schema['enums']), {k: len(v) for k, v in schema['registries'].items()}))


# ---- REFERENCE.md: every module, field and mutator, generated so it cannot drift from cd2.json

def kind(e, defs):
    while 'ref' in e:
        e = dict(defs[e['ref']], **{k: v for k, v in e.items() if k != 'ref'})
    t = e.get('type', 'any')
    if t == 'array':
        return 'array of ' + kind(e.get('items', {}), defs)
    if t == 'map':
        return 'map of ' + kind(e.get('values', {}), defs)
    if t == 'oneof':
        return ' or '.join(kind(o, defs) for o in e['options'])
    if t == 'switch':
        return 'by %s: %s' % (e['field'], ', '.join(e['cases']))
    return t


def cell(text):
    return (text or '').replace('|', '\\|').replace('\n', ' ')


def rows(fields, defs, prefix, out):
    for name, e in fields.items():
        r = e
        while 'ref' in r:
            r = dict(defs[r['ref']], **{k: v for k, v in r.items() if k != 'ref'})
        extra = []
        if r.get('required'):
            extra.append('required')
        if 'default' in r and r['default'] is not None:
            extra.append('default `%s`' % json.dumps(r['default']))
        if isinstance(r.get('enum'), list):
            extra.append('one of ' + ', '.join('`%s`' % c for c in r['enum']))
        elif isinstance(r.get('enum'), str):
            extra.append('one of enum `%s`' % r['enum'])
        if r.get('registry'):
            extra.append('names: registry `%s`' % r['registry'])
        out.append('| `%s%s` | %s | %s | %s |' % (prefix, name, kind(e, defs), cell('; '.join(extra + [r.get('notes') or ''])),
                                               cell(r.get('src'))))
        sub = r.get('fields') or (r.get('values') or {}).get('fields') if r.get('type') in ('object', 'map') else None
        if r.get('type') == 'map' and isinstance(r.get('values'), dict):
            v = r['values']
            while 'ref' in v:
                v = defs[v['ref']]
            sub = v.get('fields')
            if sub:
                rows(sub, defs, prefix + name + '.<key>.', out)
            continue
        if sub:
            rows(sub, defs, prefix + name + '.', out)


def write_reference(schema, path=os.path.join(HERE, 'REFERENCE.md')):
    defs = schema['defs']
    out = ['# CD2 schema reference (generated)', '',
           'Generated by `schema/merge.py` from `cd2.json`; do not edit. How CD2 parses is in `CD2_SCHEMA.md`.',
           '"src" is relative to the decompiled tree `AssemblyStorm/`.', '', '## Modules (top-level keys)', '']
    for name, e in schema['modules'].items():
        out += ['### %s%s' % (name, ' (test module)' if e.get('test') else ''), '', cell(e.get('notes')), '',
                'Shape: %s. Unknown keys: %s. Source: `%s`.' % (kind(e, defs), e.get('unknown', '-'), e.get('src')), '']
        fields = e.get('fields')
        if e.get('type') == 'map':
            fields = (e.get('values') or {}).get('fields')
        if e.get('type') == 'switch' or (e.get('values') or {}).get('type') == 'switch':
            sw = e if e.get('type') == 'switch' else e['values']
            for case, ce in sw['cases'].items():
                out += ['%s = `%s`:' % (sw['field'], case), '', '| key | type | notes | src |', '| --- | --- | --- | --- |']
                rows(ce.get('fields', {}), defs, '', out)
                out.append('')
            continue
        if fields:
            out += ['| key | type | notes | src |', '| --- | --- | --- | --- |']
            rows(fields, defs, '<name>.' if e.get('type') == 'map' else '', out)
            out.append('')
    out += ['## Mutators', '', '`returns any` = Select-family: it returns whatever the place it is used wants, and its '
            'options (type `same`) are built as that type.', '']
    for name, m in schema['mutators'].items():
        out += ['### %s -> %s' % (name, m.get('returns')), '', cell(m.get('notes')), '',
                'Unknown keys: %s. Source: `%s`.' % (m.get('unknown', '-'), m.get('src')), '']
        if m.get('params'):
            out += ['| param | type | notes | src |', '| --- | --- | --- | --- |']
            rows(m['params'], defs, '', out)
            out.append('')
        if m.get('extra'):
            ex = m['extra']
            out += ['Any other key: %s%s. %s' % (kind(ex, defs), (' (keys: %s)' % ', '.join(ex['keys'])) if ex.get('keys') else '',
                                                 cell(ex.get('notes'))), '']
    out += ['## Enums', '']
    for name, e in schema['enums'].items():
        out += ['- **%s**: %s. %s (`%s`)' % (name, ', '.join('`%s`' % v for v in e.get('values', [])), cell(e.get('notes')),
                                            e.get('src'))]
    out += ['', '## Registries', '']
    for name, lst in schema['registries'].items():
        out += ['- **%s**: %d names. %s' % (name, len(lst), cell(schema.get('registries_notes', {}).get(name)))]
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
