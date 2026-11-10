"""Turns CD2 difficulties back into cd2_schema build scripts: every difficulty in CD2's save (CD2.sav), or one .json.

Each mutator is written in the most readable form that gives back exactly the same JSON: an operator (`a * b`), a
DSL constant (`DwarfCount`), a DSL helper (`If(c, a, b)`), `Mut("Name", ...)`, and a dict literal last. A form is only
used after building it and comparing the result with the original node, so a helper whose keys differ from the file
(a lower-case key, a JSON null) falls through to the next form instead of changing the difficulty. Subtrees used
twice or more become named variables, and CD2 variables become `Variable`s.

Each script is then run and its output compared with the original as CD2 reads it (floats as float32).

Run: python decompile.py <CD2.sav | difficulty.json> [--out DIR] [--json]
     --json writes the difficulties' JSON as well (pretty-printed), beside the scripts.
CD2.sav lives in <DRG>/FSD/Saved/SaveGames/Mods/; it is only read."""

import inspect
import json
import keyword
import os
import re
import struct
import sys
from collections import Counter

import cd2_schema as cs

NS = {k: v for k, v in vars(cs).items() if not k.startswith('__')}


# ---- CD2.sav: a GVAS file whose CD2_Save holds `TMap<FString, FString> Difficulties` (name -> JSON text)

def read_sav(path):
    with open(path, 'rb') as f:
        d = f.read()
    if d[:4] != b'GVAS':
        raise ValueError('%s is not a GVAS save' % path)

    def fstr(p):
        n, = struct.unpack_from('<i', d, p)
        p += 4
        if n < 0:
            return d[p:p - 2 * n].decode('utf-16-le')[:-1], p - 2 * n
        return d[p:p + n].decode('utf-8')[:-1], p + n

    # First we skip the property tag: "MapProperty", its size, the key and value types and the no-GUID byte.
    tag = b'\x0d\x00\x00\x00Difficulties\x00'
    p = d.find(tag)
    if p < 0:
        raise ValueError('%s has no Difficulties map (not CD2.sav?)' % path)
    kind, p = fstr(p + len(tag))
    p += 8
    _, p = fstr(p)
    _, p = fstr(p)
    p += 1
    removed, count = struct.unpack_from('<ii', d, p)
    p += 8
    out = {}
    for _ in range(count):
        k, p = fstr(p)
        out[k], p = fstr(p)
    return out


# ---- JSON -> Python source

def canon(v):
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def lit(v):
    if isinstance(v, bool) or v is None:
        return repr(v)
    if isinstance(v, (int, float)):
        return repr(v)
    return json.dumps(v, ensure_ascii=False)   # a JSON string literal is a valid Python one


class Code(str):
    """Source text, flagged when it is an infix expression an operand must parenthesise."""
    infix = False


def infix(text):
    c = Code(text)
    c.infix = True
    return c


def is_ident(k):
    return isinstance(k, str) and k.isidentifier() and not keyword.iskeyword(k)


def block(head, items, tail, width=100):
    """`head item, item tail` on one line when it fits, else one item per line."""
    items = list(items)
    one = head + ', '.join(items) + tail
    if len(one) <= width and '\n' not in one:
        return one
    body = ''.join('    ' + it.replace('\n', '\n    ') + ',\n' for it in items)
    return head + '\n' + body + tail


ALIASES = {'Multiply': 'Mul', 'Subtract': 'Sub', 'Divide': 'Div', 'NonZero': 'Nonzero'}
INFIX = {'Add': '+', 'Multiply': '*', 'Subtract': '-', 'Divide': '/', 'Pow': '**', 'Modulo': '%'}
CMP = ('==', '>=', '>', '<=', '<')
CONSTS = {canon(cs._plain(v)): k for k, v in NS.items()
          if type(v) is cs.Expression and isinstance(v.content, dict)}
GENERIC_KEYS = {'Value', 'Values', 'Then', 'Else', 'Condition', 'Default', 'Min', 'Max', 'Mutate'}


def norm(name):
    name = name.lower().split('_or_')[0]
    for suffix in ('_val', '_str'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name.replace('_', '')


class Decompiler:
    def __init__(self, doc):
        self.doc = doc
        self.lines = []
        self.vars = {}       # CD2 variable name -> its definition, when every one is well-formed
        self.pyname = {}     # CD2 variable name or shared subtree canon -> Python name once written
        self.busy = set()
        self.env = dict(NS)  # the script so far, run: the names its bindings define
        self.memo = {}       # id(node) -> its code, so a parent's tries do not regenerate its children
        self.taken = set(NS) | set(keyword.kwlist) | {'profile', 'Var'}
        vs = doc.get('Vars')
        if isinstance(vs, dict) and all(isinstance(d, dict) and isinstance(d.get('Type'), str) and 'Value' in d and
                                        set(d) <= {'Type', 'Value', 'Watch'} for d in vs.values()):
            self.vars = vs
        self.shared = self.find_shared()

    # -- names

    def fresh(self, base):
        base = re.sub(r'\W', '_', base).strip('_') or 'x'
        if not (base[0].isalpha() or base[0] == '_'):
            base = 'v_' + base
        name, i = base, 2
        while name in self.taken:
            name, i = '%s_%d' % (base, i), i + 1
        self.taken.add(name)
        return name

    # -- repeated subtrees: the largest first, each counted with the ones already chosen standing as one name

    def find_shared(self):
        memo = {}

        def key(n):
            k = memo.get(id(n))
            if k is None:
                k = memo[id(n)] = canon(n)
            return k

        def nodes(v, path, out):
            if isinstance(v, dict):
                if isinstance(v.get('Mutate'), str):
                    out.append((v, path))
                for k, x in v.items():
                    nodes(x, path + [k], out)
            elif isinstance(v, list):
                for x in v:
                    nodes(x, path, out)

        everything = []
        nodes(self.doc, [], everything)
        occ = Counter(key(n) for n, _ in everything)
        first = {}
        for n, path in everything:
            first.setdefault(key(n), (n, path))
        cands = sorted((c for c, k in occ.items() if k >= 2 and len(c) >= 60 and c not in CONSTS),
                       key=len, reverse=True)
        chosen = {}

        def count(c, v, stop):
            if isinstance(v, dict):
                if isinstance(v.get('Mutate'), str):
                    k = key(v)
                    if k == c:
                        return 1
                    if k in stop:
                        return 0
                return sum(count(c, x, stop) for x in v.values())
            if isinstance(v, list):
                return sum(count(c, x, stop) for x in v)
            return 0

        for c in cands:
            stop = set(chosen) | {c}
            uses = count(c, self.doc, stop) + sum(
                count(c, x, stop) for s in chosen for x in first[s][0].values())
            if uses >= 2:
                chosen[c] = first[c]
        return chosen

    def shared_name(self, path):
        keys = [k for k in path if isinstance(k, str) and k not in GENERIC_KEYS and not re.fullmatch(r'[A-Z]', k)]
        return '_'.join(keys[-2:]) or 'shared'

    # -- bindings, written before their first use

    def var_ref(self, name):
        if name in self.pyname:
            return Code(self.pyname[name])
        if name in self.vars and name not in self.busy:
            return Code(self.emit_var(name))
        return Code('Var(%s)' % lit(name))

    def emit_var(self, name):
        self.busy.add(name)
        d = self.vars[name]
        args = [lit(name), self.code(d['Value']), lit(d['Type'])]
        if 'Watch' in d:
            args.append('watch=' + self.code(d['Watch']))
        py = self.fresh(name if is_ident(name) else 'var_' + name)
        self.bind('%s = %s' % (py, block('Variable(', args, ')')))
        self.pyname[name] = py
        self.busy.discard(name)
        return py

    def emit_shared(self, c):
        n, path = self.shared[c]
        self.busy.add(c)
        body = self.mutator(n)
        py = self.fresh(self.shared_name(path))
        self.bind('%s = %s' % (py, body))
        self.pyname[c] = py
        self.busy.discard(c)
        return py

    def bind(self, line):
        self.lines.append(line)
        exec(line, self.env)

    # -- values

    def code(self, v):
        if isinstance(v, (dict, list)):
            if id(v) not in self.memo:
                self.memo[id(v)] = self._code(v)
            return self.memo[id(v)]
        return Code(lit(v))

    def _code(self, v):
        if isinstance(v, list):
            return Code(block('[', (self.code(x) for x in v), ']'))
        if not isinstance(v, dict):
            return Code(lit(v))
        if set(v) == {'Var'} and isinstance(v['Var'], str):
            return self.var_ref(v['Var'])
        if isinstance(v.get('Mutate'), str):
            c = canon(v)
            if c in self.pyname:
                return Code(self.pyname[c])
            if c in self.shared and c not in self.busy:
                return Code(self.emit_shared(c))
            return self.mutator(v)
        return self.dict_lit(v)

    # -- a mutator: the first form that rebuilds the node exactly. The form is tried on its operands' real values
    # (their own code, run), since an operator's result depends on what its operands are, not only on their JSON.

    def dict_lit(self, v):
        return Code(block('{', ('%s: %s' % (lit(k), self.code(x)) for k, x in v.items()), '}'))

    def mutator(self, n):
        for form in self.forms(n):
            if self.rebuilds(n, form):
                return form(self.code)
        return Code('Expression(%s)' % self.dict_lit(n))

    def rebuilds(self, n, form):
        ph = {}

        def hole(v):
            name = '_p%d' % len(ph)
            ph[name] = eval(self.code(v), self.env)
            return Code(name)

        try:
            got = eval(form(hole), self.env, ph)
        except Exception:
            return False
        if not isinstance(got, cs.Expression):
            return False
        try:
            return canon(cs._plain(got)) == canon(n)
        except TypeError:
            return False

    def forms(self, n):
        m = n['Mutate']
        keys = [k for k in n if k != 'Mutate']
        c = canon(n)
        if c in CONSTS:
            yield lambda code: Code(CONSTS[c])

        def operand(code, v):
            s = code(v)
            neg = isinstance(v, (int, float)) and not isinstance(v, bool) and v < 0
            return '(%s)' % s if s.infix or neg else s

        if m in INFIX and set(keys) == {'A', 'B'}:
            op = INFIX[m]
            yield lambda code: infix('%s %s %s' % (operand(code, n['A']), op, operand(code, n['B'])))
        ops = [k for k in keys if k in CMP] if m == 'IfFloat' else []
        op = ops[0] if len(ops) == 1 else None
        if op:
            for shown in ((op, '!=') if op == '==' else (op,)):
                yield lambda code, shown=shown: infix('%s %s %s' % (
                    operand(code, n['Value']), shown, operand(code, n[op])))
        if op and set(keys) == {'Value', op, 'Then', 'Else'}:
            yield lambda code: Code(block('IfFloat(', [code(n['Value']), lit(op), code(n[op]), code(n['Then']),
                                                       code(n['Else'])], ')'))
        f = NS.get(ALIASES.get(m, m))
        if callable(f) and inspect.isfunction(f):
            call = self.helper_call(ALIASES.get(m, m), f, n, keys)
            if call:
                yield call
        if all(n[k] is not None for k in keys):   # Mut drops a None
            odd = [k for k in keys if not is_ident(k)]
            yield lambda code: Code(block('Mut(', [lit(m)] + ['%s=%s' % (k, code(n[k])) for k in keys if k not in odd] +
                                          (['**' + block('{', ('%s: %s' % (lit(k), code(n[k])) for k in odd), '}')]
                                           if odd else []), ')'))

    def helper_call(self, fname, f, n, keys):
        """Maps the node's keys onto the helper's parameters by name (`min_val` <- Min); a lone required
        parameter left over takes the lone key left over (a trigger's input); the rest go to *args as A, B, ... or
        to **kwargs. None when they do not fit; whether the call really rebuilds the node is checked after."""
        params = list(inspect.signature(f).parameters.values())
        left = list(keys)
        bound = {}
        named = [p for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
        for exact in (True, False):   # Min <- min_val first, then InitialValue <- initial
            for p in named:
                k = next((k for k in left if p.name not in bound and
                          (k.lower() == norm(p.name) if exact else k.lower().startswith(norm(p.name)))), None)
                if k is not None:
                    bound[p.name] = k
                    left.remove(k)
        # ByDNA's "Mining,2": a key no keyword can spell goes in the helper's *_dict parameter.
        spill = next((p for p in named if p.name.endswith('_dict') and p.name not in bound), None)
        odd = left if any(not is_ident(k) for k in left) else []   # ByDNA takes the dict or keywords, not both
        if spill and odd:
            left = []
        missing = [p for p in named if p.name not in bound and p.default is p.empty and not (spill and odd and p is spill)]
        if len(missing) == 1 and len(left) == 1 and not any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params):
            bound[missing[0].name] = left.pop()
        elif missing:
            return None
        star = [k for k in left if re.fullmatch(r'[A-Z]', k)]
        if star and (star != [chr(65 + i) for i in range(len(star))] or
                     not any(p.kind == p.VAR_POSITIONAL for p in params)):
            star = []
        kw = [k for k in left if k not in star]
        if kw and not (any(p.kind == p.VAR_KEYWORD for p in params) and all(is_ident(k) for k in kw)):
            return None
        if star and any(p.name in bound and p.default is not p.empty for p in named):
            return None   # *args after a keyword argument would be a syntax error

        def render(code):
            args = []
            for p in named:
                if p.name in bound:
                    v = code(n[bound[p.name]])
                    args.append(v if p.default is p.empty else '%s=%s' % (p.name, v))
            if spill and odd:
                args.append(block('{', ('%s: %s' % (lit(k), code(n[k])) for k in odd), '}'))
            args += [code(n[k]) for k in star]
            args += ['%s=%s' % (k, code(n[k])) for k in kw]
            return Code(block(fname + '(', args, ')'))

        return render

    # -- the script

    def script(self, title):
        fields = [k for k in cs.DifficultyProfile.__annotations__ if k != 'Extra']
        refs = [self.var_ref(k) for k in self.vars]   # first, so the variables lead the script in Vars order
        args, extra = [], {}
        for k, v in self.doc.items():
            if k == 'Vars' and self.vars:
                continue
            if k in fields:
                args.append('%s=%s' % (k, self.code(v)))
            else:
                extra[k] = v
        if self.vars:
            args.insert(min(2, len(args)), 'Vars=%s' % block('[', refs, ']'))
        if extra:
            args.append('Extra=%s' % self.dict_lit(extra))
        if 'Name' not in self.doc:
            args.insert(0, 'Name=None')
        profile = 'profile = %s' % block('DifficultyProfile(', args, ')', width=0)
        head = ['"""%s, decompiled by decompile.py. Run it to print the difficulty\'s JSON."""' %
                title.replace('\\', '\\\\').replace('"""', '\\"\\"\\"'), '', 'from cd2_schema import *', '']
        tail = ['', '', 'if __name__ == "__main__":', '    print(profile.json(optimize=False))', '']
        return '\n'.join(head + self.lines + (['', ''] if self.lines else []) + [profile] + tail)


# ---- checking a script: run it, compare its output with the original as CD2 reads it

def same(a, b):
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return struct.pack('<f', a) == struct.pack('<f', b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def run_script(src):
    env = {'__name__': 'decompiled'}
    exec(compile(src, '<decompiled>', 'exec'), env)
    return env['profile'].to_dict(optimize=False)


def decompile(doc, title):
    src = Decompiler(doc).script(title)
    return src, same(run_script(src), doc)


def file_name(name):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name).strip(' .') or 'unnamed'


def main(argv):
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__)
        return 2
    src = argv[0]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    if src.lower().endswith('.sav'):
        texts = read_sav(src)
    else:
        with open(src, encoding='utf-8-sig') as f:
            texts = {os.path.splitext(os.path.basename(src))[0]: f.read()}
    bad = 0
    for name, text in texts.items():
        doc = json.loads(text, strict=False)   # CD2's own saves hold raw newlines inside strings
        script, ok = decompile(doc, name)
        base = os.path.join(out, file_name(name))
        with open(base + '.py', 'w', encoding='utf-8', newline='\n') as f:
            f.write(script)
        if '--json' in argv:
            with open(base + '.json', 'w', encoding='utf-8', newline='\n') as f:
                json.dump(doc, f, indent=4, ensure_ascii=False)
                f.write('\n')
        bad += not ok
        print('%s  %s.py' % ('ok  ' if ok else 'DIFF', base))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
