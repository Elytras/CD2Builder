"""CD2Builder: validate, format and edit Custom Difficulty 2 difficulty files against the schema read off CD2's
decompiled Blueprints (schema/cd2.json) and its extensions (schema/ecd2a.json).

    python cd2b.py validate FILE...        CD2's would-be load errors (and warnings), with JSON paths
    python cd2b.py fmt FILE [--out OUT]    re-indent (4 spaces, key order kept)
    python cd2b.py edit [FILE] [--port N]  the editor (editor.html) in the browser, served from here

Standard library only. The rules it applies are written up in schema/CD2_SCHEMA.md."""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_DIR = os.path.join(HERE, 'schema')
DEFAULT_SCHEMAS = ('cd2.json', 'ecd2a.json')

# The six kinds CD2's value factory makes (CD2_Value_Factory); everything else in an entry is structure or raw JSON.
MUTATABLE = ('float', 'bool', 'string', 'stringlist', 'floatarray', 'randinterval')
# What a mutator may return where each kind is wanted (MakeMutator: the class must be a child of the base type, and a
# string mutator is wrapped as a one-entry list where a list is wanted). Select-family mutators ("any") fit anywhere.
ACCEPTS = {'float': ('float',), 'bool': ('bool',), 'string': ('string',), 'stringlist': ('stringlist', 'string'),
           'floatarray': ('float',), 'randinterval': ('float',)}
# The type a Select-family mutator's options are built with in each context.
SELECT_BASE = {'float': 'float', 'bool': 'bool', 'string': 'string', 'stringlist': 'stringlist',
               'floatarray': 'float', 'randinterval': 'float'}


# ---- schema

def load_schema(names=DEFAULT_SCHEMAS):
    """The base schema with each extension laid over it. An extension adds modules, mutators, defs and enums by name
    and may not replace one the base has: an addon extends CD2, it does not fork it."""
    merged = None
    for n in names:
        with open(os.path.join(SCHEMA_DIR, n), encoding='utf-8') as f:
            part = json.load(f)
        if merged is None:
            merged = part
            for k in ('modules', 'mutators', 'defs', 'enums', 'registries', 'values', 'vartypes'):
                merged.setdefault(k, {})
            merged['extensions'] = []
            continue
        for k in ('modules', 'mutators', 'defs', 'enums'):
            for name, entry in part.get(k, {}).items():
                if name in merged[k]:
                    raise ValueError('%s: %s %r is already in the base schema' % (n, k, name))
                merged[k][name] = entry
        merged['extensions'].append({'file': n, 'about': part.get('about'), 'status': part.get('status')})
    return merged


def find_key(d, key):
    """The key of d that CD2 would match for `key`: exact first, then case-insensitively (CD2's string switches
    compile to NotEqual_StriStri and its TMap<FString> lookups ignore case)."""
    if key in d:
        return key
    low = key.lower()
    for k in d:
        if k.lower() == low:
            return k
    return None


# ---- validation

class Checker:
    def __init__(self, schema, doc):
        self.s = schema
        self.doc = doc
        self.issues = []
        self.vars = {}   # variable name -> value kind, from the difficulty's own Vars
        vars_ = doc.get(find_key(doc, 'Vars') or 'Vars') if isinstance(doc, dict) else None
        if isinstance(vars_, dict):
            for name, d in vars_.items():
                t = d.get(find_key(d, 'Type') or 'Type') if isinstance(d, dict) else None
                if isinstance(t, str):
                    kind = self.s['vartypes'].get(find_key(self.s['vartypes'], t) or '', None)
                    self.vars[name] = kind

    def err(self, path, msg):
        self.issues.append(('error', path or '.', msg))

    def warn(self, path, msg):
        self.issues.append(('warning', path or '.', msg))

    def resolve(self, e):
        while 'ref' in e:
            e = dict(self.s['defs'][e['ref']], **{k: v for k, v in e.items() if k != 'ref'})
        return e

    def run(self):
        if not isinstance(self.doc, dict):
            self.err('', 'Root must be an object')
            return self.issues
        mods = self.s['modules']
        for key, val in self.doc.items():
            name = find_key(mods, key)
            path = '.' + key
            if name is None:
                self.err(path, 'Module "%s" not found' % key)
                continue
            if name != key:
                self.warn(path, 'CD2 matches this as "%s" (case differs)' % name)
            self.entry(val, mods[name], path)
        return self.issues

    # One schema entry against one JSON value.
    def entry(self, v, e, path, nested=False):
        e = self.resolve(e)
        t = e.get('type', 'rawany')
        if t in MUTATABLE:
            self.value(v, t, e, path, nested)
        elif t == 'object':
            self.obj(v, e, path)
        elif t == 'map':
            if not isinstance(v, dict):
                self.err(path, 'Expected an object (a map)')
                return
            for k, sub in v.items():
                self.entry(sub, e.get('values', {}), path + '.' + k)
        elif t == 'array':
            if not isinstance(v, list):
                self.err(path, 'Expected an array')
                return
            for i, sub in enumerate(v):
                self.entry(sub, e.get('items', {}), '%s[%d]' % (path, i), nested)
        elif t == 'oneof':
            self.oneof(v, e['options'], path, nested)
        elif t == 'switch':
            # An object whose raw string field (Vars' "Type") picks its shape.
            if not isinstance(v, dict):
                return self.err(path, 'Must Be a JSON Map')
            k = find_key(v, e['field'])
            if k is None:
                return self.err(path, '"%s" required' % e['field'])
            if not isinstance(v[k], str):
                return self.err(path + '.' + k, "'%s' must be a string" % e['field'])
            case = find_key(e['cases'], v[k])
            if case is None:
                return self.err(path + '.' + k, 'Unrecognized %s "%s" (one of %s)' % (e['field'], v[k], ', '.join(e['cases'])))
            self.entry(v, e['cases'][case], path, nested)
        elif t == 'rawnumber':
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                self.err(path, 'Expected a plain number (not mutatable here)')
        elif t == 'rawstring':
            if not isinstance(v, str):
                self.err(path, 'Expected a plain string (not mutatable here)')
            else:
                self.enum(v, e, path)
        elif t == 'rawbool':
            if not isinstance(v, bool):
                self.err(path, 'Expected true or false (not mutatable here)')
        # rawany: anything goes

    def oneof(self, v, options, path, nested=False):
        best = None
        for o in options:
            sub = Checker(self.s, self.doc)
            sub.vars = self.vars
            sub.entry(v, o, path, nested)
            errs = [i for i in sub.issues if i[0] == 'error']
            if not errs:
                self.issues += sub.issues
                return
            if best is None or len(errs) < len(best):
                best = sub.issues
        self.issues += best or []

    def obj(self, v, e, path):
        if not isinstance(v, dict):
            self.err(path, e.get('not_object', 'Expected an object'))
            return
        fields = e.get('fields', {})
        for k, sub in v.items():
            name = find_key(fields, k)
            p = path + '.' + k
            if name is None:
                if e.get('extra'):
                    self.entry(sub, e['extra'], p)
                elif e.get('unknown', 'error') == 'error':
                    self.err(p, 'Unrecognized Field "%s"' % k)
                else:
                    self.warn(p, 'CD2 ignores "%s" here' % k)
                continue
            if name != k:
                self.warn(p, 'CD2 matches this as "%s" (case differs)' % name)
            self.entry(sub, fields[name], p)
        for name, f in fields.items():
            if self.resolve(f).get('required') and find_key(v, name) is None:
                self.err(path, 'Missing required field "%s"' % name)
        # "when": a literal string choice in one field that makes another one required.
        for rule in e.get('when', []):
            k = find_key(v, rule['field'])
            val = v.get(k) if k else None
            if isinstance(val, str) and find_key({c: 1 for c in rule['in']}, val) and find_key(v, rule['require']) is None:
                self.err(path, rule['error'].replace('{value}', val))

    def enum(self, s, e, path):
        choices = e.get('enum')
        if isinstance(choices, str):
            choices = (self.s['enums'].get(choices) or {}).get('values')
        if choices and find_key({c: 1 for c in choices}, s) is None:
            self.warn(path, '"%s" is none of: %s' % (s, ', '.join(choices)))
        if e.get('format') == 'poolformula':
            msg = pool_formula_error(s)
            if msg:
                self.err(path, msg)

    # A mutatable value of one kind: a literal, a {"Mutate": ...} or a {"Var": ...}. `nested` is a value inside a
    # mutator or an array, which CD2 builds with MakeMutator rather than its Make*Value: no per-player arrays there.
    # The rules are CD2_Mutator_Manager's Make*Value / MakeMutator (CD2_SCHEMA.md, "Values").
    def value(self, v, kind, e, path, nested=False):
        if v is None:
            return self.warn(path, 'null: CD2 makes no value here and reports nothing')
        if isinstance(v, dict):
            if find_key(v, 'Mutate') is not None:
                return self.mutator(v, kind, path)
            if kind == 'randinterval':
                return self.rand_range(v, path)   # any other object is a range, {"Var"} included
            if find_key(v, 'Var') is not None:
                return self.var(v, kind, path)
            return self.err(path, "Expected Mutator Object, but no 'Mutate' field found.")
        if isinstance(v, list):
            if kind == 'stringlist':
                for i, sub in enumerate(v):
                    if not isinstance(sub, str):
                        self.value(sub, 'stringlist', e, '%s[%d]' % (path, i), True)
                    else:
                        self.enum(sub, e, '%s[%d]' % (path, i))
            elif kind == 'randinterval' and not nested:
                for i, sub in enumerate(v):
                    if isinstance(sub, dict):
                        self.rand_range(sub, '%s[%d]' % (path, i))
                    else:
                        self.err('%s[%d]' % (path, i), 'Expected Random Range')
            elif kind in ('float', 'bool', 'string', 'floatarray') and not nested:
                if kind == 'floatarray' and len(v) > 4:
                    self.warn(path, 'CD2 reads a float array of more than 4 entries as ONE per-player-count value')
                for i, sub in enumerate(v):
                    self.value(sub, 'float' if kind == 'floatarray' else kind, e, '%s[%d]' % (path, i), True)
            else:
                self.err(path, 'Invalid Value/Mutate (no per-player array inside a mutator)')
            return
        lit = 'bool' if isinstance(v, bool) else 'number' if isinstance(v, (int, float)) else 'string'
        want = {'float': 'number', 'floatarray': 'number', 'randinterval': 'number', 'bool': 'bool',
                'string': 'string', 'stringlist': 'string'}[kind]
        if lit != want:
            return self.err(path, 'Mutate Type Error: a %s literal is not a %s value' % (lit, kind))
        if kind == 'stringlist' and not nested:
            return self.err(path, 'Expected List of Strings, Got String (write ["%s"])' % v)
        if lit == 'string':
            self.enum(v, e, path)

    def rand_range(self, v, path):
        """{"Min", "Max", "Weight"?}: Max and Min required, Weight 1 by default, other keys ignored."""
        for need in ('Max', 'Min'):
            if find_key(v, need) is None:
                self.err(path, "Range Must Specify '%s'" % need)
        for k, sub in v.items():
            if find_key({'Min': 1, 'Max': 1, 'Weight': 1}, k):
                self.value(sub, 'float', {}, path + '.' + k, True)
            else:
                self.warn(path + '.' + k, 'CD2 ignores "%s" in a range' % k)

    def mutator(self, v, kind, path):
        mkey = find_key(v, 'Mutate')
        name = v[mkey]
        if not isinstance(name, str):
            return self.err(path, 'Mutate value must be a String name.')
        real = find_key(self.s['mutators'], name)
        if real is None:
            return self.err(path, 'Unrecognized Mutate %s.' % name)
        m = self.s['mutators'][real]
        if real != name:
            self.warn(path, 'CD2 matches this as "%s" (case differs)' % real)
        ret = m.get('returns')
        if ret != 'any' and ret not in ACCEPTS[kind]:
            self.err(path, 'Mutate Type Error Expected "%s" Found "%s" (%s returns %s)' % (kind, real, real, ret))
        base = SELECT_BASE[kind]
        params, extra, inp = m.get('params', {}), m.get('extra'), m.get('input')
        input_key = None
        for k, sub in v.items():
            if k == mkey:
                continue
            p = find_key(params, k)
            if p is None:
                if inp and input_key is None:   # a trigger's input, under any one free key
                    input_key = k
                    self.entry(sub, inp, path + '.' + k, True)
                elif inp:
                    self.err(path + '.' + k, 'Unexpected field "%s" (%s already has its input in "%s")' % (k, real, input_key))
                elif extra and (not extra.get('keys') or find_key({c: 1 for c in extra['keys']}, k)):
                    self.entry(sub, self.typed(extra, base), path + '.' + k, True)
                elif m.get('unknown', 'ignored') == 'error':
                    self.err(path + '.' + k, '%s has no parameter "%s"' % (real, k))
                else:
                    self.warn(path + '.' + k, '%s ignores "%s"' % (real, k))
                continue
            self.entry(sub, self.typed(params[p], base), path + '.' + k, True)
        for p, pe in params.items():
            if self.resolve(pe).get('required') and find_key(v, p) is None:
                self.err(path, '%s needs "%s"' % (real, p))
        if inp and input_key is None:
            self.err(path, 'Input Value must be set (%s)' % real)

    @staticmethod
    def typed(e, kind):
        """A Select-family option takes the type of the value it stands in ("type": "same")."""
        if isinstance(e, list):
            return [Checker.typed(x, kind) for x in e]
        if not isinstance(e, dict):
            return e
        out = {k: Checker.typed(v, kind) for k, v in e.items()}
        if out.get('type') == 'same':
            out['type'] = kind
        return out

    def var(self, v, kind, path):
        name = v[find_key(v, 'Var')]
        real = find_key(self.vars, name) if isinstance(name, str) else None
        if real is None:
            return self.err(path, 'Variable "%s" is not defined in Vars' % name)
        have = self.vars[real]
        if have and have not in ACCEPTS[kind]:
            self.err(path, 'Variable "%s" is a %s, not a %s' % (real, have, kind))


def validate(doc, schema=None):
    """[(level, path, message)] for a parsed difficulty."""
    return Checker(schema or load_schema(), doc).run()


# ---- ECD2A pool formulas (DESIGN-teleporter-formulas.md: + - & over pool names, parentheses)

POOLS = ('self', 'target', 'damaged', 'alltargets', 'alldamaged', 'all')


def pool_formula_error(src):
    """The syntax error ECD2A's compiler would report for a literal formula, or ''.
    ponytail: syntax only; ECD2A also refuses a formula that can name no one (UPoolFormula::Reach)."""
    toks, i = [], 0
    while i < len(src):
        c = src[i]
        if c.isspace():
            i += 1
        elif c in '+-&()':
            toks.append((c, i)); i += 1
        elif c.isalpha():
            j = i
            while j < len(src) and src[j].isalpha():
                j += 1
            word = src[i:j]
            if word.lower() not in POOLS:
                return 'Unknown pool "%s" at %d' % (word, i + 1)
            toks.append(('pool', i)); i = j
        else:
            return 'Unexpected character "%s" at %d' % (c, i + 1)
    if not toks:
        return 'Empty formula'
    pos = [0]

    def atom():
        if pos[0] >= len(toks):
            return 'Missing operand at the end'
        t, at = toks[pos[0]]
        if t == 'pool':
            pos[0] += 1
            return ''
        if t == '(':
            pos[0] += 1
            e = diff()
            if e:
                return e
            if pos[0] >= len(toks) or toks[pos[0]][0] != ')':
                return 'Unbalanced "(" at %d' % (at + 1)
            pos[0] += 1
            return ''
        return 'Missing operand before "%s" at %d' % (t, at + 1)

    def inter():
        e = atom()
        while not e and pos[0] < len(toks) and toks[pos[0]][0] == '&':
            pos[0] += 1
            e = atom()
        return e

    def diff():
        e = inter()
        while not e and pos[0] < len(toks) and toks[pos[0]][0] in '+-':
            pos[0] += 1
            e = inter()
        return e

    e = diff()
    if e:
        return e
    if pos[0] < len(toks):
        t, at = toks[pos[0]]
        return ('Unbalanced ")" at %d' if t == ')' else 'Missing operator before position %d') % (at + 1)
    return ''


# ---- editor server

def serve(path, port, browser=True):
    import http.server
    import webbrowser
    schema = load_schema()

    class H(http.server.BaseHTTPRequestHandler):
        def send(self, code, body, ctype='application/json'):
            data = body if isinstance(body, bytes) else body.encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', ctype + '; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/':
                with open(os.path.join(HERE, 'editor.html'), 'rb') as f:
                    return self.send(200, f.read(), 'text/html')
            if self.path == '/schema':
                return self.send(200, json.dumps(schema))
            if self.path == '/file':
                if not path or not os.path.exists(path):
                    return self.send(200, json.dumps({'path': path, 'text': None}))
                with open(path, encoding='utf-8-sig') as f:
                    return self.send(200, json.dumps({'path': path, 'text': f.read()}))
            self.send(404, '{}')

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get('Content-Length', 0))).decode('utf-8')
            if self.path == '/validate':
                try:
                    doc = json.loads(body, strict=False)
                except json.JSONDecodeError as ex:
                    return self.send(200, json.dumps([['error', '.', 'JSON: %s' % ex]]))
                return self.send(200, json.dumps(validate(doc, schema)))
            if self.path == '/save' and path:
                # Only the file the editor was started on: the page cannot write anywhere else.
                with open(path, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(body)
                return self.send(200, json.dumps({'saved': path}))
            self.send(404, '{}')

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(('127.0.0.1', port), H)
    url = 'http://127.0.0.1:%d/' % srv.server_address[1]
    print('CD2Builder editor on %s%s  (Ctrl+C to stop)' % (url, ' for ' + path if path else ''))
    if browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


# ---- CLI

def read_json(path):
    with open(path, encoding='utf-8-sig') as f:
        return json.load(f, strict=False)   # CD2's own saves hold raw newlines in strings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('validate'); v.add_argument('files', nargs='+'); v.add_argument('--quiet', action='store_true')
    f = sub.add_parser('fmt'); f.add_argument('file'); f.add_argument('--out')
    e = sub.add_parser('edit'); e.add_argument('file', nargs='?'); e.add_argument('--port', type=int, default=0)
    e.add_argument('--no-browser', action='store_true')
    a = ap.parse_args(argv)
    if a.cmd == 'validate':
        schema, bad = load_schema(), 0
        for p in a.files:
            try:
                issues = validate(read_json(p), schema)
            except json.JSONDecodeError as ex:
                issues = [('error', '.', 'JSON: %s' % ex)]
            errs = sum(1 for i in issues if i[0] == 'error')
            bad += errs > 0
            print('%s: %d error(s), %d warning(s)' % (p, errs, len(issues) - errs))
            for lvl, path, msg in issues:
                if lvl == 'error' or not a.quiet:
                    print('  %-7s %s: %s' % (lvl, path, msg))
        return 1 if bad else 0
    if a.cmd == 'fmt':
        text = json.dumps(read_json(a.file), indent=4, ensure_ascii=False) + '\n'
        with open(a.out or a.file, 'w', encoding='utf-8', newline='\n') as out:
            out.write(text)
        return 0
    if a.cmd == 'edit':
        serve(os.path.abspath(a.file) if a.file else None, a.port, not a.no_browser)
        return 0


if __name__ == '__main__':
    sys.exit(main())
