# CD2Builder: plan (2026-11-10)

## Where it was left, and why it stalled

CD2Builder was a Python DSL (`cd2_schema.py`, 3.3k lines, one file) that writes CD2 difficulty JSON from Python
expressions, with a constant folder on top, plus one generator (`nightmares.py`). It stalled for three reasons:

- **Its schema was guessed from CD2's docs**, so the builder could not say whether CD2 accepts a file. It missed
  fields CD2 reads (`Caps.MaxActiveCritters`) and had no source for the ones it kept.
- **The optimiser was the hard part, and it broke.** False-positive constant folding made the last commits about
  turning parts of it off.
- **It was a build script, not a tool.** You could only write a difficulty in Python, nothing loaded or checked an
  existing file, and `import_from_json` could not rebuild nested objects.

## The goal

A tool to create CD2 difficulties: load one, edit it with the schema at hand, see CD2's own errors before the game
does, and export a file CD2 accepts. The DSL stays as one way to write a difficulty, not the only way.

## Stack

- **Python 3, standard library only**, as before. The old repo's stack was not why it stalled.
- **`cd2b.py`** holds the validator, the CLI and the editor's local server (`http.server`):
  - `validate FILE...` prints CD2's would-be load errors and warnings, with JSON paths.
  - `fmt FILE` re-indents.
  - `edit [FILE]` opens the editor in the browser.
- **`editor.html`** is one page with no build step and no CDN:
  - a tree of the difficulty with schema-aware inputs, "add field" menus and a mutator picker;
  - a raw JSON tab;
  - a live issue list, checked by the same Python validator through the local server.
- **The DSL** (`cd2_schema.py`) is kept. Its output goes through the same validator.
- **`decompile.py`** turns difficulties back into DSL scripts: every one in CD2's save (`CD2.sav`, in
  `FSD/Saved/SaveGames/Mods/`, only read) or one `.json`. Each mutator gets the most readable form that rebuilds it
  exactly (an operator, a DSL helper, `Mut`, a dict), repeated subtrees become names, and each script is run and
  compared with the original as CD2 reads it (float32).

## Schema source

- **`schema/cd2.json`** is the schema, read off CD2's decompiled Blueprints, not its docs.
  - The pak is `custom-difficulty2.pak`, sha1 `70a6123b…` (see `schema/CD2_SCHEMA.md`).
  - It was extracted with ue-mods' `pak.py` and decompiled with `bpdecomp`.
  - Every module, field, mutator and parameter carries a `src` back to the decompiled line.
- **`schema/CD2_SCHEMA.md`** is the human reference: how CD2 parses and what it ignores.
- **Regenerating after a CD2 update** follows the steps in CD2_SCHEMA.md ("Regenerating"). Diff the new
  `cd2.json` against the old one.

## Validation

The validator follows CD2's own rules (see CD2_SCHEMA.md):

- A top-level key must be a module; anything else is CD2's `Module "X" not found`.
- A module's fields come from its parser.
- A value is a literal of the right kind, a `{"Mutate": …}` whose mutator returns that kind, or a `{"Var": …}`
  naming a variable of that type.
- Keys match case-insensitively, as CD2's string switches do. A non-canonical case is a warning.

Levels:

- **error**: CD2 reports a load error, or the value cannot work.
- **warning**: CD2 accepts it, but it does nothing or is likely a mistake. For example, a key CD2 ignores, or an
  enum string CD2 compares and never matches.

## ECD2A hooks

ECD2A is a CD2 module (`ElytrasAddon`) that CD2 finds by asset name, like its own modules. The builder models it the
same way: `schema/ecd2a.json` `extends` `cd2` and adds modules and shared `defs`; it never copies or edits CD2's
schema. Any later addon is one more extension file. The file lists what of ECD2A is pinned down (`status.done`) and
what is still open (`status.open`).

## Tests

- `python test_cd2b.py` runs the self-check:
  - known-good files load clean: `nightmares.json`, ue-mods' drgq test difficulty, and ECD2A's `nightmares-keys.json`;
  - hand-broken ones give the error CD2 would;
  - the optimiser's past fold bugs, and the decompiler's round trip.
- `python fuzz_fold.py [SEED]` fuzzes the constant folder against CD2's arithmetic.
- In game, through ue-mods' drgq:
  - submit a job whose `difficulty` is a builder export;
  - read CD2's `OnStateChanged … Errors` line;
  - an empty error list means CD2 took it.

## First slice (this branch)

1. Schema from decompilation, plus the ECD2A extension.
2. The validator and CLI.
3. The editor: load, edit, validate and export.
4. Self-check, then one drgq load check.

Later:

- Generate the DSL's dataclasses from `cd2.json`, so the two cannot drift.
- Read a difficulty straight from CD2.sav in the editor, not only through `decompile.py`.
- A standalone (no-server) build of the editor to share with other CD2 authors. This needs the validator in JS.
