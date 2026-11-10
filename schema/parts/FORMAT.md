# Schema fragment format (shared by all readers)

Source: bpdecomp C++ of CD2 under D:/tmp/cd2b/src/AssemblyStorm/ (decompiled cooked Blueprints; the truth).
Write ONE JSON file (your part name) here, plus <part>-notes.md for anything that doesn't fit.

## What is already known (don't re-derive)
- Top level: every key is a module name; module = Blueprint asset name under CustomDifficulty2/Modules (recursive, so
  Modules/TestModules/* are modules too). Unknown top-level key -> load error `Module "X" not found`.
- A module gets Enable/Update(State JSONValue, CD2_Value_Factory, Errors). Its default state is GetDefaultState's JSON.
- CD2_Value_Factory kinds: MakeFloatValue, MakeBooleanValue, MakeStringValue, MakeStringListValue, MakeFloatArrayValue,
  MakeRandIntervalsValue (impl in Mutators/CD2_Mutator_Manager.cpp). These are "mutatable" values: a literal, a
  {"Mutate": ...} object or a {"Var": ...} object. A field read straight off the JSONValue (GetNumber / GetString /
  GetBoolean / GetObject / GetArray) is NOT mutatable: record it as raw.
- Literal rules (MakeMutator): JSON string -> StringLiteral, number -> FloatLiteral, bool -> BooleanLiteral, object ->
  Mutate/Var, array -> FlatteningStringList only for stringlist; MakeFloatValue turns an array into implicit
  ByPlayerCount. The literal's class must match the base type (a number in a string field is a load error).
- Mutator name = asset name of each CD2_Mutator subclass under CustomDifficulty2/Mutators (recursive). A mutator reads
  its params in ConfigureMutator(MutatorJSON, MutatorFactory, Errors).
- Example of a module, read already: Modules/Caps.cpp -> fields MaxActiveCritters/MaxActiveSwarmers/MaxActiveEnemies,
  each MakeFloatValue (MinUpdateInterval 0.5); unknown field -> error `Unrecognized Field "X"`; non-object -> error
  "Caps Settings Must Be a Map"; GetDefaultState = {"MaxActiveCritters":60,"MaxActiveSwarmers":60,"MaxActiveEnemies":60}.

## Type vocabulary
Mutatable: "float", "bool", "string", "stringlist", "floatarray", "randinterval".
Raw (not mutatable): "rawnumber", "rawstring", "rawbool", "rawany".
Structure: "object" (with "fields"), "map" (free keys -> "values": <entry>), "array" ("items": <entry>),
"oneof" ("options": [<entry>, ...]) when the code branches on the JSON type.

## Entry shape
{"type": ..., "default": <JSON, if the code shows one>, "required": true|false, "enum": [...string choices the code
switches on...], "fields"/"values"/"items"/"options": ..., "unknown": "error"|"ignored" (objects/maps: what happens to
an unrecognized key), "notes": "short, only what matters to an author", "src": "<path relative to AssemblyStorm/>:<line>"}

Module file shape:  {"modules": {"<ModuleName>": <entry>, ...}}
Mutator file shape: {"mutators": {"<MutateName>": {"returns": "float|bool|string|stringlist|any(select)",
  "params": {"<Key>": <entry>}, "extra": <entry or null: what any other key means, e.g. Add's operands>,
  "unknown": "error|ignored", "notes": ..., "src": ...}}}

Rules: every claim cites src. If the decompiled flow is unclear, say so in notes ("unclear: ...") rather than guess.
Keep error message texts the code emits for a field in "notes" when short. JSON must parse (python -m json.tool).
