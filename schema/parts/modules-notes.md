# modules.json: notes

Built by reading `CustomDifficulty2/Modules/**` and the ModuleComponents they create. Each key's handling was traced by
following the bpdecomp `!= "Key"` / `NotEqual_StrStr` switch chains (with their gotos) to the `Make*Value` /
`RegisterSubmodule` call they reach. 36 modules: 26 regular (4 of them Events/*), 10 test (`"test": true`). Enemies and
EnemiesNoSync are not included (another reader has them).

## Patterns shared by most modules
- Submodule (CD2_Submodule) objects reject unknown keys with `Unrecognized Field "X"` (or a variant: `'X'`,
  `Unexpected field 'X'`, `Unknown field 'X'`). The exceptions that silently ignore unknown keys are **Dwarves**,
  **EWC2** (top level) and **EWC_Bad** (top level).
- `{name} Must Be a JSON Map` is formatted with an empty `{name}` almost everywhere, so the user sees
  ` Must Be a JSON Map [@path]`.
- Every error is suffixed ` [@<json path>]`, except DifficultySetting's `Unrecognized Field`, which carries the
  module's path, not the key's.
- Server-only parsing: Messages, SoundCues, WaveSpawners, SpecialEncounters and Mule run Update only on the server.
  Many values are created with ServerOnly=true (noted per field where it applies).
- Missing optional fields fall back through `Get*ValueOrDefault(key, default)`. Those fallbacks are recorded as
  `default` even though they are not part of GetDefaultState.

## GetDefaultState
Only Caps, EscortMule, Name and the test modules (TM2/TM3/TM4/SR/PLSTest, all `{"Value1":5.0,"Value2":3.1}`) return
JSON. Every other module returns null (the CD2_Module base). The test defaults don't fit their own schemas (SR wants a
number; TM2.Value2 is a string; PLSTest has no Value1/Value2 keys).

## Oddities worth knowing (likely bugs in CD2)
- Salvage Uplink/Refuel accepts `"ExtraDefenderBonus "` *with a trailing space*. Without the space, the key is an
  Unrecognized Field. With the space, the value is cached under the spaced name but read back without it, so it
  appears never to apply.
- SoundCues item: `SendOnChange` is read (CD2_SM_Cue.cpp:153) but is not an accepted key, so it errors if given.
- Resupply accepts HealAmount/Callable/Damage/KillRadius/Uses and does nothing with them.
- Escort ignores its value entirely.
- Messages `Type` other than Game/Normal/Delux/Developer/Streamer/Modder sends nothing, with no error.

## Unsure / not verified
- DifficultySetting value semantics (what each enum entry writes on UDifficultySetting) were taken from the enum
  names and the Set*Setting switch layout, not traced field by field. VeteranNormal/VeteranLarge take an inner
  switch (lines 680-683) whose target fields were not traced.
- floatarray / randinterval literal shapes belong to the mutator/value reader. This file only names the type.
- EDTest's per-ED object (CD2_SM_ED) is left to the Enemies reader; only its top-level key names are listed.
- Event-module keys and EWC wave-controller names are runtime asset names (Conv_ObjectToString of the event objects;
  BPGC asset names minus the trailing 2 chars), so no enum is given.
