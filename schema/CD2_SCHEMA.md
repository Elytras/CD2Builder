# How CD2 reads a difficulty

This is read off CD2's decompiled Blueprints, not off its docs. The per-key tables are in [REFERENCE.md](REFERENCE.md)
(generated from [cd2.json](cd2.json)). Every entry there has a `src`: a line in the decompiled tree `AssemblyStorm/`
(see "Regenerating"). `[V]` marks a claim checked in that source by hand. The rest was read by four parallel readers,
each citing its line.

## The build this describes

| | |
| --- | --- |
| Pak | `custom-difficulty2.pak` (the copy in the user's MINT profile), sha1 `70a6123b3206d638d36b937e7a45aefb48a906ee` |
| Mod info | `CD2_Mod::GetModInfo`: "Custom Difficulty v2", version "2" (the pak carries no finer version) |
| Decompiler | ue-mods `AssetGen/tools/bpdecomp_emit.py`, 431 assets, 4 kept cooked (none of them parser code) |

## The top level

- **Every top-level key is a module.** A module is a Blueprint asset under
  `/Game/_AssemblyStorm/CustomDifficulty2/Modules`, scanned recursively. Its key is the asset name
  (`CD2_Mod.h:14`, `CD2_Mod.cpp` module scan) `[V]`. That includes `Modules/TestModules/*`: `Enemies` lives there.
- **An unknown top-level key** is the load error `Module "X" not found` (`CD2_Mod.cpp`) `[V]`.
- **A root that is not an object** is the error `Root must be an object` `[V]`.
- **Addons extend CD2 by being modules.** A mod that cooks a CD2_Module subclass into that folder becomes a top-level
  key. ECD2A's `ElytrasAddon` works that way (`/Game/_AssemblyStorm/CustomDifficulty2/Modules/ElytrasAddon`). The
  builder's extension files (`ecd2a.json`) follow the same model.
- **The module lifecycle.** A key present in a state Enables its module, or Updates it if it was already enabled. A
  module whose key disappears is Disabled. Errors from every module are collected into `OnStateChanged(State, Errors)`.
  A difficulty with errors can still be partly applied.
- **Key matching ignores case.** Field switches compile to `NotEqual_StriStri` (the uasset imports show it), and the
  maps CD2 keeps are `TMap<FString>` `[V]`. A few compares are case-sensitive: `Values` in ByMissionComplexity and
  ByMissionDuration, `Mutate` in TimeDelta, and some in `CD2_SM_Enemy`. The builder warns on any non-canonical case.

## Values

A field CD2 builds through its value factory (`CD2_Value_Factory`: `MakeFloatValue`, `MakeBooleanValue`,
`MakeStringValue`, `MakeStringListValue`, `MakeFloatArrayValue`, `MakeRandIntervalsValue`, in
`Mutators/CD2_Mutator_Manager.cpp`) is **mutatable**. It takes three forms:

- **a literal.** A JSON string becomes `StringLiteral`, a number `FloatLiteral`, a bool `BooleanLiteral` (`MakeMutator`)
  `[V]`. The literal must match the field's type: a number in a string field is
  `Mutate Type Error Expected "…" Found "FloatLiteral"` `[V]`.
- **`{"Mutate": "<Name>", …params}`.** Mutator names are the asset names of every `CD2_Mutator` subclass under
  `…/CustomDifficulty2/Mutators`, scanned recursively `[V]`. The error texts:
  - an unknown name: `Unrecognized Mutate X.`;
  - a non-string `Mutate`: `Mutate value must be a String name.`;
  - an object with neither `Mutate` nor `Var`: `Expected Mutator Object, but no 'Mutate' field found.` `[V]`.
- **`{"Var": "<name>"}`.** It reads a variable defined under `Vars`, and the variable's `Type` picks its kind:
  `Float` float, `Boolean` bool, `String` string, `Strings` stringlist.

Rules by type:

| Type | Literal | JSON array |
| --- | --- | --- |
| float | number | per-player-count value (implicit `ByPlayerCount`: entry `players - 1`, the last repeats) |
| bool, string | `true`/`false`, `"text"` | per-player-count, as float |
| stringlist | a list; inside a mutator a string is wrapped to a one-entry list | the list itself; nested arrays flatten |
| floatarray | one number = a one-entry array | up to 4 entries = 4 floats; **5 or more = one per-player value** |
| randinterval | a number = fixed | a list of `{"Min","Max","Weight"?}` ranges; a single range object works too |

Two further rules:

- **Inside a mutator there are no per-player arrays.** A parameter is built with `MakeMutator`, not `Make*Value`, so an
  array there is `Invalid Value/Mutate`. The one exception is a stringlist parameter, where an array is a list.
- **The Select family returns any type.** `If`, `IfFloat`, `Select`, `ByPlayerCount`, `ByBiome`, `ByMissionType`,
  `By*Phase`, `RandomChoice`, and the others under `returns any` in REFERENCE.md return whatever type the field wants.
  Their options are built as that type (`MakeSelectorMutator`) `[V]`.

## What CD2 silently ignores or gets wrong

These are found in the decompiled code and not tested in game. The validator turns the first group into warnings.

- **No error, no effect:**
  - JSON `null` as a value gives no value and no error.
  - Unknown keys are ignored in `Dwarves`, the top level of `EWC2`, and inside a `randinterval` range.
  - `Resupply` accepts `HealAmount`, `Callable`, `Damage`, `KillRadius` and `Uses`, and uses none of them.
  - `Escort` ignores its value.
  - `Messages` with an unknown `Type` sends nothing.
  - `DuringDread`'s `StartingAt` / `StoppingAfter` have no effect.
- **Errors that are lost:**
  - A bad entry in a top-level float per-player array fails silently (`CD2_Mutator_Manager.cpp:474`).
  - `ByPlayerCount` ends with `Errors = {}` (`ByPlayerCount.cpp:122`) `[V]`.
  - `SquareWave` drops its errors too.
- **Bugs an author would trip on:**
  - `Salvage` Uplink and Refuel accept `"ExtraDefenderBonus "` only with a trailing space, then read it back without
    one. The key works in neither spelling (`CD2_SM_SalvageUplink.cpp:110` vs `:242`) `[V]`.
  - `ByDDStage`: deep-dive stage 3 uses `Stage2`'s option. `Stage3` is parsed and never chosen
    (`ByDDStage.cpp:37`) `[V]`.
  - `SoundCues`: `SendOnChange` is read, but rejected as an unrecognised field.
  - `MissionState` exists twice, a stub returning 0 and a DwarfState copy. Which one a difficulty gets depends on
    asset-registry order.
  - Triggers take their input under any unreserved key. `Value` is a convention.
  - `IfFloat` does not require `Value`, `Then` or `Else`. Exactly one comparison key (`==`, `>=`, `>`, `<=`, `<`) is
    required `[V]`.
- **Enemies:**
  - The enemy health key is `HealthMultiplier` (`CD2_EC_Health.h:7`). `HealthRaw` sets it absolutely, and the two
    exclude each other `[V]`.
  - `Health` is not a key: `Unrecognized Enemy Field 'Health'` (`CD2_SM_Enemy.cpp:233`) `[V]`.

## Registries (names an enemy, projectile or material field takes)

CD2 fills its ED, PRJ and Material registries at run time from the asset registry, so any mounted pak can add names.
Its source holds no table. The lists in `cd2.json` are suggestions, not a closed set: the editor offers them, and the
validator does not refuse other names.

| List | Size | Source |
| --- | --- | --- |
| `ED` | 95 | vanilla, from the Dumper-7 dump and the game pak |
| `ED_mods` | 320 | other mods' descriptors (yinnys_addon, EEE's NewEnemies) |
| `PRJ` | 117 | projectiles |
| `Material` | 3388 | a lower bound |

A new enemy name under `Enemies` needs `Base`, a vanilla ED. An existing ED name edits that ED in place.

## Regenerating after a CD2 update

1. Extract the new pak (in Git Bash, set `MSYS_NO_PATHCONV=1`, or the `/Game` argument gets mangled):

   ```
   python AssetGen/tools/pak.py extract <custom-difficulty2.pak> <dir>/pak Content
   python AssetGen/tools/bpdecomp_emit.py <dir>/pak/FSD/Content <dir>/src --mod /Game/_AssemblyStorm=AssemblyStorm
   ```

   Run both from ue-mods.

2. Re-read the parsers into `schema/parts/*.json` in the format of `parts/FORMAT.md`. One reader each covered: the
   non-enemy modules, `Enemies`, the core mutators plus the value factory, and the game-state mutators.
3. Build `cd2.json` and `REFERENCE.md`, then check:

   ```
   python schema/merge.py
   python test_cd2b.py
   ```

4. Diff `cd2.json` against the previous one. A changed `src` line alone is noise; a changed field, type or default is
   the update.
