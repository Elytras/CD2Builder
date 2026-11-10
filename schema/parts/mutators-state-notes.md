# mutators-state: notes

Scope: 66 assets under `Mutators/Mutators/{Mission,DwarfState,Enemies,Time,Resources,Resupply}` -> 64 mutator names
(two names are duplicated, see below). Plus 14 value lists under `"enums"`.

## Additions to the FORMAT vocabulary

- `"type": "option"`: a param that is one branch of a Select mutator. It is built with
  `MakeOptionMutator` / `MutatorFactory->MakeMutator(json, OptionsBaseClass, ...)`. `OptionsBaseClass` is the base type the
  parent expects (set in `CD2_Mutator_Manager::MakeSelectorMutator`, Mutators/CD2_Mutator_Manager.cpp:254). So an option
  takes whatever the select mutator as a whole has to return.
- Each mutator has a `"base"` field naming its CD2_*_Mutator base class (`returns` restates it in FORMAT terms).
- A string in `"enum"` such as `"enums.EscortPhase"` points to the shared list.

## Rules that apply to every mutator here

- The base `CD2_Mutator::ConfigureMutator` (CD2_Mutator.cpp:126) calls `HandleOption` on every key except `Mutate`,
  matched case-insensitively (`NotEqual_StriStri`). It runs `Validate` only when no option produced an error.
- Every key switch is case-insensitive. The uasset imports show `NotEqual_StriStri` / `EqualEqual_StriStri`, not the
  case-sensitive form. Exceptions:
  - `Values` in ByMissionComplexity and ByMissionDuration uses `EqualEqual_StrStr`, so it is case-sensitive.
  - `Mutate` in TimeDelta uses `NotEqual_StrStr`, so it is case-sensitive.
  - TMap/TSet `FString` lookups (biome keys, mission types, salvage states, phase names) use UE's default hashing, which
    ignores case.
- Every error text gets ` [@<json path>]` appended.
- The params marked "mutatable" go through `MakeFloatMutator` / `MakeBooleanMutator` / `MakeStringMutator` /
  `MakeStringListMutator`, which call `MutatorFactory->MakeMutator(json, <base>)` directly. They do not go through
  `MakeFloatValue`. So the "array -> implicit ByPlayerCount" rule likely does not apply to mutator params. That rule
  belongs to the factory reader; unverified here.
- Some mutators find their monitor only `if (HasAuthority)`. On clients their Validate passes without the monitor.

## Bugs and quirks found (code-level, not tested in game)

- **Name collisions.** `MissionState` exists in both Mission/ and DwarfState/:
  - Mission/MissionState: `Evaluate` returns 0 (a stub).
  - DwarfState/MissionState: returns the RemainingIWs count, like IWsLeft.

  `MuleDistanceToDroppod` also exists in both folders, but the two copies are identical. CD2_Mod.cpp:324 builds the
  mutator registry as a `TMap` keyed by asset name, so the copy enumerated last wins. Which copy that is depends on
  asset-registry order: **unclear**.
- **ByDDStage**: `Stage3` is parsed, but deep-dive index 2 returns `Stage2Mutator` (ByDDStage.cpp:37). Stage 3 uses
  Stage2's option.
- **SquareWave**: `ConfigureMutator` ends with `Errors = {}` (SquareWave.cpp:232), so unknown keys and type errors are
  silently dropped. Its unknown-field message also formats an empty FText, giving `Unexpected field ""`.
- **Accumulate**: an unknown key reports `Unexpected field ""` because the key is not passed (Accumulate.cpp:20).
- **DuringDread**: `StartingAt` and `StoppingAfter` are parsed and evaluated but have no effect.
- **ByBiome**: with no `Default`, any 9+ biome entries make CrystalCaves the default. If CrystalCaves is the missing one,
  the default is null.
- **EnemyDistance**: its unknown-key error says only ED/EDs are allowed, but Type and Default are accepted too.
- **ElapsedExtraction** leaves out `EWC_Escort_EndMission_C`, which DuringExtraction includes.
- **ByEscortPhase / ByRefineryPhase / BySaboPhase**: if the monitor cast fails, Validate returns invalid without a message.
  The base then reports "Unspecified Mutator Error".
- **ByResuppliesCalled**: there is no presence check on `Values`. With an empty list, `Select` reads index 0 of an empty
  array: **unclear**.
- **TimeDelta**: `RateOfChange` defaults to 0. ByTime, which has the same formula, defaults it to 1.

## Uncertain

- Which `MissionState` wins (see above).
- Whether the `*Pct` values from the player monitor are 0..1. They are averaged from per-player items that were not read.
- Accumulate / Delta / Sequence semantics depend on the evaluation cadence, which the base classes own (another reader).
