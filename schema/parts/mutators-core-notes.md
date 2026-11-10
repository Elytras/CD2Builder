# mutators-core notes

Scope covered: Mutators/Mutators top level, Math, Logic, String, Lock, Trigger, Mutators/Select,
Mutators/Variables, the base classes, the value factory in CD2_Mutator_Manager.cpp, and (task item 3)
the Select-family option keys in Mutators/Mutators/Mission (ByBiome, ByMissionType, BySecondary, ByDNA,
ByMissionComplexity, ByMissionDuration, ByDDStage, By{Escort,Refinery,Sabo,Salvage}Phase, IfOnSpaceRig).
These Mission entries overlap the Mission reader's part; merge by name.

Not covered (other readers' folders, also Select-family): Resupply/ByResuppliesCalled, Time/Sequence,
Enemies/EnemyCooldown.

## Type token "any(select)"

Used for Select-family options. It means: the same type as the context the Select-family mutator is
used in (float, bool, string or stringlist), because MakeSelectorMutator sets OptionsBaseClass to that
base and each option goes through MakeMutator(<base>). Not a JSON-type branch, so not "oneof".

## How keys reach a mutator

- CD2_Mutator::ConfigureMutator (CustomDifficulty2/CD2_Mutator.cpp:126): non-object -> 'Internal Parsing
  Error - CD2 Mutator'; every key except 'Mutate' goes to HandleOption; if no errors, Validate. Validate
  returning invalid with no message -> 'Unspecified Mutator Error'. Base HandleOption ignores keys.
- Mutators that override ConfigureMutator (Abs/Ceil/Floor/Round/Clamp, Random, RandomPerMission, If,
  IfFloat, ByPlayerCount, ByMissionType, ByDDStage, BySecondary, IfOnSpaceRig) walk the object keys
  themselves and error 'Unexpected field "{field}"' on unknown keys.
- Nested params are built with MakeMutator (via CD2_Mutator::Make*Mutator), so they take literals /
  Mutate / Var, but an array is never per-player there ('Invalid Value/Mutate'; for a stringlist param
  an array is a FlatteningStringList).

## Case sensitivity [inferred]

bpdecomp renders `==`/`!=` on FString as the StriStri (case-insensitive) Kismet functions (AssetGen's
genueapi.py maps FString operators to *_StriStri) and prints case-sensitive compares as explicit
`EqualEqual_StrStr(...)`. TMap/TSet<FString> lookups are case-insensitive in UE. So Mutate names, param
keys, Select option keys, ByBiome/ByMissionType/BySecondary keys are case-insensitive. Exceptions seen:
ByMissionComplexity/ByMissionDuration 'Values' (EqualEqual_StrStr) and the Var-ready event match.

## Bugs / oddities found in the decompile (worth an author's note)

1. MakeFloatValue, implicit per-player array branch (CD2_Mutator_Manager.cpp:474-488): element errors
   go into a local `Errs`, and the function returns with neither Value nor Errors. A bad element in a
   top-level float per-player array fails silently. Bool/string versions report correctly.
2. Every Make*Value: JSON null returns silently (no value, no error).
3. ByPlayerCount::ConfigureMutator ends with `Errors = {}` (ByPlayerCount.cpp:122): its own errors are
   dropped. Decompile loop structure there is also tangled; treat as "errors not reported" [unclear].
4. ByDDStage: CurrentMissionIndex 2 (stage 3) returns Stage2Mutator (ByDDStage.cpp:37); Stage3 is
   parsed but unused.
5. MakeFloatArrayValue: an array of more than 4 elements becomes ONE per-player float (array length 1),
   not a 5+ element array (CD2_Mutator_Manager.cpp:669).
6. MakeRandIntervalsValue: `{"Var": ...}` is read as a range (only 'Mutate' is checked), giving
   "Range Must Specify 'Max'". In the array form, values are created even after element errors (the
   module still sees Errors).
7. RandomChoice / RandomChoicePerMission do not check Choices exists / is non-empty.
8. RandomChoicePerMission computes Salt = djb2(path) in Validate but Select seeds only with the PLS seed
   [unclear whether Salt is mixed in elsewhere; not in the decompiled Select].
9. IfFloat does not require Value / Then / Else at load (If and IfOnSpaceRig do).
10. Triggers take their input under any key name that is not reserved (first such key); 'Value' is a
    convention, not a requirement.

## Uncertain

- Float2String formatting: decompile shows `AsCurrency_Float(v, HalfToEven, false, true, 1, 324, 0, 3)`;
  the argument pattern matches Conv_FloatToText (grouping on, max 3 fractional digits).
- By*Phase enum keys use GetEnumeratorUserFriendlyName; I listed the decompiled enumerator names and
  assume the display names equal them.
- BySecondary: which key wins when several secondaries match depends on TMap key order.
