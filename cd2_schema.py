from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional, TypeAlias, Sequence

# ==========================================
# CONSTANTS & VALIDATION SETS
# ==========================================

_VALID_DNA_MISSIONS = {
    "x",
    "DeepScan",
    "Egg",
    "Elimination",
    "Escort",
    "Mining",
    "PE",
    "Refinery",
    "Sabotage",
    "Salvage",
}

_VALID_LENGTH_LEVELS = {"x", "1", "2", "3"}

_VALID_COMPLEXITY_LEVELS = {"x", "1", "2", "3"}

_VALID_BIOME_ALIASES = {
    "AzureWeald",
    "CrystalCaves",
    "CrystallineCaverns",
    "DenseBiozone",
    "Biozone",
    "FungusBogs",
    "HollowBough",
    "IceCaves",
    "GlacialStrata",
    "MagmaCaves",
    "MagmaCore",
    "RadioactiveExclusionZone",
    "REZ",
    "SaltCaves",
    "SaltPits",
    "SandblastedCorridors",
    "Sandblasted",
}

_VALID_SECONDARY_OBJS = {
    "OBJ_2nd_Mine_Dystrum",
    "OBJ_2nd_Mine_Hollomite",
    "OBJ_2nd_KillFleas",
    "OBJ_2nd_Find_Gunkseed",
    "OBJ_2nd_Find_Fossil",
    "OBJ_2nd_Find_Ebonut",
    "OBJ_2nd_Find_BooloCap",
    "OBJ_2nd_Find_ApocaBloom",
    "OBJ_2nd_DestroyEggs",
    "OBJ_2nd_DestroyBhaBarnacles",
    "OBJ_DD_RepairMinimules",
    "OBJ_DD_Defense",
    "OBJ_DD_DeepScan",
    "OBJ_DD_Morkite",
    "OBJ_DD_Elimination_Eggs",
    "OBJ_DD_AlienEggs",
    "OBJ_DD_MorkiteWell",
}

_VALID_ESCORT_PHASES = {
    "InGarage",
    "Stationary",
    "Moving",
    "WaitingForFuel",
    "FinalEventA",
    "FinalEventB",
    "FinalEventC",
    "FinalEventD",
    "Finished",
}

_VALID_REFINERY_PHASES = {
    "Landing",
    "ConnectingPipes",
    "PipesConnected",
    "Refining",
    "RefiningStalled",
    "RefiningComplete",
    "RocketLaunched",
}

_VALID_SABO_PHASES = {
    "Hacking",
    "BetweenHacks",
    "HackingFinished",
    "Phase1Vent",
    "Phase1Eye",
    "Phase2Vent",
    "Phase2Eye",
    "Phase3Vent",
    "Phase3Eye",
    "Finished",
}

_VALID_SALVAGE_PHASES = {
    "Mules",
    "PreUplink",
    "Uplink",
    "PreRefuel",
    "Refuel",
    "Finished",
}

# ==========================================
# PART 2: SERIALIZER BASE CLASS
# ==========================================


@dataclass
class CD2Object:
    def to_dict(self) -> Any:
        result = {}
        for key in self.__annotations__:
            val = getattr(self, key, None)
            if val is None:
                continue

            # Special handling for Vars if it's a list of Variables
            if key == "Vars" and isinstance(val, list):
                val = {v.Name: v.definition for v in val if isinstance(v, Variable)}

            result[key] = self._serialize(val)
        return result

    def _serialize(self, val):
        if isinstance(val, CD2Object):
            return val.to_dict()
        elif isinstance(val, list):
            return [self._serialize(item) for item in val]
        elif isinstance(val, dict):
            return {k: self._serialize(v) for k, v in val.items()}
        return val

    def json(self, indent: Optional[int] = 4) -> str:
        """Returns the JSON string representation of the object."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str, indent: Optional[int] = 4):
        """Saves the JSON representation to a file."""
        with open(path, "w") as f:
            f.write(self.json(indent))


# ==========================================
# PART 3: TYPE DEFINITIONS & SCHEMA
# ==========================================

# Mutator-compatible type aliases
# Lists in MStr/MFloat exist because they are implicitly ByPlayerCount mutators
# These types include Variable and Expression to support direct use in schema without
MStr: TypeAlias = Union[str, dict, List[str], "Expression"]
MStrArray: TypeAlias = Union[Sequence[Union[str, dict, "Expression"]], dict]
MFloat: TypeAlias = Union[
    float, int, dict, List[Union[float, int]], "Expression", "Variable"
]
MBool: TypeAlias = Union[bool, dict, "Expression", "Variable"]


@dataclass
class VarDefinition(CD2Object):
    Type: MStr
    Value: Any
    Watch: Optional[MBool] = None


def _variadic_mutator(name: str, *args, **kwargs) -> Expression:
    """Helper for Add, Mul, Or, And to map positional args to A, B, C, etc.

    Automatically maps positional arguments to letter keys (A, B, C, ...)
    while preserving named keyword arguments and expression aliases.
    """
    result = {"Mutate": name}

    # Fill named kwargs
    result.update(kwargs)

    # Fill positional args, finding the first unused letter key

    def get_next_key():
        for i in range(26):
            k = chr(65 + i)
            if k not in result:
                return k
        raise ValueError("Too many positional arguments (A-Z exhausted)")

    for val in args:
        # Check if expression has an Alias
        alias = getattr(val, "alias", None)
        if alias:
            result[alias] = val
        else:
            result[get_next_key()] = val

    return Expression(result)


# ==========================================
# PART 4: EXPRESSION CLASS & OPERATOR OVERLOADING
# ==========================================


class Expression(CD2Object):
    """Wraps a mutator dictionary to allow operator overloading."""

    def __init__(self, content, alias=None):
        self.content = content
        self.alias = alias
        self.no_optimize = False  # Flag to disable optimization on this expression

    def Alias(self, name: str):
        self.alias = name
        return self

    def SetOptimize(self, optimize: bool):
        self.no_optimize = not (optimize)
        return self

    def NoOptimize(self):
        """Mark this expression to skip optimization."""
        return self.SetOptimize(False)

    def to_dict(self):
        return self._serialize(self.content)

    def __class_getitem__(cls, item):
        """Allow Expression[SomeType] for type hints."""
        return cls

    # Flattening helper
    def _flatten_op(self, other, mutate_name, constructor):
        collected_args = []
        collected_kwargs = {}

        def collect(obj):
            # If obj is Expression, check alias
            alias = getattr(obj, "alias", None)

            # Peel content
            content = obj.content if isinstance(obj, Expression) else obj

            # If it's a flattened structure of same type, unpack it
            is_flattenable = (
                isinstance(content, dict) and content.get("Mutate") == mutate_name
            )

            if is_flattenable and not alias:
                # Only flatten operands if the node itself has no alias.
                # Named aliases should be preserved as-is (e.g., Double=Mul(A,B)).

                # Separate positional vs named
                positional = []
                for k, v in content.items():
                    if k == "Mutate":
                        continue
                    if len(k) == 1 and k.isupper():
                        positional.append((k, v))
                    else:
                        # Merge named args
                        collected_kwargs[k] = v

                # Sort positional by key to maintain stable order A,B,C
                positional.sort(key=lambda x: x[0])
                for _, v in positional:
                    collected_args.append(v)
            else:
                # Treat as a leaf unit
                if alias:
                    collected_kwargs[alias] = obj
                else:
                    collected_args.append(obj)

        collect(self)
        collect(other)
        return constructor(*collected_args, **collected_kwargs)

    # Arithmetic
    def __add__(self, other):
        return self._flatten_op(other, "Add", Add)

    def __radd__(self, other):
        return self._flatten_op(other, "Add", Add)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __mul__(self, other):
        return self._flatten_op(other, "Multiply", Mul)

    def __rmul__(self, other):
        return self._flatten_op(other, "Multiply", Mul)

    def __truediv__(self, other):
        return Div(self, other)

    def __rtruediv__(self, other):
        return Div(other, self)

    def __pow__(self, other):
        return Pow(self, other)

    def __rpow__(self, other):
        return Pow(other, self)

    def __mod__(self, other):
        return Modulo(self, other)

    def __rmod__(self, other):
        return Modulo(other, self)

    def __neg__(self):
        return Negate(self)

    def __abs__(self):
        return Abs(self)

    # Comparison
    def __eq__(self, other):  # type: ignore
        return Eq(self, other)

    def __ne__(self, other):  # type: ignore
        return Neq(self, other)

    def __lt__(self, other):
        return Lt(self, other)

    def __le__(self, other):
        return Lte(self, other)

    def __gt__(self, other):
        return Gt(self, other)

    def __ge__(self, other):
        return Gte(self, other)

    # Logic
    def __and__(self, other):
        return self._flatten_op(other, "And", And)

    def __rand__(self, other):
        return self._flatten_op(other, "And", And)

    def __or__(self, other):
        return self._flatten_op(other, "Or", Or)

    def __ror__(self, other):
        return self._flatten_op(other, "Or", Or)

    def __invert__(self):
        return Not(self)


class Variable(Expression):
    """
    Defines a variable easier:
    my_var = Variable("MyVar", 10)
    # Use 'my_var' in mutators -> references it.
    # Use 'DefineVars(my_var)' in Vars section -> defines it.
    """

    def __init__(
        self,
        name: str,
        value: Any,
        type_str: Optional[str] = None,
        watch: Optional[bool] = None,
    ):
        super().__init__({"Var": name})
        self.Name = name
        self.Value = value
        self.Watch = watch

        if type_str:
            self.Type = type_str
        else:
            self.Type = self._infer_type(value)

    def _infer_type(self, value):
        if isinstance(value, bool):
            return "Boolean"
        if isinstance(value, str):
            return "String"
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return "Strings"
        if isinstance(value, (int, float)):
            return "Float"
        if isinstance(value, dict):
            mut = value.get("Mutate")
            if mut in [
                "And",
                "Or",
                "Not",
                "Nand",
                "Nor",
                "Xor",
                "TriggerOnce",
                "TriggerNTimes",
                "TriggerSometimes",
                "TriggerCooldown",
                "TriggerFixedDuration",
                "TriggerSustain",
                "TriggerDelay",
                "TriggerOnChange",
                "OnKill",
                "OnSpawn",
                "IsSolo",
                "PostDrillevator",
                "IsExtraction",
                "Eq",
                "Neq",
                "Gt",
                "Lt",
                "Gte",
                "Lte",
            ]:
                return "Boolean"
            # Check for string-returning mutators
            if mut in ["Float2String", "Int2String", "Join", "Countdown", "LockString"]:
                return "String"
        return "Float"

    def to_dict(self):
        return {"Var": self.Name}

    def NoOptimize(self):
        """Mark this variable to skip optimization."""
        self.no_optimize = True
        return self

    @property
    def definition(self):
        return VarDefinition(self.Type, self.Value, self.Watch)


# ==========================================
# UPDATED TYPE ALIASES (after Expression & Variable definitions)
# ==========================================
# These override the forward-reference versions from earlier.
# Now Expression and Variable are proper runtime types for type checking.


def DefineVars(*vars) -> Dict[str, VarDefinition]:
    """Helper to convert Variable objects into a Vars dictionary.

    Args:
        *vars: Variable objects to convert

    Returns:
        Dictionary mapping variable names to their definitions
    """
    return {v.Name: v.definition for v in vars}


# ==========================================
# PART 5: CORE HELPER FUNCTIONS
# ==========================================


def Mut(name: str, **kwargs) -> Expression:
    """Generic wrapper for any mutator.

    Args:
        name: The mutator name
        **kwargs: Mutator parameters (None values are filtered out)

    Returns:
        Expression wrapping the mutator dictionary
    """
    return Expression(
        {"Mutate": name, **{k: v for k, v in kwargs.items() if v is not None}}
    )


def Var(name: str) -> Expression:
    """References a variable defined in 'Vars'.

    Args:
        name: Variable name to reference

    Returns:
        Expression that references the variable
    """
    return Expression({"Var": name})


@dataclass
class WeightedBracket(CD2Object):
    weight: MFloat
    min: MFloat
    max: MFloat

    def __init__(self, weight: MFloat, min: MFloat, max: MFloat):
        self.weight = weight

        # Case 1: Both numeric → enforce ordering unambiguously.
        if isinstance(min, (int, float)) and isinstance(max, (int, float)):
            if min <= max:
                self.min = float(min)
                self.max = float(max)
            else:
                # Swap automatically
                self.min = float(max)
                self.max = float(min)
            self.ambiguous = False
            return

        # Case 2: At least one side is an Expression, Variable, dict, etc.
        #         Ordering is ambiguous → record that fact.
        self.min = min
        self.max = max
        self.ambiguous = True


@dataclass
class MinMax(CD2Object):
    Min: MFloat
    Max: MFloat

    def __init__(self, min: MFloat, max: MFloat):
        # Case 1: Both numeric → enforce ordering unambiguously.
        if isinstance(min, (int, float)) and isinstance(max, (int, float)):
            if min <= max:
                self.Min = float(min)
                self.Max = float(max)
            else:
                # Swap automatically
            self.ambiguous = False
                self.Min = float(max)
                self.Max = float(min)
            return

        # Case 2: At least one side is an Expression, Variable, dict, etc.
        self.ambiguous = True
        #         Ordering is ambiguous → keep as-is.
        self.Min = min
        self.Max = max


@dataclass
class PoolOperation(CD2Object):
    clear: Optional[MBool] = None
    add: Optional[MStrArray] = None
    remove: Optional[MStrArray] = None


@dataclass
class UsableSettings(CD2Object):
    Duration: Optional[MFloat] = (
        None  # Some usables might not have duration? Making optional to be safe
    )
    Text: Optional[MStr] = None
    CoopUse: Optional[MBool] = None
    CoopUseMultiplier: Optional[MFloat] = None
    Priority: Optional[MFloat] = None
    ResetOnFail: Optional[MBool] = None
    UseText: Optional[MStr] = None
    UseCooldown: Optional[MFloat] = None
    UseDuration: Optional[MFloat] = None


@dataclass
class Message(CD2Object):
    Send: Optional[MBool]  # False
    SendOnChange: Optional[MBool]  # False
    Type: Optional[MStr]  #  Game
    Sender: Optional[MStr]  #  CD2
    Message: MStr


@dataclass
class SoundCue(CD2Object):
    Play: MBool
    Cue: MStr


@dataclass
class SpecialEncounter(CD2Object):
    Enemy: MStr
    BaseChance: MFloat
    CanSpawnInDeepDive: Optional[MBool] = None


# --- Enemy Sub-Modules ---


@dataclass
class EnemyResistances(CD2Object):
    DamageMultiplier: Optional[MFloat] = None
    FireDamageMultiplier: Optional[MFloat] = None
    ExplosionDamageMultiplier: Optional[MFloat] = None
    ElectricDamageMultiplier: Optional[MFloat] = None
    ColdDamageMultiplier: Optional[MFloat] = None
    CorrosiveDamageMultiplier: Optional[MFloat] = None
    PoisonDamageMultiplier: Optional[MFloat] = None
    PiercingDamageMultiplier: Optional[MFloat] = None
    KineticDamageMultiplier: Optional[MFloat] = None


@dataclass
class EnemyMovement(CD2Object):
    MaxPawnSpeed: Optional[MFloat] = None
    MaxAcceleration: Optional[MFloat] = None
    MaxBrakingDeceleration: Optional[MFloat] = None
    AlignDirectionSpeed: Optional[MFloat] = None
    AlignToTargetMinRequiredAngle: Optional[MFloat] = None
    MaxStrafeDistance: Optional[MFloat] = None
    StrafeSpeed: Optional[MFloat] = None


@dataclass
class EnemyTemperature(CD2Object):
    FreezeTemperature: Optional[MFloat] = None
    BurnTemperature: Optional[MFloat] = None
    DieIfFrozen: Optional[MBool] = None
    WarmingRate: Optional[MFloat] = None
    CoolingRate: Optional[MFloat] = None


@dataclass
class DeathSpawner(CD2Object):
    ED: MStr
    OnDeathCount: Optional[MFloat] = None


@dataclass
class JellyBreederSettings(CD2Object):
    EnemyToSpawn: Optional[MStr] = None
    MaxJellies: Optional[MFloat] = None
    EggTime: Optional[MFloat] = None
    TimeBetweenBursts: Optional[MFloat] = None
    MultiplierOnHighPlayerCount: Optional[MFloat] = None


@dataclass
class CaveLeechSettings(CD2Object):
    GrapDelay: Optional[MFloat] = None
    TentacleSpeed: Optional[MFloat] = None
    TentaclePullSpeed: Optional[MFloat] = None
    MaxDistanceXY: Optional[MFloat] = None
    BiteDamage: Optional[MFloat] = None
    Silent: Optional[MBool] = None


# --- Settings Modules ---


@dataclass
class DwarvesSettings(CD2Object):
    RegenHealthPerSecond: Optional[MFloat] = None
    RegenDelayAfterDamage: Optional[MFloat] = None
    FallDamageStartVelocity: Optional[MFloat] = None  # Default 1000
    FallDamageModifier: Optional[MFloat] = None  # Default 0.175
    Test: Optional[MBool] = None  # Actual field in CD2


@dataclass
class WarningSettings(CD2Object):
    Banned: Optional[Sequence[MStr]] = None


@dataclass
class CapSettings(CD2Object):
    MaxActiveSwarmers: Optional[MFloat] = None
    MaxActiveEnemies: Optional[MFloat] = None


@dataclass
class ResupplySettings(CD2Object):
    Cost: Optional[MFloat] = None


@dataclass
class EscortMuleSettings(CD2Object):
    FriendlyFireModifier: Optional[MFloat] = None
    BigHitDamageModifier: Optional[MFloat] = None
    BigHitDamageReductionThreshold: Optional[MFloat] = None
    NeutralDamageModifier: Optional[MFloat] = None


@dataclass
class MiniMulesSettings(CD2Object):
    ScanUsable: Optional[UsableSettings] = None
    RepairUsable: Optional[UsableSettings] = None
    LegsRequired: Optional[MFloat] = None
    LegsPerMule: Optional[MFloat] = None
    LegDistance: Optional[MFloat] = None
    Count: Optional[MFloat] = None
    NitraToGive: Optional[MFloat] = None


@dataclass
class DefenseSettings(CD2Object):
    RepairUsable: Optional[UsableSettings] = None
    Scale: Optional[MFloat] = None
    Duration: Optional[MFloat] = None
    ExtraDefenderBonus: Optional[MFloat] = None
    DisableLeaveShout: Optional[MBool] = None
    LeavePenaltyMultiplier: Optional[MFloat] = None


@dataclass
class SalvageSettings(CD2Object):
    MiniMules: Optional[MiniMulesSettings] = None
    Uplink: Optional[DefenseSettings] = None
    Refuel: Optional[DefenseSettings] = None


@dataclass
class GlobalDifficultySettings(CD2Object):
    BaseHazard: Optional[MStr] = None

    ExtraLargeEnemyDamageResistance: Optional[
        Union[MFloat, List[Union[float, int]]]
    ] = None
    ExtraLargeEnemyDamageResistanceB: Optional[
        Union[MFloat, List[Union[float, int]]]
    ] = None
    ExtraLargeEnemyDamageResistanceC: Optional[
        Union[MFloat, List[Union[float, int]]]
    ] = None
    ExtraLargeEnemyDamageResistanceD: Optional[
        Union[MFloat, List[Union[float, int]]]
    ] = None
    EnemyDamageResistance: Optional[Union[MFloat, List[Union[float, int]]]] = None

    EnemyDamageModifier: Optional[MFloat] = None
    EnemyCountModifier: Optional[MFloat] = None

    EncounterDifficulty: Optional[List[WeightedBracket]] = None
    StationaryDifficulty: Optional[List[WeightedBracket]] = None
    EnemyWaveInterval: Optional[List[WeightedBracket]] = None
    EnemyNormalWaveInterval: Optional[List[WeightedBracket]] = None
    EnemyNormalWaveDifficulty: Optional[List[WeightedBracket]] = None
    EnemyDiversity: Optional[List[WeightedBracket]] = None
    StationaryEnemyDiversity: Optional[List[WeightedBracket]] = None
    VeteranNormal: Optional[List[WeightedBracket]] = None

    VeteranLarge: Optional[MFloat] = None
    EnvironmentalDamageModifier: Optional[MFloat] = None
    PointExtractionScalar: Optional[MFloat] = None
    FriendlyFireModifier: Optional[MFloat] = None
    SpeedModifier: Optional[MFloat] = None
    AttackCooldownModifier: Optional[MFloat] = None
    ProjectileSpeedModifier: Optional[MFloat] = None
    HealthRegenerationMax: Optional[MFloat] = None
    ReviveHealthRatio: Optional[MFloat] = None


# --- Complex Objects ---


@dataclass
class PoolSettings(CD2Object):
    MinPoolSize: Optional[MFloat] = None
    DisruptiveEnemyPoolCount: Optional[MinMax] = None
    StationaryEnemyCount: Optional[MinMax] = None
    EnemyPool: Optional[PoolOperation] = None
    StationaryPool: Optional[PoolOperation] = None


@dataclass
class WaveSpawner(CD2Object):
    Name: Optional[MStr] = None
    Enabled: Optional[MBool] = None
    SpawnOnEnable: Optional[MBool] = None
    PauseOnDisable: Optional[MBool] = None
    UnlockInterval: Optional[MBool] = None
    Interval: Optional[MFloat] = None
    Enemies: Optional[MStrArray] = None
    Difficulty: Optional[MFloat] = None
    Distance: Optional[MFloat] = None
    Locations: Optional[MFloat] = None


@dataclass
class EnemyDescriptor(CD2Object):
    Base: Optional[MStr] = None
    DisplayName: Optional[MStr] = None
    CanBeUsedInEncounters: Optional[MBool] = None
    CanBeUsedForConstantPressure: Optional[MBool] = None
    DifficultyRating: Optional[MFloat] = None
    Rarity: Optional[MFloat] = None
    MaxSpawnCount: Optional[MFloat] = None
    MinSpawnCount: Optional[MFloat] = None
    SpawnAmountModifier: Optional[MFloat] = None
    SpawnSpread: Optional[MFloat] = None
    NoSpawnWithin: Optional[MFloat] = None

    HealthMultiplier: Optional[MFloat] = None  # Exclusive with HealthRaw
    HealthRaw: Optional[MFloat] = None  # Exclusive with HealthMultiplier

    Scale: Optional[MFloat] = None
    TimeDilation: Optional[MFloat] = None
    ShowHealthBar: Optional[MBool] = None
    Significance: Optional[MStr] = None
    Elite: Optional[MFloat] = None
    ForceEliteBase: Optional[MStr] = None
    IsBossFight: Optional[MBool] = None

    Resistances: Optional[EnemyResistances] = None
    Movement: Optional[EnemyMovement] = None
    Courage: Optional[MFloat] = None
    WeakpointHP: Optional[MFloat] = None
    StaggerDurationMultiplier: Optional[MFloat] = None
    Temperature: Optional[EnemyTemperature] = None
    Materials: Optional[MStrArray] = None

    UsesBiomeVariants: Optional[MBool] = None  # Exclusive with CustomVeterans
    CustomVeterans: Optional[Dict[str, MFloat]] = (
        None  # Exclusive with UsesBiomeVariants
    )

    Projectile: Optional[MStr] = None
    Direct: Optional[Dict[str, Any]] = None
    Spawner: Optional[DeathSpawner] = None
    JellyBreeder: Optional[JellyBreederSettings] = None
    CaveLeech: Optional[CaveLeechSettings] = None


# ==========================================
# PART 6: ROOT DIFFICULTY PROFILE
# ==========================================


@dataclass
class DifficultyProfile(CD2Object):
    Name: str
    Description: Optional[MStr] = None
    MaxPlayers: Optional[MFloat] = None
    BaseHazard: Optional[MStr] = None

    Dwarves: Optional[DwarvesSettings] = None
    Vars: Optional[Union[Dict[str, VarDefinition], List["Variable"]]] = None
    Warnings: Optional[WarningSettings] = None
    Caps: Optional[CapSettings] = None
    Resupply: Optional[ResupplySettings] = None
    NitraMultiplier: Optional[MFloat] = None
    DifficultySetting: Optional[GlobalDifficultySettings] = None
    EscortMule: Optional[EscortMuleSettings] = None
    Salvage: Optional[SalvageSettings] = None
    Pools: Optional[PoolSettings] = None

    Enemies: Optional[Dict[str, EnemyDescriptor]] = None
    EnemiesNoSync: Optional[Dict[str, EnemyDescriptor]] = None

    SpecialEncounters: Optional[List[SpecialEncounter]] = None
    WaveSpawners: Optional[List[WaveSpawner]] = None
    Messages: Optional[List[Message]] = None
    SoundCues: Optional[List[SoundCue]] = None

    def optimize(self) -> DifficultyProfile:
        try:
            optimizer = ExpressionOptimizer()
            optimized_profile = optimizer.optimize(self)
            return optimized_profile
        except ValidationError:
            # If optimization fails (e.g., circular dependencies), fall back to unoptimized
            pass
        return self

    def to_dict(self, optimize: bool = True) -> dict:
        """Convert to dictionary with optional expression optimization.

        Args:
            optimize: If True, runs expression optimizer on all expressions

        Returns:
            Dictionary representation of the profile
        """
        if optimize:
            try:
                optimizer = ExpressionOptimizer()
                optimized_profile = optimizer.optimize(self)
                return self._to_dict_impl(optimized_profile)
            except ValidationError:
                # If optimization fails (e.g., circular dependencies), fall back to unoptimized
                pass

        return self._to_dict_impl(self)

    def _to_dict_impl(self, obj) -> dict:
        """Internal implementation of to_dict that handles Vars properly."""
        result = {}
        for key in obj.__annotations__:
            val = getattr(obj, key, None)
            if val is None:
                continue

            # Convert list of Variables to dict of VarDefinitions
            if key == "Vars" and isinstance(val, list):
                val = {v.Name: v.definition for v in val if isinstance(v, Variable)}

            result[key] = obj._serialize(val)
        return result

    def json(self, indent: Optional[int] = 4, optimize: bool = True) -> str:
        """Returns the JSON string representation with optional optimization."""
        return json.dumps(self.to_dict(optimize=optimize), indent=indent)

    def save(self, path: str, indent: Optional[int] = 4, optimize: bool = True):
        """Saves the JSON representation to a file with optional optimization."""
        with open(path, "w") as f:
            f.write(self.json(optimize=optimize, indent=indent))


# ==========================================
# PART 7: ARITHMETIC MUTATORS
# ==========================================


def Accumulate(initial, value, min_val=None, max_val=None) -> Expression:
    """Returns Float: Adds 'Value' to 'Initial' every second."""
    return Mut("Accumulate", Initial=initial, Value=value, Min=min_val, Max=max_val)


def Add(*args, **kwargs) -> Expression:
    """Returns Float: Sum of all inputs."""
    return _variadic_mutator("Add", *args, **kwargs)


def Sub(a, b) -> Expression:
    """Returns Float: A minus B."""
    return Mut("Subtract", A=a, B=b)


def Mul(*args, **kwargs) -> Expression:
    """Returns Float: Product of all inputs."""
    return _variadic_mutator("Multiply", *args, **kwargs)


def Div(a, b) -> Expression:
    """Returns Float: A divided by B."""
    return Mut("Divide", A=a, B=b)


def Pow(a, b) -> Expression:
    """Returns Float: A raised to the power of B."""
    return Mut("Pow", A=a, B=b)


def Modulo(a, b) -> Expression:
    """Returns Float: Remainder of A divided by B."""
    return Mut("Modulo", A=a, B=b)


def Round(value) -> Expression:
    """Returns Float: Rounds to nearest integer."""
    return Mut("Round", Value=value)


def Ceil(value) -> Expression:
    """Returns Float: Rounds up to nearest integer."""
    return Mut("Ceil", Value=value)


def Floor(value) -> Expression:
    """Returns Float: Rounds down to nearest integer."""
    return Mut("Floor", Value=value)


def Clamp(value, min_val=None, max_val=None) -> Expression:
    """Returns Float: Restricts value between Min and Max."""
    if min_val is None and max_val is None:
        raise ValueError("Clamp requires at least Min or Max.")
    return Mut("Clamp", Value=value, Min=min_val, Max=max_val)


def Max(*args, **kwargs) -> Expression:
    """Returns Float: The largest value among inputs (maps to A, B, C...)."""
    return _variadic_mutator("Max", *args, **kwargs)


def Min(*args, **kwargs) -> Expression:
    """Returns Float: The smallest value among inputs (maps to A, B, C...)."""
    return _variadic_mutator("Min", *args, **kwargs)


def Nonzero(value) -> Expression:
    """Returns Bool: True if value != 0, else False."""
    return Mut("Nonzero", Value=value)


# ==========================================
# PART 8: BOOLEAN & LOGIC MUTATORS
# ==========================================


def And(*args, **kwargs) -> Expression:
    """Returns Boolean: True if all inputs are True."""
    return _variadic_mutator("And", *args, **kwargs)


def Or(*args, **kwargs) -> Expression:
    """Returns Boolean: True if any input is True."""
    return _variadic_mutator("Or", *args, **kwargs)


def Not(value) -> Expression:
    """Returns Boolean: True if input is False, and vice versa."""
    return Mut("Not", Value=value)


def If(condition, then_val, else_val) -> Expression:
    """Returns Any: 'Then' value if Condition is True, otherwise 'Else' value."""
    return Mut("If", Condition=condition, Then=then_val, Else=else_val)


def IfFloat(value, operator: str, benchmark, then_val, else_val) -> Expression:
    """
    Returns Any: Compares Value vs Benchmark.
    Valid operators: ==, >=, >, <=, <
    """
    return Mut(
        "IfFloat", Value=value, **{operator: benchmark}, Then=then_val, Else=else_val
    )


def Select(select_input, default, **options) -> Expression:
    """
    Returns Any: Matches 'Select' input (string) against keys in options.
    Example: Select(Var("Mode"), Default=1, Easy=0.5, Hard=1.5)
    """
    return Mut("Select", Select=select_input, Default=default, **options)


# ==========================================
# PART 9: GAME STATE MUTATORS (NO ARGUMENTS)
# ==========================================


DrillerCount = Mut("DrillerCount")
"""Float: Number of Drillers in the game."""

EngineerCount = Mut("EngineerCount")
"""Float: Number of Engineers in the game."""

GunnerCount = Mut("GunnerCount")
"""Float: Number of Gunners in the game."""

ScoutCount = Mut("ScoutCount")
"""Float: Number of Scouts in the game."""

DwarfCount = Mut("DwarfCount")
"""Float: Total number of players."""

DwarvesAmmo = Mut("DwarvesAmmo")
"""Float: Average ammo percentage of all dwarves (0.0 to 1.0)."""

DwarvesDown = Mut("DwarvesDown")
"""Float: Amount of dwarves being down right now"""

DwarvesDowns = Mut("DwarvesDowns")
"""Float: Total number of times dwarves have gone down this mission."""

DwarvesDownTime = Mut("DwarvesDownTime")
"""Float: Time (seconds) dwarves have spent down. Chooses highest, if multiple down at once."""

DwarvesHealth = Mut("DwarvesHealth")
"""Float: Average health percentage of all dwarves. (0.0 to 1.0)"""

DwarvesRevives = Mut("DwarvesRevives")
"""Float: Total number of revives performed."""

DwarvesShield = Mut("DwarvesShield")
"""Float: Average shield percentage of all dwarves. (0.0 to 1.0)"""


IWsLeft = Mut("IWsLeft")
"""Float: Number of Iron Wills remaining."""

DuringDefend = Mut("DuringDefend")
"""Boolean: True during a defense event (Uplink, Refuel, Black Box)."""

DuringDread = Mut("DuringDread")
"""Boolean: True while a Dreadnought is active."""

DuringDrillevator = Mut("DuringDrillevator")
"""Boolean: True during the Drillevator sequence (Deep Scan)."""

DuringExtraction = Mut("DuringExtraction")
"""Boolean: True after the Drop Pod button has been pushed."""

DuringGenericSwarm = Mut("DuringGenericSwarm")
"""Boolean: True during a mission control announced swarm."""

DuringPECountdown = Mut("DuringPECountdown")
"""Boolean: True during Point Extraction minehead launch countdown."""

DuringEncounters = Mut("DuringEncounters")
"""Boolean: True during mission loading."""

IfOnSpaceRig = Mut("IfOnSpaceRig")
"""Boolean: True if players are in the Space Rig lobby."""

ElapsedExtraction = Mut("ElapsedExtraction")
"""Float: Seconds elapsed since extraction started."""

ResuppliesCalled = Mut("ResuppliesCalled")
"""Float: Total number of supply pods ordered."""

ResupplyUsesConsumed = Mut("ResupplyUsesConsumed")
"""Float: Total supply racks taken by players."""

ResupplyUsesLeft = Mut("ResupplyUsesLeft")
"""Float: Total supply racks currently available on map."""

SecondaryFinished = Mut("SecondaryFinished")
"""Boolean: True if the secondary objective is complete."""

DefenseProgress = Mut("DefenseProgress")
"""Float: Progress of current defense event (0.0 to 1.0). Starts at 0.3"""


# ==========================================
# PART 10: GAME STATE MUTATORS (WITH ARGUMENTS)
# ==========================================


def ByPlayerCount(values_or_first, *args) -> Expression:
    """
    Returns Any: Selects value by index [PlayerCount - 1], or last if out of range.
    Supports usage:
      ByPlayerCount([10, 20, 30])
      ByPlayerCount(10, 20, 30)
    """
    if isinstance(values_or_first, list) and not args:
        values = values_or_first
    else:
        values = [values_or_first, *args]

    return Mut("ByPlayerCount", Values=values)


def ByResuppliesCalled(values_or_first, *args) -> Expression:
    """
    Returns Any: Selects value by index [ResuppliesCalled] (clamped).
    Usage: ByResuppliesCalled(1.0, 1.2, 1.5)
    """
    if isinstance(values_or_first, list) and not args:
        values = values_or_first
    else:
        values = [values_or_first, *args]
    return Mut("ByResuppliesCalled", Values=values)


def ByTime(initial, rate, start_delay=None) -> Expression:
    """Returns Float: initial + rate * Max(0,Time-StartDelay). Time is time since mission start"""
    return Mut(
        "ByTime", InitialValue=initial, RateOfChange=rate, StartDelay=start_delay
    )


def TimeDelta(initial, rate, start_delay=None) -> Expression:
    """Alias for ByTime logic with different name in documentation."""
    return Mut(
        "TimeDelta", InitialValue=initial, RateOfChange=rate, StartDelay=start_delay
    )


def Delta(value) -> Expression:
    """Returns Float: The difference in 'Value' since the last tick."""
    return Mut("Delta", Value=value)


def DepositedResource(resource: str) -> Expression:
    """Returns Float: Amount of specific resource currently in Molly/Refinery."""
    return Mut("DepositedResource", Resource=resource)


def HeldResource(resource: str) -> Expression:
    """Returns Float: Amount of specific resource currently in players' pockets."""
    return Mut("HeldResource", Resource=resource)


def TotalResource(resource: str) -> Expression:
    """Returns Float: Sum of Deposited and Held resource."""
    return Mut("TotalResource", Resource=resource)


def DroppodDistance(default, include_downed: bool = False) -> Expression:
    """Returns Float: Average distance of the players to the Drop Pod. If IncludeDowned is true, includes downed players as well. If Droppod is not present, returns default."""
    return Mut("DroppodDistance", Default=default, IncludeDowned=include_downed)


def MuleDroppodDistance(default) -> Expression:
    """Returns Float: Distance of Molly to the Drop Pod."""
    return Mut("MuleDroppodDistance", Default=default)


def DuringMission(starting_at=None, stopping_after=None) -> Expression:
    """Returns Boolean: True if mission time is between Start and Stop."""
    return Mut("DuringMission", StartingAt=starting_at, StoppingAfter=stopping_after)


def DuringEggAmbush(starting_at=None, stopping_after=None) -> Expression:
    """Returns Boolean: True if within time window after pulling an Egg."""
    return Mut("DuringEggAmbush", StartingAt=starting_at, StoppingAfter=stopping_after)


def DescriptorExists(ed: str) -> Expression:
    """Returns Boolean: True if the Enemy Descriptor (ED) exists in the game."""
    return Mut("DescriptorExists", ED=ed)


# ==========================================
# PART 11: ENEMY TRACKING MUTATORS
# ==========================================


def _ed_helper(ed_or_eds) -> Dict[str, Any]:
    """Internal helper: Maps single string to ED, list to EDs.

    Args:
        ed_or_eds: Either a single enemy descriptor string or a list of them

    Returns:
        Dictionary with either 'ED' or 'EDs' key
    """
    if isinstance(ed_or_eds, list):
        return {"EDs": ed_or_eds}
    return {"ED": ed_or_eds}


def EnemiesKilled(ed_or_eds=None) -> Expression:
    """Returns Float: Total kills of specified enemy type(s)."""
    args = _ed_helper(ed_or_eds) if ed_or_eds else {}
    return Mut("EnemiesKilled", **args)


def EnemyCount(ed_or_eds=None) -> Expression:
    """Returns Float: Current active count of specified enemy type(s)."""
    args = _ed_helper(ed_or_eds) if ed_or_eds else {}
    return Mut("EnemyCount", **args)


def EnemiesRecentlySpawned(seconds: float, ed_or_eds=None) -> Expression:
    """Returns Float: Count of enemies spawned in the last N seconds."""
    ed_args = _ed_helper(ed_or_eds) if ed_or_eds else {}
    return Mut("EnemiesRecentlySpawned", Seconds=seconds, **ed_args)


def EnemyCooldown(
    ed_or_eds, cooldown_time, val_during_cooldown, default_val
) -> Expression:
    """Returns Any: Switches value based on if enemy type spawned recently."""
    ed_args = _ed_helper(ed_or_eds)
    return Mut(
        "EnemyCooldown",
        CooldownTime=cooldown_time,
        ValueDuringCooldown=val_during_cooldown,
        DefaultValue=default_val,
        **ed_args,
    )


def EnemyHealth(ed_or_eds, default, type_str="Average") -> Expression:
    """Returns Float: Health (Average/Min/Max) of active enemies."""
    ed_args = _ed_helper(ed_or_eds)
    return Mut("EnemyHealth", Default=default, Type=type_str, **ed_args)


def EnemyDistance(ed_or_eds, default, type_str="Min") -> Expression:
    """Returns Float: Distance (Average/Min/Max) of enemies to players."""
    ed_args = _ed_helper(ed_or_eds)
    return Mut("EnemyDistance", Default=default, Type=type_str, **ed_args)


# ==========================================
# PART 12: STRING & MESSAGE MUTATORS
# ==========================================


def Float2String(value) -> Expression:
    """Returns String: Converts a numeric value to string."""
    return Mut("Float2String", Value=value)


def Int2String(value) -> Expression:
    """Returns String: Converts number to integer string (no decimals)."""
    return Mut("Int2String", Value=value)


def Join(values: list, sep="") -> Expression:
    """Returns String: Joins a list of strings with a separator."""
    return Mut("Join", Values=values, Sep=sep)


def Countdown(start, enable=True, stop=None) -> Expression:
    """Returns String: Formats a countdown timer (MM:SS)."""
    return Mut("Countdown", Start=start, Enable=enable, Stop=stop)


# ==========================================
# PART 13: LOCKING & WAVEFORM MUTATORS
# ==========================================


def LockFloat(value, lock_condition=None, update_condition=None) -> Expression:
    """Returns Float: Holds a value when Lock is True or until Update is True."""
    return Mut("LockFloat", Value=value, Lock=lock_condition, Update=update_condition)


def LockBoolean(value, lock_condition=None, update_condition=None) -> Expression:
    """Returns Boolean: Holds a value when Lock is True or until Update is True."""
    return Mut("LockBoolean", Value=value, Lock=lock_condition, Update=update_condition)


def LockString(value, lock_condition=None, update_condition=None) -> Expression:
    """Returns String: Holds a value when Lock is True or until Update is True."""
    return Mut("LockString", Value=value, Lock=lock_condition, Update=update_condition)


def SquareWave(period, high, low) -> Expression:
    """Returns Float: Alternates between High and Low every Period/2 seconds."""
    return Mut("SquareWave", Period=period, High=high, Low=low)


# ==========================================
# PART 14: RANDOMIZATION MUTATORS
# ==========================================


def Random(min_val, max_val) -> Expression:
    """Returns Float: Random number between Min and Max (changes every tick)."""
    return Mut("Random", Min=min_val, Max=max_val)


def RandomPerMission(min_val, max_val) -> Expression:
    """Returns Float: Random number generated once per mission."""
    return Mut("RandomPerMission", Min=min_val, Max=max_val)


def RandomChoice(choices: list, weights: Optional[list] = None) -> Expression:
    """Returns Any: Randomly selects an item from Choices, optionally weighted."""
    return Mut("RandomChoice", Choices=choices, Weights=weights)


def RandomChoicePerMission(choices: list, weights: Optional[list] = None) -> Expression:
    """Returns Any: Randomly selects an item from Choices once per mission, optionally weighted."""
    return Mut("RandomChoicePerMission", Choices=choices, Weights=weights)


# ==========================================
# PART 15: MISSION & BIOME CONTEXT MUTATORS
# ==========================================


def ByBiome(default, **biomes) -> Expression:
    """Returns Any: Switches value based on biome."""
    for key in biomes:
        if key not in _VALID_BIOME_ALIASES:
            raise ValueError(f"Invalid Biome Alias: '{key}'.")
    return Mut("ByBiome", Default=default, **biomes)


def ByDNA(default, variants_dict=None, **variants) -> Expression:
    """Returns Any: Switches value based on DNA mission variant.

    DNA format: Type[,Length][,Complexity]
    - Type: Mission type (required)
    - Length: Level 1-3 or 'x' (optional)
    - Complexity: Level 1-3 or 'x' (optional)

    Args:
        default: Default value if no variant matches
        variants_dict: Dict for complex keys (e.g., {"Mining,1,1": 200})
        **variants: Simple identifier keys (e.g., PE=1.5, Mining=2.0)
    """
    # Merge variants_dict and kwargs
    if variants_dict is not None:
        if variants:
            raise ValueError("Cannot specify both variants_dict and keyword arguments")
        variants = variants_dict

    for key in variants:
        parts = [part.strip() for part in key.split(",")]

        # Must have 1-3 parts
        if len(parts) < 1 or len(parts) > 3:
            raise ValueError(
                f"Invalid DNA string '{key}': must have format Type[,Length][,Complexity]"
            )

        # Validate Type (required, first part)
        mission_type = parts[0]
        if mission_type not in _VALID_DNA_MISSIONS:
            raise ValueError(
                f"Invalid DNA Mission type '{mission_type}' in '{key}'. "
                f"Valid types: {sorted(_VALID_DNA_MISSIONS)}"
            )

        # Validate Length (optional, second part)
        if len(parts) >= 2:
            length = parts[1]
            if length not in _VALID_LENGTH_LEVELS:
                raise ValueError(
                    f"Invalid DNA Length '{length}' in '{key}'. "
                    f"Valid lengths: {sorted(_VALID_LENGTH_LEVELS)}"
                )

        # Validate Complexity (optional, third part)
        if len(parts) >= 3:
            complexity = parts[2]
            if complexity not in _VALID_COMPLEXITY_LEVELS:
                raise ValueError(
                    f"Invalid DNA Complexity '{complexity}' in '{key}'. "
                    f"Valid complexities: {sorted(_VALID_COMPLEXITY_LEVELS)}"
                )

    return Mut("ByDNA", Default=default, **variants)


def ByDDStage(default, **stages) -> Expression:
    """Returns Any: Switches value based on Deep Dive stage."""
    for key in stages:
        if key not in ["Stage1", "Stage2", "Stage3"]:
            raise ValueError(f"Invalid DD Stage: '{key}'.")
    return Mut("ByDDStage", Default=default, **stages)


def ByMissionType(default, **missions) -> Expression:
    """Returns Any: Switches value based on mission type."""
    for key in missions:
        if key not in _VALID_DNA_MISSIONS:
            raise ValueError(f"Invalid Mission Type: '{key}'.")
    return Mut("ByMissionType", Default=default, **missions)


def BySecondary(default, **variants) -> Expression:
    """Returns Any: Switches value based on secondary objective variant."""
    for key in variants:
        if key not in _VALID_SECONDARY_OBJS:
            raise ValueError(f"Invalid Secondary Obj: '{key}'.")
    return Mut("BySecondary", Default=default, **variants)


# --- Mission Phase Mutators ---


def ByEscortPhase(default, **phases) -> Expression:
    """Returns Any: Switches value based on Escort mission phase, Default if not Escort."""
    for key in phases:
        if key not in _VALID_ESCORT_PHASES:
            raise ValueError(f"Invalid Escort Phase: {key}")
    return Mut("ByEscortPhase", Default=default, **phases)


def ByRefineryPhase(default, **phases) -> Expression:
    """Returns Any: Switches value based on Refinery mission phase, Default if not Refinery."""
    for key in phases:
        if key not in _VALID_REFINERY_PHASES:
            raise ValueError(f"Invalid Refinery Phase: {key}")
    return Mut("ByRefineryPhase", Default=default, **phases)


def BySaboPhase(default, **phases) -> Expression:
    """Returns Any: Switches value based on Sabotage mission phase, Default if not Sabotage."""
    for key in phases:
        if key not in _VALID_SABO_PHASES:
            raise ValueError(f"Invalid Sabo Phase: {key}")
    return Mut("BySaboPhase", Default=default, **phases)


def BySalvagePhase(default, **phases) -> Expression:
    """Returns Any: Switches value based on Salvage mission phase, Default if not Salvage."""
    for key in phases:
        if key not in _VALID_SALVAGE_PHASES:
            raise ValueError(f"Invalid Salvage Phase: {key}")
    return Mut("BySalvagePhase", Default=default, **phases)


# ==========================================
# PART 16: TRIGGER MUTATORS
# ==========================================


def TriggerOnce(input_val, reset=None) -> Expression:
    """Returns Boolean: True the first time Input is True, then false until Reset."""
    return Mut("TriggerOnce", In=input_val, Reset=reset)


def TriggerNTimes(input_val, n, reset=None) -> Expression:
    """Returns Boolean: True the first N times Input is True, then false until Reset."""
    return Mut("TriggerNTimes", In=input_val, N=n, Reset=reset)


def TriggerSometimes(input_val, p) -> Expression:
    """Returns Boolean: When Input is True, has P chance to return True."""
    return Mut("TriggerSometimes", In=input_val, P=p)


def TriggerCooldown(input_val, n) -> Expression:
    """Returns Boolean: True when input is true. After it goes to false, can't return true for N seconds."""
    return Mut("TriggerCooldown", In=input_val, N=n)


def TriggerFixedDuration(input_val, n, reset=None) -> Expression:
    """Returns Boolean: True for N seconds after Input becomes True, regardless of Input state. If Reset is True, resets state during duration."""
    return Mut("TriggerFixedDuration", In=input_val, N=n, Reset=reset)


def TriggerSustain(input_val, n) -> Expression:
    """Returns Boolean: Extends True state for extra N seconds."""
    return Mut("TriggerSustain", In=input_val, N=n)


def TriggerDelay(input_val, n) -> Expression:
    """Returns Boolean: Delays Transition between states by N seconds."""
    return Mut("TriggerDelay", In=input_val, N=n)


def TriggerOnChange(input_val, rise_only=None, fall_only=None) -> Expression:
    """Returns Boolean: True for one tick when Input(Float) changes. If RiseOnly is true, only on False->True. If FallOnly is true, only on True->False."""
    args = {"In": input_val}
    if rise_only is not None:
        args["RiseOnly"] = rise_only
    if fall_only is not None:
        args["FallOnly"] = fall_only
    return Mut("TriggerOnChange", **args)


# ==========================================
# PART 17: COMPARISON & EXTENSION HELPERS
# ==========================================

E = Variable("E", 2.7182818284590, "Float").Alias("E")
Pi = Variable("Pi", 3.141592653589, "Float").Alias("Pi")


def Exp(val) -> Expression:
    """Returns Float: e^val. E must be added to Vars."""
    return Pow(E, val)


def Sign(val) -> Expression:
    """Returns Float: sign(val)."""
    return MatchRange(val, 0, (0, -1), (1, 1))


def Sinh(val) -> Expression:
    """Returns Float: sinh(val)."""
    return (Exp(val) - Exp(-val)) / 2


def Cosh(val) -> Expression:
    """Returns Float: cosh(val)."""
    return (Exp(val) + Exp(-val)) / 2


def Tanh(val) -> Expression:
    """Returns Float: tanh(val)."""
    return Sinh(val) / Cosh(val)


def Sech(val) -> Expression:
    """Returns Float: sech(val)."""
    return 1 / Cosh(val)


def Csch(val) -> Expression:
    """Returns Float: csch(val)."""
    return 1 / Sinh(val)


def Coth(val) -> Expression:
    """Returns Float: coth(val)."""
    return Cosh(val) / Sinh(val)


def Eq(a, b) -> Expression:
    """Returns Boolean: True if A == B."""
    return IfFloat(a, "==", b, True, False)


def Neq(a, b) -> Expression:
    """Returns Boolean: True if A != B."""
    return IfFloat(a, "!=", b, True, False)


def Gt(a, b) -> Expression:
    """Returns Boolean: True if A > B."""
    return IfFloat(a, ">", b, True, False)


def Lt(a, b) -> Expression:
    """Returns Boolean: True if A < B."""
    return IfFloat(a, "<", b, True, False)


def Gte(a, b) -> Expression:
    """Returns Boolean: True if A >= B."""
    return IfFloat(a, ">=", b, True, False)


def Lte(a, b) -> Expression:
    """Returns Boolean: True if A <= B."""
    return IfFloat(a, "<=", b, True, False)


def IsPositive(val) -> Expression:
    """Returns Boolean: True if val > 0."""
    return Gt(val, 0)


def Negate(val) -> Expression:
    """Returns Float: -val."""
    return Sub(0, val)


def Abs(val) -> Expression:
    """Returns Float: The absolute value |val|."""
    return IfFloat(val, "<", 0, Negate(val), val)


def Inverse(val) -> Expression:
    """Returns Float: 1 / val."""
    return Div(1, val)


def Lerp(start, end, t) -> Expression:
    """
    Linear Interpolation. Returns Float.
    Result = Start + (End - Start) * t
    t should be between 0.0 and 1.0.
    """
    return Add(start, Mul(Sub(end, start), t))


def InverseLerp(val, min_val, max_val) -> Expression:
    """Returns Float: Normalized position of val between min_val and max_val (0.0 to 1.0)."""
    return Div(Sub(val, min_val), Sub(max_val, min_val))


def Remap(val, in_min, in_max, out_min, out_max, clamp=False) -> Expression:
    """
    Returns Float: Maps val from [in_min, in_max] to [out_min, out_max].
    Equivalent to Arduino map() function.
    """
    t = InverseLerp(val, in_min, in_max)
    if clamp:
        t = Clamp(t, 0.0, 1.0)
    return Lerp(out_min, out_max, t)


def Distance2D(x1, y1, x2, y2) -> Expression:
    """
    Returns Float: Euclidean distance between two points (Pythagorean theorem).
    Sqrt is equivalent to Pow(x, 0.5).
    """
    delta_x = Sub(x2, x1)
    delta_y = Sub(y2, y1)
    return Pow(Add(Pow(delta_x, 2), Pow(delta_y, 2)), 0.5)


def Nand(*args) -> Expression:
    """Returns Boolean: Not And (False only if all are True)."""
    return Not(And(*args))


def Nor(*args) -> Expression:
    """Returns Boolean: Not Or (True only if all are False)."""
    return Not(Or(*args))


def Xor(a, b) -> Expression:
    """Returns Boolean: Exclusive Or (True if exactly one input is True)."""
    return Or(And(a, Not(b)), And(Not(a), b))


def ScaleLinear(base_value, add_per_player) -> Expression:
    """
    Returns Float: base + (PlayerCount - 1) * add
    1 Player: base
    4 Players: base + 3*add
    """
    extra_players = Max(Sub(DwarfCount, 1), 0)
    return Add(base_value, Mul(extra_players, add_per_player))


def ScaleMultiplicative(base_value, mult_per_player) -> Expression:
    """
    Returns Float: base * (mult ^ (PlayerCount - 1))
    Useful for HP scaling.
    """
    extra_players = Max(Sub(DwarfCount, 1), 0)
    return Mul(base_value, Pow(mult_per_player, extra_players))


def ScaleByHaz(base_hz1, add_per_hz) -> Expression:
    """
    Returns Float: Scales value based on Hazard Level (1-5).
    Assumes 'Haz' variable is defined elsewhere.
    """
    return Add(base_hz1, Mul(Sub(Var("Haz"), 1), add_per_hz))


def TimeRamp(start_val, end_val, duration_sec, delay_sec=0) -> Expression:
    """
    Returns Float: Smoothly interpolates from start to end over duration.
    Stops at end_val after duration.
    """
    local_time = Max(Sub(ByTime(0, 1), delay_sec), 0)
    progress = Clamp(Div(local_time, duration_sec), 0.0, 1.0)
    return Lerp(start_val, end_val, progress)


def SawtoothWave(period, min_val, max_val) -> Expression:
    """
    Returns Float: Linearly rises from min to max, then resets instantly.
    """
    t = Modulo(ByTime(0, 1), period)
    return Remap(t, 0, period, min_val, max_val)


def TriangleWave(period, min_val, max_val) -> Expression:
    """
    Returns Float: Linearly rises and falls between min and max.
    """
    t = Div(Modulo(ByTime(0, 1), period), period)
    return IfFloat(
        t,
        "<",
        0.5,
        Remap(t, 0.0, 0.5, min_val, max_val),
        Remap(t, 0.5, 1.0, max_val, min_val),
    )


def OnKill(ed_or_eds) -> Expression:
    """Returns Boolean: True for one tick when enemy type is killed."""
    return TriggerOnChange(EnemiesKilled(ed_or_eds))


def OnSpawn(ed_or_eds) -> Expression:
    """Returns Boolean: True for one tick when enemy type spawns."""
    return TriggerOnChange(EnemyCount(ed_or_eds), rise_only=True)


# ==========================================
# PART 18: PREDEFINED GAME STATE EXPRESSIONS
# ==========================================

IsSolo = Eq(DwarfCount, 1)
"""Expression: True if only 1 player."""

IsFullTeam = Gte(DwarfCount, 4)
"""Expression: True if 4+ players."""

AnyDwarfDown = Gt(DwarvesDown, 0)
"""Expression: True if at least one dwarf is down."""

AllDwarvesAlive = Eq(DwarvesDown, 0)
"""Expression: True if 0 dwarves are down."""

TeamWipeImminent = Eq(DwarvesDown, DwarfCount)
"""Expression: True if everyone is down (Iron Will context)."""

PostDrillevator = And(
    LockBoolean(
        TriggerOnChange(DuringDrillevator, rise_only=True),
        update_condition=DuringDrillevator,
    ),
    Not(DuringDrillevator),
)
"""Expression: True after Drillevator sequence is complete."""

IsExtraction = Or(DuringExtraction, DuringPECountdown)
"""Expression: True if currently in extraction phase (Drop Pod button or PE countdown)."""

IsLastStanding = And(Gt(DwarfCount, 1), Eq(DwarvesDown, Sub(DwarfCount, 1)))
"""Expression: True if in multiplayer and only 1 dwarf is standing."""


# ==========================================
# PART 19: ADVANCED HELPER FUNCTIONS
# ==========================================


def InRange(val, min_val, max_val) -> Expression:
    """Returns Boolean: True if min <= val <= max."""
    return And(Gte(val, min_val), Lte(val, max_val))


def Switch(default, *cases) -> Expression:
    """Functional switch/case: returns first matching value.
    Usage: Switch(DefaultVal, (Cond1, Val1), (Cond2, Val2))
    """
    expr = default
    for cond, val in reversed(cases):
        expr = If(cond, val, expr)
    return Expression(expr)


def Match(input_val, default, *cases) -> Expression:
    """
    Switch-case for numeric values (equality check).
    Usage: Match(DwarfCount, DefaultVal, (1, ValFor1), (2, ValFor2))
    """
    expr = default
    for target, val in reversed(cases):
        expr = IfFloat(input_val, "==", target, val, expr)
    return Expression(expr)


def MatchRange(input_val, default, *cases) -> Expression:
    """
    Switch-case for numeric values (range check).
    Usage: MatchRange(DwarfCount, DefaultVal, (1, ValFor1), (2, ValFor2))
    """
    expr = default
    for target, val in reversed(cases):
        expr = IfFloat(input_val, "<", target, val, expr)
    return Expression(expr)


def ScaleByPlayerCount(base) -> Expression:
    """
    Returns Float: base * playercount
    """
    return Mul(base, DwarfCount)


def Sigmoid(val, steepness=1, midpoint=0) -> Expression:
    """Returns Float: Sigmoid function for smooth S-curve transitions.
    Useful for difficulty scaling that ramps up smoothly."""
    return Div(1, Add(1, Pow(2.71828, Negate(Mul(steepness, Sub(val, midpoint))))))


def SmoothStep(val, edge0=0, edge1=1) -> Expression:
    """Returns Float: Smooth Hermite interpolation between 0 and 1.
    Better than linear for smooth difficulty curves."""
    t = Clamp(Div(Sub(val, edge0), Sub(edge1, edge0)), 0, 1)
    return Mul(Mul(t, t), Sub(3, Mul(2, t)))


# Class composition helpers
HasDriller = Gt(DrillerCount, 0)
"""Expression: True if team has at least one Driller."""

HasEngineer = Gt(EngineerCount, 0)
"""Expression: True if team has at least one Engineer."""

HasGunner = Gt(GunnerCount, 0)
"""Expression: True if team has at least one Gunner."""

HasScout = Gt(ScoutCount, 0)
"""Expression: True if team has at least one Scout."""

IsBalancedTeam = And(HasDriller, HasEngineer, HasGunner, HasScout)
"""Expression: True if team has all 4 classes."""


def TeamHasClasses(*classes) -> Expression:
    """Returns Boolean: True if team has all specified classes.
    Usage: TeamHasClasses('Driller', 'Scout')"""
    checks = []
    class_map = {
        "Driller": DrillerCount,
        "Engineer": EngineerCount,
        "Gunner": GunnerCount,
        "Scout": ScoutCount,
    }
    for cls in classes:
        checks.append(Gt(class_map[cls], 0))
    return And(*checks)


# Mission time helpers
MissionTime = ByTime(0, 1)
"""Expression: Current mission time in seconds."""


def MissionPhase(early_max=300, mid_max=900) -> Expression:
    """Returns String: 'Early', 'Mid', or 'Late' based on mission time."""
    time = MissionTime
    return IfFloat(
        time, "<", early_max, "Early", IfFloat(time, "<", mid_max, "Mid", "Late")
    )


def TimeWindow(start, end) -> Expression:
    """Returns Boolean: True if mission time is between start and end."""
    return And(Gte(MissionTime, start), Lte(MissionTime, end))


IsEarlyGame = Lt(MissionTime, 300)
"""Expression: True if mission is less than 5 minutes in."""

IsMidGame = InRange(MissionTime, 300, 900)
"""Expression: True if mission is between 5-15 minutes."""

IsLateGame = Gt(MissionTime, 900)
"""Expression: True if mission is more than 15 minutes in."""


# Resource management helpers
def ResourceProgress(resource, target) -> Expression:
    """Returns Float: Progress toward resource goal (0.0 to 1.0+)."""
    return Div(TotalResource(resource), target)


def NitraRatio() -> Expression:
    """Returns Float: Ratio of nitra to resupplies called."""
    return Div(TotalResource("Nitra"), Max(Mul(ResuppliesCalled, 80), 1))


IsNitraStarved = Lt(TotalResource("Nitra"), 40)
"""Expression: True if team has less than 40 nitra."""

IsNitraRich = Gt(TotalResource("Nitra"), 200)
"""Expression: True if team has more than 200 nitra."""


# Danger and adaptive difficulty
def DangerLevel() -> Expression:
    """Returns Float: Composite danger score based on multiple factors."""
    return Add(
        Mul(DwarvesDown, 10),  # Each down dwarf adds danger
        Mul(Sub(1, DwarvesHealth), 5),  # Low health adds danger
        Mul(Sub(1, DwarvesAmmo), 3),  # Low ammo adds danger
        If(DuringExtraction, 15, 0),  # Extraction is dangerous
        If(DuringDefend, 10, 0),  # Defense is dangerous
    )


IsHighDanger = Gt(DangerLevel(), 20)
"""Expression: True if danger level exceeds threshold."""


def AdaptiveDifficulty(base, danger_mult=0.1) -> Expression:
    """Returns Float: Scales difficulty based on current danger."""
    return Mul(base, Add(1, Mul(DangerLevel(), danger_mult)))


# Math utilities
def Average(*values) -> Expression:
    """Returns Float: Average of all input values."""
    return Div(Add(*values), len(values))


def WeightedAverage(*weighted_pairs) -> Expression:
    """Returns Float: Weighted average.
    Usage: WeightedAverage((value1, weight1), (value2, weight2))"""
    numerator = Add(*[Mul(val, weight) for val, weight in weighted_pairs])
    denominator = Add(*[weight for _, weight in weighted_pairs])
    return Div(numerator, denominator)


def Percentage(part, whole) -> Expression:
    """Returns Float: Percentage (0-100)."""
    return Mul(Div(part, whole), 100)


def Root(val, n) -> Expression:
    """Returns Float: Nth root of val."""
    return Pow(val, Inverse(n))


def Sqrt(val) -> Expression:
    """Returns Float: Square root of val."""
    return Root(val, 2)


def Cbrt(val) -> Expression:
    """Returns Float: Cube root of val."""
    return Root(val, 3)


IsBlackBoxDefense = DuringDefend & ByDNA(True, Salvage=False)
"""Expression: True if team is defending a black box. BlackBox secondary can't spawn on Salvage"""

MorkiteCompletion = Percentage(
    DepositedResource("Morkite"),
    ByDNA(
        DepositedResource("Morkite"),  # Default value
        {
            "Mining,1,1": 200,
            "Mining,1,2": 225,
            "Mining,2,2": 250,
            "Mining,3,2": 325,
            "Mining,3,3": 400,
        },
    ),
)
"""Expression: Percentage of Morkite extracted."""


def IfRefinery(thenval, elseval) -> Expression:
    """Expression: If on refinery, returns first argument, else returns second"""
    return If(ByDNA(False, Refinery=True), thenval, elseval)


def IfSalvage(thenval, elseval) -> Expression:
    """Expression: If on salvage, returns first argument, else returns second"""
    return If(ByDNA(False, Salvage=True), thenval, elseval)


def PerMinuteToPerSecond(val) -> Expression:
    """Expression: Returns an accumulator that increases by 'val' per minute"""
    return ByTime(0, val / 60)


def ByTimeWhen(condition, initial, rate) -> Expression:
    """
    Returns Float: Starts counting from 'initial' at rate 'rate' when condition becomes true.
    Uses a locked ByTime with dynamic StartDelay that captures mission time when condition triggers.

    Args:
        condition: Boolean expression that triggers the timer
        initial: Starting value
        rate: Rate of increase per second

    Returns:
        Expression that counts up from initial after condition is first met

    Example:
        ByTimeWhen(DuringMission(), 0, 1)  # Counts 0, 1, 2, 3... after mission starts
        ByTimeWhen(DuringDefend, 10, 2)    # Counts 10, 12, 14... after defense starts
    """
    stop_delay = LockFloat(
        ByTime(0, 1), lock_condition=TriggerFixedDuration(condition, 999999999)
    )
    return ByTime(initial, rate, start_delay=stop_delay)


# ==========================================
# PART 20: VALIDATION & BUILDER UTILITIES
# ==========================================


class ValidationError(Exception):
    """Custom exception for CD2 schema validation errors."""

    pass


def validate_mutator(mutator_dict) -> bool:
    """Validates mutator structure and provides helpful error messages.

    Args:
        mutator_dict: Dictionary to validate as a mutator

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(mutator_dict, dict):
        raise ValidationError(f"Mutator must be dict, got {type(mutator_dict)}")

    if "Mutate" not in mutator_dict and "Var" not in mutator_dict:
        raise ValidationError(f"Mutator missing 'Mutate' or 'Var' key: {mutator_dict}")

    return True


class DifficultyBuilder:
    """Fluent builder for DifficultyProfile with validation."""

    def __init__(self, name: str):
        self.config: dict[str, Any] = {"Name": name}
        self._vars = {}
        self._enemies = {}

    def with_description(self, desc: str):
        self.config["Description"] = desc
        return self

    def with_max_players(self, count: int):
        if count < 1 or count > 16:
            raise ValidationError("MaxPlayers must be 1-16")
        self.config["MaxPlayers"] = count
        return self

    def add_variable(self, name: str, value, type_str=None, watch=False):
        """Add a variable with automatic type inference."""
        var = Variable(name, value, type_str, watch)
        self._vars[name] = var.definition
        return self

    def add_enemy(self, ed_name: str, **kwargs):
        """Add enemy descriptor with validation."""
        self._enemies[ed_name] = EnemyDescriptor(**kwargs)
        return self

    def build(self) -> DifficultyProfile:
        """Build and validate the final profile."""
        if self._vars:
            self.config["Vars"] = self._vars
        if self._enemies:
            self.config["Enemies"] = self._enemies
        return DifficultyProfile(**self.config)


# Example: profile = (DifficultyBuilder("MyDifficulty")
#     .with_description("Test difficulty")
#     .with_max_players(4)
#     .add_variable("SpeedMult", 1.5)
#     .build())


class Presets:
    """Common difficulty preset templates."""

    @staticmethod
    def scaling_by_time(start_val, end_val, duration=1800):
        """Linear scaling over mission time."""
        return TimeRamp(start_val, end_val, duration)

    @staticmethod
    def player_scaling(solo, duo, trio, quad):
        """Standard player count scaling."""
        return ByPlayerCount([solo, duo, trio, quad])

    @staticmethod
    def extraction_multiplier(normal=1, extraction=2):
        """Standard extraction difficulty spike."""
        return If(IsExtraction, extraction, normal)

    @staticmethod
    def adaptive_spawn_rate(base_interval=180) -> Expression:
        """Spawn rate that adapts to team performance."""
        return Mul(
            base_interval,
            If(
                Gt(DwarvesHealth, 0.7),
                1,  # Full health = normal
                If(Gt(DwarvesHealth, 0.4), 1.2, 1.5),  # Damaged = slower
            ),
        )  # Critical = much slower


def compare_profiles(profile1: DifficultyProfile, profile2: DifficultyProfile):
    """Generate a diff showing differences between two profiles."""
    diff = {}

    dict1 = profile1.to_dict()
    dict2 = profile2.to_dict()

    def recursive_diff(d1, d2, path=""):
        for key in set(list(d1.keys()) + list(d2.keys())):
            current_path = f"{path}.{key}" if path else key

            if key not in d1:
                diff[current_path] = {"added": d2[key]}
            elif key not in d2:
                diff[current_path] = {"removed": d1[key]}
            elif d1[key] != d2[key]:
                if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    recursive_diff(d1[key], d2[key], current_path)
                else:
                    diff[current_path] = {"old": d1[key], "new": d2[key]}

    recursive_diff(dict1, dict2)
    return diff


def simplify_expression(expr: Expression) -> Expression:
    """Attempt to simplify constant expressions."""
    content = expr.content if isinstance(expr, Expression) else expr

    if not isinstance(content, dict):
        return expr

    mutate = content.get("Mutate")

    # Simplify Add(5, 3) -> 8
    if mutate == "Add":
        values = [v for k, v in content.items() if k != "Mutate"]
        if all(isinstance(v, (int, float)) for v in values):
            return Expression(sum(values))

    # Simplify Multiply(2, 3) -> 6
    if mutate == "Multiply":
        values = [v for k, v in content.items() if k != "Mutate"]
        if all(isinstance(v, (int, float)) for v in values):
            result = 1
            for v in values:
                result *= v
            return Expression(result)

    return expr


# ==========================================
# PART 21: EXPRESSION OPTIMIZATION SYSTEM
# ==========================================


class PurityAnalyzer:
    """Classifies mutators as pure (deterministic) or impure (game state dependent)."""

    PURE_MUTATORS = {
        # Arithmetic
        "Add",
        "Subtract",
        "Multiply",
        "Divide",
        "Pow",
        "Modulo",
        "Round",
        "Ceil",
        "Floor",
        "Clamp",
        "Max",
        "Min",
        "Nonzero",
        # Boolean Logic
        "And",
        "Or",
        "Not",
        "Nand",
        "Nor",
        "Xor",
        # Conditionals
        "If",
        "IfFloat",
        # String Conversion
        "Float2String",
        "Int2String",
        "Join",
    }

    @staticmethod
    def is_pure(mutator_name: str) -> bool:
        """Returns True if mutator is deterministic (can be evaluated at compile-time)."""
        return mutator_name in PurityAnalyzer.PURE_MUTATORS


class CircularDependencyDetector:
    """Detects circular variable dependencies and reports with detailed error info."""

    def detect(self, variables: Dict[str, Variable]) -> None:
        """Raises ValidationError if circular dependency found.

        Args:
            variables: Dictionary of Variable objects

        Raises:
            ValidationError: If circular dependency detected
        """
        if not variables:
            return

        # Build dependency graph: var_name → set of variables it references
        deps = {}
        for var_name, var_obj in variables.items():
            deps[var_name] = self._extract_var_references(var_obj.Value)

        # Check for cycles using DFS
        for start_var in variables:
            visited = set()
            rec_stack = set()
            path = []

            if self._has_cycle_dfs(start_var, deps, visited, rec_stack, path):
                cycle_path = " → ".join(path + [start_var])
                raise ValidationError(
                    f"Circular variable dependency detected!\n"
                    f"Cycle: {cycle_path}\n"
                    f"This circular dependency must be resolved before optimization can proceed.\n"
                    f"Variables involved: {sorted(rec_stack)}"
                )

    def _extract_var_references(self, expr: Any) -> set:
        """Extract all variable names referenced in an expression."""
        refs = set()

        if isinstance(expr, dict):
            if "Var" in expr:
                refs.add(expr["Var"])
            for v in expr.values():
                refs.update(self._extract_var_references(v))
        elif isinstance(expr, list):
            for item in expr:
                refs.update(self._extract_var_references(item))

        return refs

    def _has_cycle_dfs(
        self, node: str, deps: dict, visited: set, rec_stack: set, path: list
    ) -> bool:
        """Detect cycles using DFS with recursion stack."""
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in deps.get(node, set()):
            if neighbor not in visited:
                if self._has_cycle_dfs(neighbor, deps, visited, rec_stack, path):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(node)
        path.pop()
        return False


class VariableInliner:
    """Inlines constant variable values during optimization."""

    def __init__(self, circular_detector: CircularDependencyDetector):
        self.circular_detector = circular_detector

    def inline(self, expr: Expression, constant_map: Dict[str, Any]) -> Expression:
        """Replace variable references with their constant values if possible.

        Args:
            expr: Expression to inline
            constant_map: Dictionary mapping variable names to their constant literal values

        Returns:
            Expression with variable references replaced by constants
        """
        # constant_map already contains only literal constants, so use it directly
        return self._inline_recursive(expr, constant_map)

    def _inline_recursive(self, expr: Any, const_vars: dict) -> Any:
        """Recursively inline variable references."""
        if isinstance(expr, Expression):
            content = expr.content
        elif isinstance(expr, dict):
            content = expr
        else:
            return expr

        if isinstance(content, dict):
            if "Var" in content and len(content) == 1:
                var_name = content["Var"]
                if var_name in const_vars:
                    return const_vars[var_name]

            # Recursively inline in children
            inlined = {}
            for k, v in content.items():
                if isinstance(v, (Expression, dict)):
                    inlined[k] = self._inline_recursive(v, const_vars)
                elif isinstance(v, list):
                    inlined[k] = [
                        self._inline_recursive(item, const_vars) for item in v
                    ]
                else:
                    inlined[k] = v

            return Expression(inlined) if isinstance(expr, Expression) else inlined

        return expr


class ConstantFolder:
    """Recursively folds deterministic expressions with constant operands."""

    def __init__(self, purity_analyzer: PurityAnalyzer, precision: int = 4):
        self.purity = purity_analyzer
        self.precision = precision

    def optimize(
        self, expr: Any, variables: Optional[Dict[str, Variable]] = None
    ) -> Any:
        """Fold constant expressions recursively (bottom-up)."""
        if variables is None:
            variables = {}

        # 1. Handle primitives directly (Base case)
        if isinstance(expr, float):
            return self._round(expr)
        if isinstance(expr, (int, str, bool, type(None))):
            return expr

        # 2. Extract content
        if isinstance(expr, Expression):
            content = expr.content
        else:
            content = expr

        # 3. Handle Lists (recurse)
        if isinstance(content, list):
            return [self.optimize(item, variables) for item in content]

        # 4. Handle Dictionaries (recurse through children)
        if isinstance(content, dict):
            optimized = {}
            for k, v in content.items():
                # Recursively optimize Expressions, dicts, and lists
                if isinstance(v, (Expression, dict, list)):
                    optimized[k] = self.optimize(v, variables)
                # explicit check for float to round simple values (like RateOfChange)
                elif isinstance(v, float):
                    optimized[k] = self._round(v)
                else:
                    optimized[k] = v
            content = optimized

            # 5. Try to fold "Mutate" operations
            if "Mutate" in content:
                mutator_name = content.get("Mutate")
                if mutator_name and self.purity.is_pure(mutator_name):
                    folded = self._try_fold(mutator_name, content)
                    if folded is not None:
                        return folded

        return Expression(content) if isinstance(expr, Expression) else content

    def _round(self, value):
        """Rounds floats to the configured precision. Returns ints as-is."""
        if isinstance(value, float):
            return round(value, self.precision)
        return value

    def _try_fold(self, mutator_name: str, content: dict):
        """Attempt to fold a mutator. Returns None if cannot fold."""

        # Extract operands (skip "Mutate" key)
        operands = {k: v for k, v in content.items() if k != "Mutate"}

        if mutator_name == "Add":
            a = operands.get("A", 0)
            b = operands.get("B", 0)
            # Identity: x + 0 = x
            if b == 0:
                return a
            if a == 0:
                return b

        elif mutator_name == "Subtract":
            a = operands.get("A", 0)
            b = operands.get("B", 0)
            # Identity: x - 0 = x
            if b == 0:
                return a
            # Note: 0 - x is not x, it is -x (which we don't simplify here yet)

        elif mutator_name == "Multiply":
            a = operands.get("A", 1)
            b = operands.get("B", 1)
            # Identity: x * 1 = x
            if b == 1:
                return a
            if a == 1:
                return b
            # Zero property: x * 0 = 0 (Evaluates to 0 even if x is complex)
            if a == 0 or b == 0:
                return 0

        elif mutator_name == "Divide":
            a = operands.get("A")
            b = operands.get("B")
            # Identity: x / 1 = x
            if b == 1:
                return a
            # Zero property: 0 / x = 0 (assuming x != 0)
            if a == 0:
                return 0

        # Check if all operands are constant
        if not self._all_constant(operands):
            return None

        try:
            if mutator_name == "Add":
                return self._round(
                    sum(v for v in operands.values() if isinstance(v, (int, float)))
                )

            elif mutator_name == "Subtract":
                a = operands.get("A", 0)
                b = operands.get("B", 0)
                return (
                    self._round(a - b)
                    if isinstance(a, (int, float)) and isinstance(b, (int, float))
                    else None
                )

            elif mutator_name == "Multiply":
                result = 1
                for v in operands.values():
                    if not isinstance(v, (int, float)):
                        return None
                    result *= v
                return self._round(result)

            elif mutator_name == "Divide":
                a = operands.get("A")
                b = operands.get("B")
                if (
                    isinstance(a, (int, float))
                    and isinstance(b, (int, float))
                    and b != 0
                ):
                    return self._round(a / b)
                return None

            elif mutator_name == "Pow":
                a = operands.get("A")
                b = operands.get("B")
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    return self._round(a**b)
                return None

            elif mutator_name == "Modulo":
                a = operands.get("A")
                b = operands.get("B")
                if (
                    isinstance(a, (int, float))
                    and isinstance(b, (int, float))
                    and b != 0
                ):
                    return self._round(a % b)
                return None

            elif mutator_name == "Round":
                val = operands.get("Value")
                if isinstance(val, (int, float)):
                    return round(val)
                return None

            elif mutator_name == "Ceil":
                val = operands.get("Value")
                if isinstance(val, (int, float)):
                    import math

                    return math.ceil(val)
                return None

            elif mutator_name == "Floor":
                val = operands.get("Value")
                if isinstance(val, (int, float)):
                    import math

                    return math.floor(val)
                return None

            elif mutator_name == "Max":
                values = [v for v in operands.values() if isinstance(v, (int, float))]
                if len(values) == len(operands):
                    return max(values) if values else None
                return None

            elif mutator_name == "Min":
                values = [v for v in operands.values() if isinstance(v, (int, float))]
                if len(values) == len(operands):
                    return min(values) if values else None
                return None

            elif mutator_name == "And":
                if all(isinstance(v, bool) for v in operands.values()):
                    return all(operands.values())
                return None

            elif mutator_name == "Or":
                if all(isinstance(v, bool) for v in operands.values()):
                    return any(operands.values())
                return None

            elif mutator_name == "Not":
                val = operands.get("Value")
                if isinstance(val, bool):
                    return not val
                return None

            elif mutator_name == "If":
                cond = operands.get("Condition")
                if isinstance(cond, bool):
                    return operands.get("Then") if cond else operands.get("Else")
                return None

            elif mutator_name == "IfFloat":
                value = operands.get("Value")
                then_val = operands.get("Then")
                else_val = operands.get("Else")

                # Find operator
                op_key = None
                benchmark = None
                for key in ["==", "!=", ">", "<", ">=", "<="]:
                    if key in content:
                        op_key = key
                        benchmark = content[key]
                        break

                if (
                    op_key
                    and isinstance(value, (int, float))
                    and isinstance(benchmark, (int, float))
                ):
                    if op_key == "==":
                        result = value == benchmark
                    elif op_key == "!=":
                        result = value != benchmark
                    elif op_key == ">":
                        result = value > benchmark
                    elif op_key == "<":
                        result = value < benchmark
                    elif op_key == ">=":
                        result = value >= benchmark
                    elif op_key == "<=":
                        result = value <= benchmark
                    else:
                        return None

                    return self._round(then_val) if result else self._round(else_val)

                return None

        except (ZeroDivisionError, TypeError, ValueError):
            return None

        return None

    def _all_constant(self, operands: dict) -> bool:
        """Check if all operands are constants (no variables, no mutators)."""
        for v in operands.values():
            if isinstance(v, dict) and ("Mutate" in v or "Var" in v):
                return False
            if isinstance(v, Expression):
                return False
        return True


class NonZeroPromoter:
    """Promotes (value != 0) patterns to NonZero mutator."""

    def optimize(self, expr: Any) -> Any:
        """Promote IfFloat(val, "!=", 0, ...) patterns to NonZero.

        Args:
            expr: Expression to optimize

        Returns:
            Optimized expression
        """
        if isinstance(expr, Expression):
            content = expr.content
        elif isinstance(expr, dict):
            content = expr
        elif isinstance(expr, list):
            return [self.optimize(item) for item in expr]
        else:
            return expr

        if isinstance(content, dict):
            # Recursively optimize children first
            optimized = {}
            for k, v in content.items():
                if isinstance(v, (Expression, dict)):
                    optimized[k] = self.optimize(v)
                elif isinstance(v, list):
                    optimized[k] = [self.optimize(item) for item in v]
                else:
                    optimized[k] = v
            content = optimized

            # Check if this is a NonZero-promotable IfFloat
            if content.get("Mutate") == "IfFloat":
                value = content.get("Value")
                benchmark = content.get("!=") if "!=" in content else None
                then_val = content.get("Then")
                else_val = content.get("Else")

                if benchmark == 0:
                    # Pattern: IfFloat(val, "!=", 0, True, False) → NonZero(val)
                    if then_val is True and else_val is False:
                        return Expression({"Mutate": "Nonzero", "Value": value})

                    # Pattern: IfFloat(val, "!=", 0, False, True) → Not(NonZero(val))
                    elif then_val is False and else_val is True:
                        return Expression(
                            {
                                "Mutate": "Not",
                                "Value": {"Mutate": "Nonzero", "Value": value},
                            }
                        )

        return Expression(content) if isinstance(expr, Expression) else content


class RedundantSelectorRemover:
    """Detects and removes redundant selectors and specific redundant branches."""

    BY_MUTATORS = {
        "ByBiome",
        "ByDNA",
        "ByDDStage",
        "ByMissionType",
        "BySecondary",
        "ByEscortPhase",
        "ByRefineryPhase",
        "BySaboPhase",
        "BySalvagePhase",
    }

    ARRAY_MUTATORS = {"ByPlayerCount", "ByResuppliesCalled"}

    def optimize(self, expr: Any) -> Any:
        """Remove redundant branches and selectors."""
        if isinstance(expr, Expression):
            content = expr.content
        elif isinstance(expr, dict):
            content = expr
        elif isinstance(expr, list):
            return [self.optimize(item) for item in expr]
        else:
            return expr

        if isinstance(content, dict):
            optimized = {}
            for k, v in content.items():
                if isinstance(v, (Expression, dict)):
                    optimized[k] = self.optimize(v)
                elif isinstance(v, list):
                    optimized[k] = [self.optimize(item) for item in v]
                else:
                    optimized[k] = v
            content = optimized

            mutator_name = content.get("Mutate")

            # Helper to safely get raw value for comparison
            def get_raw(val):
                return val.content if isinstance(val, Expression) else val

            # 2. Logic for Key-Value Mutators (ByMissionType, etc)
            if mutator_name in self.BY_MUTATORS:
                default_val = content.get("Default")
                raw_default = get_raw(default_val)

                # A. Partial Stripping
                keys_to_remove = []
                for k, v in content.items():
                    if k in {"Mutate", "Default", "Alias"}:
                        continue
                    if get_raw(v) == raw_default:
                        keys_to_remove.append(k)

                for k in keys_to_remove:
                    del content[k]

                # B. Total Collapse
                remaining_keys = [
                    k for k in content if k not in {"Mutate", "Default", "Alias"}
                ]
                if not remaining_keys:
                    return default_val

            # 3. Logic for Array Mutators (ByPlayerCount)
            elif mutator_name in self.ARRAY_MUTATORS:
                values = content.get("Values")
                if isinstance(values, list) and values:
                    first_raw = get_raw(values[0])

                    # STRICT CHECK: All items must match the first one
                    is_redundant = True
                    for item in values[1:]:
                        if get_raw(item) != first_raw:
                            is_redundant = False
                            break

                    if is_redundant:
                        return values[0]

            # 4. Logic for Select
            elif mutator_name == "Select":
                default_val = content.get("Default")
                raw_default = get_raw(default_val)

                keys_to_remove = []
                for k, v in content.items():
                    if k in {"Mutate", "Select", "Default", "Alias"}:
                        continue
                    if get_raw(v) == raw_default:
                        keys_to_remove.append(k)

                for k in keys_to_remove:
                    del content[k]

                remaining_options = [
                    k
                    for k in content
                    if k not in {"Mutate", "Select", "Default", "Alias"}
                ]
                if not remaining_options:
                    return default_val

        return Expression(content) if isinstance(expr, Expression) else content


class ExpressionOptimizer:
    """Master optimizer coordinating all optimization passes."""

    def __init__(
        self,
        enable_circular_check=True,
        enable_inlining=True,
        enable_constant_folding=True,
        enable_nonzero=True,
        enable_selector_removal=True,
    ):
        self.enable_circular_check = enable_circular_check
        self.enable_inlining = enable_inlining
        self.enable_constant_folding = enable_constant_folding
        self.enable_nonzero = enable_nonzero
        self.enable_selector_removal = enable_selector_removal

        self.circular_detector = CircularDependencyDetector()
        self.purity_analyzer = PurityAnalyzer()
        self.variable_inliner = VariableInliner(self.circular_detector)
        self.constant_folder = ConstantFolder(self.purity_analyzer)
        self.nonzero_promoter = NonZeroPromoter()
        self.selector_remover = RedundantSelectorRemover()

    def optimize(self, profile: DifficultyProfile) -> DifficultyProfile:
        """Optimize all expressions in a difficulty profile.

        Args:
            profile: DifficultyProfile to optimize

        Returns:
            Optimized DifficultyProfile
        """
        import copy

        # Clone the profile to avoid modifying the original
        profile = copy.deepcopy(profile)

        # Extract variables as dictionary
        variables = self._extract_variables(profile.Vars)

        # Step 1: Circular dependency check (MANDATORY)
        if self.enable_circular_check:
            self.circular_detector.detect(variables)

        # Step 1b: Optimize variable values themselves
        # This folds pure expressions in variable definitions to constants
        optimized_variables = self._optimize_variables(variables)

        # Step 1c: Build constant propagation map from optimized variable values
        # This includes both literal constants and optimized folded expressions
        constant_map = self._build_constant_map(optimized_variables)

        # Step 2-5: Optimize all other expressions in the profile using constant_map
        if (
            self.enable_inlining
            or self.enable_constant_folding
            or self.enable_nonzero
            or self.enable_selector_removal
        ):
            profile = self._optimize_object(profile, constant_map)

        # Step 6: Remove variables that were completely inlined (became constants)
        # Keep only variables with non-literal values (expressions) that couldn't be folded
        vars_to_keep = {}
        for name, var in optimized_variables.items():
            # Skip if marked as no_optimize (must be kept as-is)
            if getattr(var, "no_optimize", False):
                vars_to_keep[name] = var
                continue

            # Keep variable if its value is not a simple constant
            value = var.Value if isinstance(var, Variable) else var
            if not isinstance(value, (int, float, bool, str)):
                # Complex value (Expression/dict), keep it
                vars_to_keep[name] = var

        # Update Vars - only keep non-constant variables
        if vars_to_keep:
            profile.Vars = list(vars_to_keep.values())
        else:
            profile.Vars = None

        return profile

    def _optimize_variables(
        self, variables: Dict[str, Variable]
    ) -> Dict[str, Variable]:
        """Optimize variable values recursively with inlining before folding.

        This handles interdependencies by:
        1. Building a constant map from literal variable values
        2. Iteratively inlining variable references and folding expressions
        3. Adding newly-folded constants to the map for next iteration

        This allows Coth(1*Pi) to fold to a constant by first inlining E and Pi values.
        Literal constants (int, float, bool, str) are kept as-is and not wrapped in Expressions.
        """
        # Build constant map from literal constants (no dependencies)
        constant_map = {}
        for name, var in variables.items():
            # Skip variables marked with no_optimize
            if getattr(var, "no_optimize", False):
                continue

            value = var.Value if isinstance(var, Variable) else var
            if isinstance(value, (int, float, bool, str)):
                constant_map[name] = value

        # Iteratively optimize variables until convergence
        optimized = {}
        max_iterations = 10
        for iteration in range(max_iterations):
            changed = False
            for name, var in variables.items():
                if name in optimized:
                    continue

                # Try to optimize this variable using current constant map
                value = var.Value if isinstance(var, Variable) else var

                # Don't process literal constants - keep them as-is
                if isinstance(value, (int, float, bool, str)):
                    optimized[name] = var
                    continue

                # Step 1: Inline variable references using current constant map
                if self.enable_inlining:
                    inlined = self.variable_inliner.inline(
                        value if isinstance(value, Expression) else Expression(value),
                        constant_map,
                    )
                else:
                    inlined = value

                # Step 2: Then fold the expression (now with variables inlined)
                optimized_value = self._optimize_expression_value(inlined)

                # Update var with optimized value
                var.Value = optimized_value
                optimized[name] = var

                # Step 3: If this became a constant, add it to the map for other variables
                if isinstance(optimized_value, (int, float, bool, str)):
                    constant_map[name] = optimized_value
                    changed = True

            # If nothing changed in this iteration, we're done
            if not changed:
                break

        # Optimize any remaining variables (shouldn't happen normally)
        for name, var in variables.items():
            if name not in optimized:
                value = var.Value if isinstance(var, Variable) else var

                # Don't process literal constants - keep them as-is
                if isinstance(value, (int, float, bool, str)):
                    optimized[name] = var
                    continue

                if self.enable_inlining:
                    value = self.variable_inliner.inline(
                        value if isinstance(value, Expression) else Expression(value),
                        constant_map,
                    )
                optimized_value = self._optimize_expression_value(value)
                var.Value = optimized_value
                optimized[name] = var

        return optimized

    def _optimize_expression_value(self, value: Any) -> Any:
        """Optimize a single expression value (used for variable values).

        Runs full optimization pipeline on variable values so they become constants
        if possible (e.g., Coth(1*Pi) → 0.54... or Add(2,3) → 5).
        """
        if isinstance(value, Expression):
            expr = value
            # Apply full optimization pipeline to variable values
            if self.enable_constant_folding:
                expr = self.constant_folder.optimize(expr, None)
            # Also try NonZero and selector removal for completeness
            if self.enable_nonzero:
                expr = self.nonzero_promoter.optimize(expr)
            if self.enable_selector_removal:
                expr = self.selector_remover.optimize(expr)
            return expr
        elif isinstance(value, dict):
            # Handle dict representation
            if self.enable_constant_folding:
                value = self.constant_folder.optimize(value, None)
            if self.enable_nonzero:
                value = self.nonzero_promoter.optimize(value)
            if self.enable_selector_removal:
                value = self.selector_remover.optimize(value)
            return value
        return value

    def _build_constant_map(self, variables: Dict[str, Variable]) -> Dict[str, Any]:
        """Build a map of variables that have constant (non-expression) values.

        This is used for constant propagation during optimization.
        Variables with expression values are left alone.
        Variables marked with no_optimize are excluded from constant propagation.
        """
        const_map = {}
        for name, var in variables.items():
            # Skip variables marked with no_optimize
            if getattr(var, "no_optimize", False):
                continue

            value = var.Value if isinstance(var, Variable) else var
            # Only include literal constants, not expressions
            if isinstance(value, (int, float, bool, str)):
                const_map[name] = value
        return const_map

    def _extract_variables(self, vars_input) -> Dict[str, Variable]:
        """Convert Vars input (dict or list) to dictionary of Variables."""
        if isinstance(vars_input, dict):
            # Convert to Variables if needed
            result = {}
            for name, var_def in vars_input.items():
                if isinstance(var_def, Variable):
                    result[name] = var_def
                elif isinstance(var_def, VarDefinition):
                    # Create a Variable from VarDefinition
                    type_str = var_def.Type if isinstance(var_def.Type, str) else None
                    watch = var_def.Watch if isinstance(var_def.Watch, bool) else None
                    result[name] = Variable(name, var_def.Value, type_str, watch)
            return result
        elif isinstance(vars_input, list):
            return {v.Name: v for v in vars_input if isinstance(v, Variable)}
        return {}

    def _optimize_object(self, obj: Any, constant_map: Dict[str, Any]) -> Any:
        """Recursively optimize all Expression objects in an object.

        Args:
            obj: Object to optimize (may be Expression, dict, list, or CD2Object)
            constant_map: Map of variable names to their constant values for propagation
        """

        if isinstance(obj, Variable):
            # Skip optimization if marked with no_optimize
            if getattr(obj, "no_optimize", False):
                return obj

            # Check if this variable is a constant that should be inlined
            var_name = obj.Name if hasattr(obj, "Name") else None
            if var_name and var_name in constant_map and self.enable_inlining:
                # Replace the variable reference with its constant value
                return constant_map[var_name]
            # Otherwise keep the variable as-is
            return obj

        if isinstance(obj, Expression):
            expr = obj

            # Skip optimization if explicitly disabled on this expression
            if getattr(expr, "no_optimize", False):
                return expr

            # Step 2: Variable inlining (using constant_map)
            if self.enable_inlining and constant_map:
                expr = self.variable_inliner.inline(expr, constant_map)

            # Step 3: Constant folding
            if self.enable_constant_folding:
                expr = self.constant_folder.optimize(expr, None)

            # Step 4: NonZero promotion
            if self.enable_nonzero:
                expr = self.nonzero_promoter.optimize(expr)

            # Step 5: Selector removal
            if self.enable_selector_removal:
                expr = self.selector_remover.optimize(expr)

            return expr

        elif isinstance(obj, dict):
            return {k: self._optimize_object(v, constant_map) for k, v in obj.items()}

        elif isinstance(obj, list):
            return [self._optimize_object(item, constant_map) for item in obj]

        elif isinstance(obj, CD2Object):
            # Optimize all fields in the dataclass (but skip Vars field)
            for field_name in obj.__annotations__:
                # Never optimize the Vars field itself - variables stay unchanged
                if field_name == "Vars":
                    continue

                field_value = getattr(obj, field_name, None)
                if field_value is not None:
                    optimized_value = self._optimize_object(field_value, constant_map)
                    setattr(obj, field_name, optimized_value)
            return obj

        return obj


def generate_docs(profile: DifficultyProfile, output_path: str):
    """Generate markdown documentation for a difficulty profile."""
    doc = f"# {profile.Name}\n\n"

    if profile.Description:
        doc += f"{profile.Description}\n\n"

    doc += f"**Max Players:** {profile.MaxPlayers or 4}\n\n"

    if profile.Vars:
        doc += "## Variables\n\n"
        for var_name, var_def in profile.Vars.items():
            doc += f"- **{var_name}** ({var_def.Type}): `{var_def.Value}`\n"

    if profile.Enemies:
        doc += "\n## Custom Enemies\n\n"
        for ed_name, descriptor in profile.Enemies.items():
            doc += f"### {ed_name}\n"
            if descriptor.Base:
                doc += f"- Base: {descriptor.Base}\n"
            if descriptor.HealthMultiplier:
                doc += f"- Health: {descriptor.HealthMultiplier}x\n"

    with open(output_path, "w") as f:
        f.write(doc)


def import_from_json(path: str) -> DifficultyProfile:
    """Load a difficulty profile from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return DifficultyProfile(**data)


def export_to_json(profile: DifficultyProfile, path: str, minify=False):
    """Export profile to JSON with optional minification."""
    indent: Optional[int] = None if minify else 4
    profile.save(path, indent=indent)


# ==========================================
# PART 4: EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    # Example showing the Variadic arguments

    # "Or" with keyword args
    is_event = Or(
        PE=Mut("DuringPECountdown"),
        Extr=Mut("DuringExtraction"),
        Def=Mut("DuringDefend"),
    )
    # Output: {"Mutate": "Or", "PE": ..., "Extr": ..., "Def": ...}

    # "Add" with positional args
    basic_math = Add(10, 20, 30)
    # Output: {"Mutate": "Add", "A": 10, "B": 20, "C": 30}

    profile = DifficultyProfile(
        Name="Nightmares",
        Description="Example",
        MaxPlayers=16,
        Vars={"IsEvent": VarDefinition(Type="Boolean", Value=is_event)},
    )

    print(json.dumps(profile.to_dict(), indent=2))
