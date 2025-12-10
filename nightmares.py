from cd2_schema import *

if __name__ == "__main__":
    # Constants
    Name = "Nightmares"
    Description = "v0.25"

    # Variables
    BaseGruntModifier = Variable("BaseGruntModifier", 2, "Float").Alias(
        "BaseGruntModifier"
    )

    BaseEnemyCountModifier = (
        Variable("BaseEnemyCountModifier", 2.3, "Float")
        .Alias("BaseEnemyCountModifier")
        .SetOptimize(False)
    )

    HighPlayerEnemyCountNerf = Variable(
        "HighPlayerEnemyCountNerf", -0.01, "Float"
    ).Alias("HighPlayerEnemyCountNerf")

    BaseNitraMultiplier = Variable("BaseNitraMultiplier", 1, "Float").Alias(
        "BaseNitraMultiplier"
    )
    HighPlayerCountNitraBonus = Variable(
        "HighPlayerCountNitraBonus", 0.05, "Float"
    ).Alias("HighPlayerCountNitraBonus")

    # Aliases
    DoubleOnExtraction = If(IsExtraction, 2, 1).Alias("DoubleOnExtraction")

    # Caps
    SwarmerCap = (
        ByPlayerCount(120, 120 + 30 * DwarfCount).Alias("PlayerScale")
        * If(DuringEggAmbush(), 1.5, 1).Alias("AmbushScale")
        * DoubleOnExtraction
    )

    EnemyCap = (
        ByPlayerCount(90, 150, 120 + 10 * DwarfCount).Alias("PlayerScale")
        * If(DuringEggAmbush(), 2, 1).Alias("AmbushScale")
        * DoubleOnExtraction
    )

    # EnemyCountModifier
    BaseModifier = ByPlayerCount(
        [
            BaseEnemyCountModifier,
            BaseEnemyCountModifier
            + 0.2 * DwarfCount
            + IfRefinery(If(DwarfCount >= 3, -0.1, 0), 0),
        ]
    ).Alias("Base")

    AmbushModifier = If(DuringEggAmbush(stopping_after=10), 3, 1).Alias("Ambush")

    SwarmOnEgg = If(DuringGenericSwarm, 1.15, 1)

    EscortMultipliers = ByEscortPhase(
        1,
        Moving=1.3,
        FinalEventA=1.1,
        FinalEventB=1.3,
        FinalEventC=1.5,
        FinalEventD=1.7,
    )

    PEModifier = 1 + PerMinuteToPerSecond(0.01)

    RefineryMultipliers = ByRefineryPhase(
        1,
        PipesConnected=1.3,
        Refining=1.75,
        RefiningStalled=0.8,
        RefiningComplete=1.3,
        RocketLaunched=1.75,
    )

    SaboMultipliers = BySaboPhase(
        1,
        Hacking=1.5,
        BetweenHacks=1.3,
        Phase1Vent=1.1,
        Phase1Eye=1.2,
        Phase2Vent=1.3,
        Phase2Eye=1.4,
        Phase3Vent=1.5,
        Phase3Eye=1.6,
    )

    SalvageMultiplers = BySalvagePhase(
        1, Uplink=1 + DefenseProgress, Refuel=1.3 + DefenseProgress
    )

    DeepScanModifiers = If(DuringDrillevator, 2, 1)

    BlackBoxModifiers = (1 + IfSalvage(0, 0.3)).Alias("BlackBoxModifier")

    ExtractionModifier = If(
        IsExtraction, ByTimeWhen(IsExtraction, 1, PerMinuteToPerSecond(0.5)), 1
    ).Alias("OnExtraction")

    # Warnings
    BannedWarnings = sorted(
        [
            "WRN_Plague",
            "WRN_RivalIncursion",
            "WRN_NoOxygen",
            "WRN_Ghost",
            "WRN_RockInfestation",
            "MMUT_ExterminationContract",
            "MMUT_BloodSugar",
            "MMUT_ExplosiveEnemies",
            "MMUT_GoldRush",
            "MMUT_LowGravity",
            "MMUT_OxygenRich",
            "MMUT_Weakspot",
            "WRN_BulletHell",
            "WRN_CaveLeechDen",
            "WRN_ExploderInfestation",
            "WRN_HeroEnemies",
            "WRN_InfestedEnemies",
            "WRN_LethalEnemies",
            "WRN_MacteraCave",
            "WRN_NoShields",
            "WRN_RegenerativeEnemies",
            "WRN_Swarmagedon",
        ],
        key=len,
    )

    # Dwarves
    DwarvesCFG = DwarvesSettings(6, 3)

    def efficiency_ratio_key(b: WeightedBracket):
        if isinstance(b.weight, (float, int)):
            w = float(b.weight)
        else:
            return 0
        if isinstance(b.min, (float, int)):
            mn = float(b.min)
        else:
            return 0
        if isinstance(b.max, (float, int)):
            mx = float(b.max)
        else:
            return 0
        return w / (mx - mn)

    # EWI
    BaseDelay = 145 + ByDNA(0, {"Mining,x,2": 12}).Alias("Base")
    FrequencyIncrease = PerMinuteToPerSecond(10) / (
        ByDNA(
            10,
            {
                "Mining,2": 12,
                "Mining,x,2": 15,
                "Mining,x,3": 20,
                "PE": 12,
                "PE,3": 20,
                "Escort,2": 20,
                "Escort,3": 25,
                "Refinery,x,2": 15,
                "Refinery,x,3": 17.5,
                "Sabotage": 20,
                "DeepScan": 15,
            },
        )
        + ByPlayerCount(0)
    )

    WaveInterval = WeightedBracket(
        1,
        BaseDelay - FrequencyIncrease,
        BaseDelay + 40 - FrequencyIncrease,
    )

    # Difficulty
    difficulty = DifficultyProfile(
        Name=Name,
        Description=Description,
        MaxPlayers=16,
        # ======================================================
        Vars=[
            BaseGruntModifier,
            BaseEnemyCountModifier,
            HighPlayerEnemyCountNerf,
            BaseNitraMultiplier,
            HighPlayerCountNitraBonus,
            E,
            Pi,
        ],
        # ======================================================
        Warnings=WarningSettings(Banned=BannedWarnings),
        # ======================================================
        Dwarves=DwarvesCFG,
        # ======================================================
        Caps=CapSettings(SwarmerCap, EnemyCap),
        # ======================================================
        NitraMultiplier=BaseNitraMultiplier
        + ByDNA(0, Refinery=0.1)
        + (Ceil(DwarfCount / 4) - 1) * HighPlayerCountNitraBonus,
        # ======================================================
        Resupply=ResupplySettings(ByResuppliesCalled([20, 60])),
        # ======================================================
        DifficultySetting=GlobalDifficultySettings(
            BaseHazard="Hazard 5",
            ExtraLargeEnemyDamageResistance=[1, 1.3, 1.7, 2.1],
            ExtraLargeEnemyDamageResistanceB=[1, 1.3, 1.5, 1.7],
            ExtraLargeEnemyDamageResistanceC=[1, 1.3, 1.7, 2.1],
            ExtraLargeEnemyDamageResistanceD=[1.5, 1.55, 1.75, 1.9],
            EnemyDamageResistance=[1.4, 1.5, 1.6, 1.75],
            EnemyDamageModifier=4,
            EnemyCountModifier=BaseModifier
            * AmbushModifier
            * ExtractionModifier
            * BlackBoxModifiers
            * ByDNA(
                1,
                Egg=SwarmOnEgg,
                Escort=EscortMultipliers,
                Refinery=RefineryMultipliers,
                DeepScan=DeepScanModifiers,
                PE=PEModifier,
                Sabotage=SaboMultipliers,
                Salvage=SalvageMultiplers,
            ).Alias("ByMissionType"),
            EncounterDifficulty=sorted(
                [
                    WeightedBracket(1, 615, 800),
                    WeightedBracket(6, 520, 715),
                    WeightedBracket(1, 525, 620),
                ],
                key=efficiency_ratio_key,
            ),
            StationaryDifficulty=[WeightedBracket(1, ByDNA(375), ByDNA(525))],
            EnemyWaveInterval=[WaveInterval],
        ),
    )

    difficulty.save("out.json")
