FS = "Fs"
FOIL = "Foil"
FG = "Fg"
PRES = "pressure"
DISCHARGE = "discharge"
WATER = "Fw"
PAA = "Fpaa"

DEFAULT_PENICILLIN_RECIPE_ORDER = [FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA]

FS_DEFAULT_PROFILE = [
    {"time": 3.0, "value": 8.0},
    {"time": 12.0, "value": 15.0},
    {"time": 16.0, "value": 30.0},
    {"time": 20.0, "value": 75.0},
    {"time": 24.0, "value": 150.0},
    {"time": 28.0, "value": 30.0},
    {"time": 32.0, "value": 37.0},
    {"time": 36.0, "value": 43.0},
    {"time": 40.0, "value": 47.0},
    {"time": 44.0, "value": 51.0},
    {"time": 48.0, "value": 57.0},
    {"time": 52.0, "value": 61.0},
    {"time": 56.0, "value": 65.0},
    {"time": 60.0, "value": 72.0},
    {"time": 64.0, "value": 76.0},
    {"time": 68.0, "value": 80.0},
    {"time": 72.0, "value": 84.0},
    {"time": 76.0, "value": 90.0},
    {"time": 80.0, "value": 116.0},
    {"time": 160.0, "value": 90.0},
    {"time": 230.0, "value": 80.0}
]

FOIL_DEFAULT_PROFILE = [
    {"time": 4.0, "value": 22.0},
    {"time": 16.0, "value": 30.0},
    {"time": 56.0, "value": 35.0},
    {"time": 60.0, "value": 34.0},
    {"time": 64.0, "value": 33.0},
    {"time": 68.0, "value": 32.0},
    {"time": 72.0, "value": 31.0},
    {"time": 76.0, "value": 30.0},
    {"time": 80.0, "value": 29.0},
    {"time": 230.0, "value": 23.0},
]

FG_DEFAULT_PROFILE = [
    {"time": 8.0, "value": 30.0},
    {"time": 20.0, "value": 42.0},
    {"time": 40.0, "value": 55.0},
    {"time": 90.0, "value": 60.0},
    {"time": 200.0, "value": 75.0},
    {"time": 230.0, "value": 65.0}
]

PRESS_DEFAULT_PROFILE = [
    {"time": 12.4, "value": 0.6},
    {"time": 25.0, "value": 0.7},
    {"time": 30.0, "value": 0.8},
    {"time": 40.0, "value": 0.9},
    {"time": 100.0, "value": 1.1},
    {"time": 150.0, "value": 1.0},
    {"time": 200.0, "value": 0.9},
    {"time": 230.0, "value": 0.9},
]

DISCHARGE_DEFAULT_PROFILE = [
    {"time": 100.0, "value": 0.0},
    {"time": 102.0, "value": 4000.0},
    {"time": 130.0, "value": 0.0},
    {"time": 132.0, "value": 4000.0},
    {"time": 150.0, "value": 0.0},
    {"time": 152.0, "value": 4000.0},
    {"time": 170.0, "value": 0.0},
    {"time": 172.0, "value": 4000.0},
    {"time": 190.0, "value": 0.0},
    {"time": 192.0, "value": 4000.0},
    {"time": 210.0, "value": 0.0},
    {"time": 212.0, "value": 4000.0},
    {"time": 230.0, "value": 0.0}
]

WATER_DEFAULT_PROFILE = [
    {"time": 50.0, "value": 0.0},
    {"time": 75.0, "value": 500.0},
    {"time": 150.0, "value": 100.0},
    {"time": 160.0, "value": 0.0},
    {"time": 170.0, "value": 400.0},
    {"time": 200.0, "value": 150.0},
    {"time": 230.0, "value": 250.0}
]

PAA_DEFAULT_PROFILE = [
    {"time": 5.0, "value": 5.0},
    {"time": 40.0, "value": 0.0},
    {"time": 200.0, "value": 10.0},
    {"time": 230.0, "value": 4.0}
]
