
COLOUR_MAP = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'purple': (255, 0, 255),
    'cyan': (0, 255, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'grey': (128, 128, 128),
    'lightgrey': (192, 192, 192),
    'darkgrey': (64, 64, 64),
    'lightgreen': (144, 238, 144),
    'pink': (255, 192, 203),
    'lightblue': (173, 216, 230),
    'brown': (165, 42, 42),
    'darkgreen': (0, 100, 0),
    'darkred': (139, 0, 0),
    'softred': (216, 0, 68),
    'softgreen': (0, 216, 87),
    'violet': (142, 0, 216),
}

# converts colour map from RGB to BGR (expected by openCV)
COLOUR_MAP_BGR = {k: (v[2], v[1], v[0]) for k, v in COLOUR_MAP.items()}
