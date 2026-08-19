"""Image -> watertight face STL conversion.

Reconstructed from the original ``back_head_v6_2.py`` as a single-file stage.
This version estimates metric scale from MediaPipe iris landmarks instead of
forcing every face to the same 200 mm landmark height.

Changes from the historical script:
- uses ``project_paths.py`` instead of user-specific absolute paths
- removes persistent intermediate vertices/faces CSV files
- removes enclosure / combined-STL generation from this stage
- keeps the original MediaPipe landmark extraction, mesh subdivision,
  smoothing, back-head reconstruction, and final STL orientation conversion
- estimates mm/pixel scale from the mean detected iris diameter
- fixes triangle winding and validates watertight/volume status before export

Default data flow:
    ai-cfd-data/01_images/<face_id>.jpg
        -> ai-cfd-data/02_stl/<face_id>.stl
"""

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# Repository / data paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_paths import IMAGE_DIR, STL_DIR


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}

# MediaPipe Face Mesh with refine_landmarks=True returns 478 landmarks:
# 468 base face landmarks + 10 iris landmarks (5 per eye).
BASE_FACE_LANDMARK_COUNT = 468
RIGHT_IRIS_RING = (469, 470, 471, 472)
LEFT_IRIS_RING = (474, 475, 476, 477)
DEFAULT_IRIS_DIAMETER_MM = 11.7


# ---------------------------------------------------------------------------
# Original face topology
# ---------------------------------------------------------------------------

mesh_index = np.array([
    [127,  34, 139],
    [ 11,   0,  37],
    [232, 231, 120],
    [ 72,  37,  39],
    [128, 121,  47],
    [232, 121, 128],
    [104,  69,  67],
    [175, 171, 148],
    [118,  50, 101],
    [ 73,  39,  40],
    [  9, 151, 108],
    [ 48, 115, 131],
    [194, 204, 211],
    [ 74,  40, 185],
    [ 80,  42, 183],
    [ 40,  92, 186],
    [230, 229, 118],
    [202, 212, 214],
    [ 83,  18,  17],
    [ 76,  61, 146],
    [160,  29,  30],
    [ 56, 157, 173],
    [106, 204, 194],
    [135, 214, 192],
    [203, 165,  98],
    [ 21,  71,  68],
    [ 51,  45,   4],
    [144,  24,  23],
    [ 77, 146,  91],
    [205,  50, 187],
    [201, 200,  18],
    [ 91, 106, 182],
    [ 90,  91, 181],
    [ 85,  84,  17],
    [206, 203,  36],
    [148, 171, 140],
    [ 92,  40,  39],
    [193, 189, 244],
    [159, 158,  28],
    [247, 246, 161],
    [236,   3, 196],
    [ 54,  68, 104],
    [193, 168,   8],
    [117, 228,  31],
    [189, 193,  55],
    [ 98,  97,  99],
    [126,  47, 100],
    [166,  79, 218],
    [155, 154,  26],
    [209,  49, 131],
    [135, 136, 150],
    [ 47, 126, 217],
    [223,  52,  53],
    [ 45,  51, 134],
    [211, 170, 140],
    [ 67,  69, 108],
    [ 43, 106,  91],
    [230, 119, 120],
    [226, 130, 247],
    [ 63,  53,  52],
    [238,  20, 242],
    [ 46,  70, 156],
    [ 78,  62,  96],
    [ 46,  53,  63],
    [143,  34, 227],
    [123, 117, 111],
    [ 44, 125,  19],
    [236, 134,  51],
    [216, 206, 205],
    [154, 153,  22],
    [ 39,  37, 167],
    [200, 201, 208],
    [ 36, 142, 100],
    [ 57, 212, 202],
    [ 20,  60,  99],
    [ 28, 158, 157],
    [ 35, 226, 113],
    [160, 159,  27],
    [204, 202, 210],
    [113, 225,  46],
    [ 43, 202, 204],
    [ 62,  76,  77],
    [137, 123, 116],
    [ 41,  38,  72],
    [203, 129, 142],
    [ 64,  98, 240],
    [ 49, 102,  64],
    [ 41,  73,  74],
    [212, 216, 207],
    [ 42,  74, 184],
    [169, 170, 211],
    [170, 149, 176],
    [105,  66,  69],
    [122,   6, 168],
    [123, 147, 187],
    [ 96,  77,  90],
    [ 65,  55, 107],
    [ 89,  90, 180],
    [101, 100, 120],
    [ 63, 105, 104],
    [ 93, 137, 227],
    [ 15,  86,  85],
    [129, 102,  49],
    [ 14,  87,  86],
    [ 55,   8,   9],
    [100,  47, 121],
    [145,  23,  22],
    [ 88,  89, 179],
    [  6, 122, 196],
    [ 88,  95,  96],
    [138, 172, 136],
    [215,  58, 172],
    [115,  48, 219],
    [ 42,  80,  81],
    [195,   3,  51],
    [ 43, 146,  61],
    [171, 175, 199],
    [ 81,  82,  38],
    [ 53,  46, 225],
    [144, 163, 110],
    [ 52,  65,  66],
    [229, 228, 117],
    [ 34, 127, 234],
    [107, 108,  69],
    [109, 108, 151],
    [ 48,  64, 235],
    [ 62,  78, 191],
    [129, 209, 126],
    [111,  35, 143],
    [117, 123,  50],
    [222,  65,  52],
    [ 19, 125, 141],
    [221,  55,  65],
    [  3, 195, 197],
    [ 25,   7,  33],
    [220, 237,  44],
    [ 70,  71, 139],
    [122, 193, 245],
    [247, 130,  33],
    [ 71,  21, 162],
    [170, 169, 150],
    [188, 174, 196],
    [216, 186,  92],
    [  2,  97, 167],
    [141, 125, 241],
    [164, 167,  37],
    [ 72,  38,  12],
    [ 38,  82,  13],
    [ 63,  68,  71],
    [226,  35, 111],
    [101,  50, 205],
    [206,  92, 165],
    [209, 198, 217],
    [165, 167,  97],
    [220, 115, 218],
    [133, 112, 243],
    [239, 238, 241],
    [214, 135, 169],
    [190, 173, 133],
    [171, 208,  32],
    [125,  44, 237],
    [ 86,  87, 178],
    [ 85,  86, 179],
    [ 84,  85, 180],
    [ 83,  84, 181],
    [201,  83, 182],
    [137,  93, 132],
    [ 76,  62, 183],
    [ 61,  76, 184],
    [ 57,  61, 185],
    [212,  57, 186],
    [214, 207, 187],
    [ 34, 143, 156],
    [ 79, 239, 237],
    [123, 137, 177],
    [ 44,   1,   4],
    [201, 194,  32],
    [ 64, 102, 129],
    [213, 215, 138],
    [ 59, 166, 219],
    [242,  99,  97],
    [  2,  94, 141],
    [ 75,  59, 235],
    [ 24, 110, 228],
    [ 25, 130, 226],
    [ 23,  24, 229],
    [ 22,  23, 230],
    [ 26,  22, 231],
    [112,  26, 232],
    [189, 190, 243],
    [221,  56, 190],
    [ 28,  56, 221],
    [ 27,  28, 222],
    [ 29,  27, 223],
    [ 30,  29, 224],
    [247,  30, 225],
    [238,  79,  20],
    [166,  59,  75],
    [ 60,  75, 240],
    [147, 177, 215],
    [ 20,  79, 166],
    [187, 147, 213],
    [112, 233, 244],
    [233, 128, 245],
    [128, 114, 188],
    [114, 217, 174],
    [131, 115, 220],
    [217, 198, 236],
    [198, 131, 134],
    [177, 132,  58],
    [143,  35, 124],
    [110, 163,   7],
    [228, 110,  25],
    [356, 389, 368],
    [ 11, 302, 267],
    [452, 350, 349],
    [302, 303, 269],
    [357, 343, 277],
    [452, 453, 357],
    [333, 332, 297],
    [175, 152, 377],
    [347, 348, 330],
    [303, 304, 270],
    [  9, 336, 337],
    [278, 279, 360],
    [418, 262, 431],
    [304, 408, 409],
    [310, 415, 407],
    [270, 409, 410],
    [450, 348, 347],
    [422, 430, 434],
    [313, 314,  17],
    [306, 307, 375],
    [387, 388, 260],
    [286, 414, 398],
    [335, 406, 418],
    [364, 367, 416],
    [423, 358, 327],
    [251, 284, 298],
    [281,   5,   4],
    [373, 374, 253],
    [307, 320, 321],
    [425, 427, 411],
    [421, 313,  18],
    [321, 405, 406],
    [320, 404, 405],
    [315,  16,  17],
    [426, 425, 266],
    [377, 400, 369],
    [322, 391, 269],
    [417, 465, 464],
    [386, 257, 258],
    [466, 260, 388],
    [456, 399, 419],
    [284, 332, 333],
    [417, 285,   8],
    [346, 340, 261],
    [413, 441, 285],
    [327, 460, 328],
    [355, 371, 329],
    [392, 439, 438],
    [382, 341, 256],
    [429, 420, 360],
    [364, 394, 379],
    [277, 343, 437],
    [443, 444, 283],
    [275, 440, 363],
    [431, 262, 369],
    [297, 338, 337],
    [273, 375, 321],
    [450, 451, 349],
    [446, 342, 467],
    [293, 334, 282],
    [458, 461, 462],
    [276, 353, 383],
    [308, 324, 325],
    [276, 300, 293],
    [372, 345, 447],
    [352, 345, 340],
    [274,   1,  19],
    [456, 248, 281],
    [436, 427, 425],
    [381, 256, 252],
    [269, 391, 393],
    [200, 199, 428],
    [266, 330, 329],
    [287, 273, 422],
    [250, 462, 328],
    [258, 286, 384],
    [265, 353, 342],
    [387, 259, 257],
    [424, 431, 430],
    [342, 353, 276],
    [273, 335, 424],
    [292, 325, 307],
    [366, 447, 345],
    [271, 303, 302],
    [423, 266, 371],
    [294, 455, 460],
    [279, 278, 294],
    [271, 272, 304],
    [432, 434, 427],
    [272, 407, 408],
    [394, 430, 431],
    [395, 369, 400],
    [334, 333, 299],
    [351, 417, 168],
    [352, 280, 411],
    [325, 319, 320],
    [295, 296, 336],
    [319, 403, 404],
    [330, 348, 349],
    [293, 298, 333],
    [323, 454, 447],
    [ 15,  16, 315],
    [358, 429, 279],
    [ 14,  15, 316],
    [285, 336,   9],
    [329, 349, 350],
    [374, 380, 252],
    [318, 402, 403],
    [  6, 197, 419],
    [318, 319, 325],
    [367, 364, 365],
    [435, 367, 397],
    [344, 438, 439],
    [272, 271, 311],
    [195,   5, 281],
    [273, 287, 291],
    [396, 428, 199],
    [311, 271, 268],
    [283, 444, 445],
    [373, 254, 339],
    [282, 334, 296],
    [449, 347, 346],
    [264, 447, 454],
    [336, 296, 299],
    [338,  10, 151],
    [278, 439, 455],
    [292, 407, 415],
    [358, 371, 355],
    [340, 345, 372],
    [346, 347, 280],
    [442, 443, 282],
    [ 19,  94, 370],
    [441, 442, 295],
    [248, 419, 197],
    [263, 255, 359],
    [440, 275, 274],
    [300, 383, 368],
    [351, 412, 465],
    [263, 467, 466],
    [301, 368, 389],
    [395, 378, 379],
    [412, 351, 419],
    [436, 426, 322],
    [  2, 164, 393],
    [370, 462, 461],
    [164,   0, 267],
    [302,  11,  12],
    [268,  12,  13],
    [293, 300, 301],
    [446, 261, 340],
    [330, 266, 425],
    [426, 423, 391],
    [429, 355, 437],
    [391, 327, 326],
    [440, 457, 438],
    [341, 382, 362],
    [459, 457, 461],
    [434, 430, 394],
    [414, 463, 362],
    [396, 369, 262],
    [354, 461, 457],
    [316, 403, 402],
    [315, 404, 403],
    [314, 405, 404],
    [313, 406, 405],
    [421, 418, 406],
    [366, 401, 361],
    [306, 408, 407],
    [291, 409, 408],
    [287, 410, 409],
    [432, 436, 410],
    [434, 416, 411],
    [264, 368, 383],
    [309, 438, 457],
    [352, 376, 401],
    [274, 275,   4],
    [421, 428, 262],
    [294, 327, 358],
    [433, 416, 367],
    [289, 455, 439],
    [462, 370, 326],
    [  2, 326, 370],
    [305, 460, 455],
    [254, 449, 448],
    [255, 261, 446],
    [253, 450, 449],
    [252, 451, 450],
    [256, 452, 451],
    [341, 453, 452],
    [413, 464, 463],
    [441, 413, 414],
    [258, 442, 441],
    [257, 443, 442],
    [259, 444, 443],
    [260, 445, 444],
    [467, 342, 445],
    [459, 458, 250],
    [289, 392, 290],
    [290, 328, 460],
    [376, 433, 435],
    [250, 290, 392],
    [411, 416, 433],
    [341, 463, 464],
    [453, 464, 465],
    [357, 465, 412],
    [343, 412, 399],
    [360, 363, 440],
    [437, 399, 456],
    [420, 456, 363],
    [401, 435, 288],
    [372, 383, 353],
    [339, 255, 249],
    [448, 261, 255],
    [133, 243, 190],
    [133, 155, 112],
    [ 33, 246, 247],
    [ 33, 130,  25],
    [398, 384, 286],
    [362, 398, 414],
    [362, 463, 341],
    [263, 359, 467],
    [263, 249, 255],
    [466, 467, 260],
    [ 75,  60, 166],
    [238, 239,  79],
    [162, 127, 139],
    [ 72,  11,  37],
    [121, 232, 120],
    [ 73,  72,  39],
    [114, 128,  47],
    [233, 232, 128],
    [103, 104,  67],
    [152, 175, 148],
    [119, 118, 101],
    [ 74,  73,  40],
    [107,   9, 108],
    [ 49,  48, 131],
    [ 32, 194, 211],
    [184,  74, 185],
    [191,  80, 183],
    [185,  40, 186],
    [119, 230, 118],
    [210, 202, 214],
    [ 84,  83,  17],
    [ 77,  76, 146],
    [161, 160,  30],
    [190,  56, 173],
    [182, 106, 194],
    [138, 135, 192],
    [129, 203,  98],
    [ 54,  21,  68],
    [  5,  51,   4],
    [145, 144,  23],
    [ 90,  77,  91],
    [207, 205, 187],
    [ 83, 201,  18],
    [181,  91, 182],
    [180,  90, 181],
    [ 16,  85,  17],
    [205, 206,  36],
    [176, 148, 140],
    [165,  92,  39],
    [245, 193, 244],
    [ 27, 159,  28],
    [ 30, 247, 161],
    [174, 236, 196],
    [103,  54, 104],
    [ 55, 193,   8],
    [111, 117,  31],
    [221, 189,  55],
    [240,  98,  99],
    [142, 126, 100],
    [219, 166, 218],
    [112, 155,  26],
    [198, 209, 131],
    [169, 135, 150],
    [114,  47, 217],
    [224, 223,  53],
    [220,  45, 134],
    [ 32, 211, 140],
    [109,  67, 108],
    [146,  43,  91],
    [231, 230, 120],
    [113, 226, 247],
    [105,  63,  52],
    [241, 238, 242],
    [124,  46, 156],
    [ 95,  78,  96],
    [ 70,  46,  63],
    [116, 143, 227],
    [116, 123, 111],
    [  1,  44,  19],
    [  3, 236,  51],
    [207, 216, 205],
    [ 26, 154,  22],
    [165,  39, 167],
    [199, 200, 208],
    [101,  36, 100],
    [ 43,  57, 202],
    [242,  20,  99],
    [ 56,  28, 157],
    [124,  35, 113],
    [ 29, 160,  27],
    [211, 204, 210],
    [124, 113,  46],
    [106,  43, 204],
    [ 96,  62,  77],
    [227, 137, 116],
    [ 73,  41,  72],
    [ 36, 203, 142],
    [235,  64, 240],
    [ 48,  49,  64],
    [ 42,  41,  74],
    [214, 212, 207],
    [183,  42, 184],
    [210, 169, 211],
    [140, 170, 176],
    [104, 105,  69],
    [193, 122, 168],
    [ 50, 123, 187],
    [ 89,  96,  90],
    [ 66,  65, 107],
    [179,  89, 180],
    [119, 101, 120],
    [ 68,  63, 104],
    [234,  93, 227],
    [ 16,  15,  85],
    [209, 129,  49],
    [ 15,  14,  86],
    [107,  55,   9],
    [120, 100, 121],
    [153, 145,  22],
    [178,  88, 179],
    [197,   6, 196],
    [ 89,  88,  96],
    [135, 138, 136],
    [138, 215, 172],
    [218, 115, 219],
    [ 41,  42,  81],
    [  5, 195,  51],
    [ 57,  43,  61],
    [208, 171, 199],
    [ 41,  81,  38],
    [224,  53, 225],
    [ 24, 144, 110],
    [105,  52,  66],
    [118, 229, 117],
    [227,  34, 234],
    [ 66, 107,  69],
    [ 10, 109, 151],
    [219,  48, 235],
    [183,  62, 191],
    [142, 129, 126],
    [116, 111, 143],
    [118, 117,  50],
    [223, 222,  52],
    [ 94,  19, 141],
    [222, 221,  65],
    [196,   3, 197],
    [ 45, 220,  44],
    [156,  70, 139],
    [188, 122, 245],
    [139,  71, 162],
    [149, 170, 150],
    [122, 188, 196],
    [206, 216,  92],
    [164,   2, 167],
    [242, 141, 241],
    [  0, 164,  37],
    [ 11,  72,  12],
    [ 12,  38,  13],
    [ 70,  63,  71],
    [ 31, 226, 111],
    [ 36, 101, 205],
    [203, 206, 165],
    [126, 209, 217],
    [ 98, 165,  97],
    [237, 220, 218],
    [237, 239, 241],
    [210, 214, 169],
    [140, 171,  32],
    [241, 125, 237],
    [179,  86, 178],
    [180,  85, 179],
    [181,  84, 180],
    [182,  83, 181],
    [194, 201, 182],
    [177, 137, 132],
    [184,  76, 183],
    [185,  61, 184],
    [186,  57, 185],
    [216, 212, 186],
    [192, 214, 187],
    [139,  34, 156],
    [218,  79, 237],
    [147, 123, 177],
    [ 45,  44,   4],
    [208, 201,  32],
    [ 98,  64, 129],
    [192, 213, 138],
    [235,  59, 219],
    [141, 242,  97],
    [ 97,   2, 141],
    [240,  75, 235],
    [229,  24, 228],
    [ 31,  25, 226],
    [230,  23, 229],
    [231,  22, 230],
    [232,  26, 231],
    [233, 112, 232],
    [244, 189, 243],
    [189, 221, 190],
    [222,  28, 221],
    [223,  27, 222],
    [224,  29, 223],
    [225,  30, 224],
    [113, 247, 225],
    [ 99,  60, 240],
    [213, 147, 215],
    [ 60,  20, 166],
    [192, 187, 213],
    [243, 112, 244],
    [244, 233, 245],
    [245, 128, 188],
    [188, 114, 174],
    [134, 131, 220],
    [174, 217, 236],
    [236, 198, 134],
    [215, 177,  58],
    [156, 143, 124],
    [ 25, 110,   7],
    [ 31, 228,  25],
    [264, 356, 368],
    [  0,  11, 267],
    [451, 452, 349],
    [267, 302, 269],
    [350, 357, 277],
    [350, 452, 357],
    [299, 333, 297],
    [396, 175, 377],
    [280, 347, 330],
    [269, 303, 270],
    [151,   9, 337],
    [344, 278, 360],
    [424, 418, 431],
    [270, 304, 409],
    [272, 310, 407],
    [322, 270, 410],
    [449, 450, 347],
    [432, 422, 434],
    [ 18, 313,  17],
    [291, 306, 375],
    [259, 387, 260],
    [424, 335, 418],
    [434, 364, 416],
    [391, 423, 327],
    [301, 251, 298],
    [275, 281,   4],
    [254, 373, 253],
    [375, 307, 321],
    [280, 425, 411],
    [200, 421,  18],
    [335, 321, 406],
    [321, 320, 405],
    [314, 315,  17],
    [423, 426, 266],
    [396, 377, 369],
    [270, 322, 269],
    [413, 417, 464],
    [385, 386, 258],
    [248, 456, 419],
    [298, 284, 333],
    [168, 417,   8],
    [448, 346, 261],
    [417, 413, 285],
    [326, 327, 328],
    [277, 355, 329],
    [309, 392, 438],
    [381, 382, 256],
    [279, 429, 360],
    [365, 364, 379],
    [355, 277, 437],
    [282, 443, 283],
    [281, 275, 363],
    [395, 431, 369],
    [299, 297, 337],
    [335, 273, 321],
    [348, 450, 349],
    [359, 446, 467],
    [283, 293, 282],
    [250, 458, 462],
    [300, 276, 383],
    [292, 308, 325],
    [283, 276, 293],
    [264, 372, 447],
    [346, 352, 340],
    [354, 274,  19],
    [363, 456, 281],
    [426, 436, 425],
    [380, 381, 252],
    [267, 269, 393],
    [421, 200, 428],
    [371, 266, 329],
    [432, 287, 422],
    [290, 250, 328],
    [385, 258, 384],
    [446, 265, 342],
    [386, 387, 257],
    [422, 424, 430],
    [445, 342, 276],
    [422, 273, 424],
    [306, 292, 307],
    [352, 366, 345],
    [268, 271, 302],
    [358, 423, 371],
    [327, 294, 460],
    [331, 279, 294],
    [303, 271, 304],
    [436, 432, 427],
    [304, 272, 408],
    [395, 394, 431],
    [378, 395, 400],
    [296, 334, 299],
    [  6, 351, 168],
    [376, 352, 411],
    [307, 325, 320],
    [285, 295, 336],
    [320, 319, 404],
    [329, 330, 349],
    [334, 293, 333],
    [366, 323, 447],
    [316,  15, 315],
    [331, 358, 279],
    [317,  14, 316],
    [  8, 285,   9],
    [277, 329, 350],
    [253, 374, 252],
    [319, 318, 403],
    [351,   6, 419],
    [324, 318, 325],
    [397, 367, 365],
    [288, 435, 397],
    [278, 344, 439],
    [310, 272, 311],
    [248, 195, 281],
    [375, 273, 291],
    [175, 396, 199],
    [312, 311, 268],
    [276, 283, 445],
    [390, 373, 339],
    [295, 282, 296],
    [448, 449, 346],
    [356, 264, 454],
    [337, 336, 299],
    [337, 338, 151],
    [294, 278, 455],
    [308, 292, 415],
    [429, 358, 355],
    [265, 340, 372],
    [352, 346, 280],
    [295, 442, 282],
    [354,  19, 370],
    [285, 441, 295],
    [195, 248, 197],
    [457, 440, 274],
    [301, 300, 368],
    [417, 351, 465],
    [251, 301, 389],
    [394, 395, 379],
    [399, 412, 419],
    [410, 436, 322],
    [326,   2, 393],
    [354, 370, 461],
    [393, 164, 267],
    [268, 302,  12],
    [312, 268,  13],
    [298, 293, 301],
    [265, 446, 340],
    [280, 330, 425],
    [322, 426, 391],
    [420, 429, 437],
    [393, 391, 326],
    [344, 440, 438],
    [458, 459, 461],
    [364, 434, 394],
    [428, 396, 262],
    [274, 354, 457],
    [317, 316, 402],
    [316, 315, 403],
    [315, 314, 404],
    [314, 313, 405],
    [313, 421, 406],
    [323, 366, 361],
    [292, 306, 407],
    [306, 291, 408],
    [291, 287, 409],
    [287, 432, 410],
    [427, 434, 411],
    [372, 264, 383],
    [459, 309, 457],
    [366, 352, 401],
    [  1, 274,   4],
    [418, 421, 262],
    [331, 294, 358],
    [435, 433, 367],
    [392, 289, 439],
    [328, 462, 326],
    [ 94,   2, 370],
    [289, 305, 455],
    [339, 254, 448],
    [359, 255, 446],
    [254, 253, 449],
    [253, 252, 450],
    [252, 256, 451],
    [256, 341, 452],
    [414, 413, 463],
    [286, 441, 414],
    [286, 258, 441],
    [258, 257, 442],
    [257, 259, 443],
    [259, 260, 444],
    [260, 467, 445],
    [309, 459, 250],
    [305, 289, 290],
    [305, 290, 460],
    [401, 376, 435],
    [309, 250, 392],
    [376, 411, 433],
    [453, 341, 464],
    [357, 453, 465],
    [343, 357, 412],
    [437, 343, 399],
    [344, 360, 440],
    [420, 437, 456],
    [360, 420, 363],
    [361, 401, 288],
    [265, 372, 353],
    [390, 339, 249],
    [161,163,144],
    [246,161,163],
    [246,7,163],
    [33,246,7],
    [161,144,160],
    [160,145,159],
    [159,145,153],
    [159,158,153],
    [158,153,154],
    [157,154,155],
    [173,155,133],
    [157,173,155],
    [158,157,154],
    [160,144,145],
    [362,398,382],
    [398,382,381],
    [398,384,381],
    [384,381,380],
    [384,380,385],
    [385,380,374],
    [385,374,386],
    [386,374,373],
    [386,373,387],
    [387,373,390],
    [387,390,388],
    [388,390,249],
    [466,249,263],
    [388,466,249],
    [255,339,448],
    [78,95,191],
    [191,95,80],
    [80,95,88],
    [80,88,81],
    [81,88,178],
    [81,178,82],
    [82,178,87],
    [82,87,13],
    [13,87,14],
    [13,14,317],
    [13,317,312],
    [312,317,402],
    [312,402,311],
    [311,402,318],
    [311,318,310],
    [310,318,324],
    [310,324,415],
    [415,324,308]
])


# ---------------------------------------------------------------------------
# 1. Face landmark extraction
# ---------------------------------------------------------------------------

def _pixel_xy(landmarks, index, width, height):
    """Return one normalized MediaPipe landmark as image-pixel (x, y)."""
    landmark = landmarks[index]
    return np.array(
        [
            landmark.x * width,
            landmark.y * height,
        ],
        dtype=float,
    )


def _iris_diameter_px(landmarks, ring_indices, width, height):
    """Estimate one iris diameter in image pixels.

    MediaPipe gives four contour landmarks around each iris, plus a center
    landmark. The contour indices form a ring. The two opposite-point
    distances are the two principal contour diameters; for a near-frontal
    eye, the larger one is used as the horizontal/major iris diameter.
    """
    p0 = _pixel_xy(landmarks, ring_indices[0], width, height)
    p1 = _pixel_xy(landmarks, ring_indices[1], width, height)
    p2 = _pixel_xy(landmarks, ring_indices[2], width, height)
    p3 = _pixel_xy(landmarks, ring_indices[3], width, height)

    diameter_a = float(np.linalg.norm(p0 - p2))
    diameter_b = float(np.linalg.norm(p1 - p3))

    return max(diameter_a, diameter_b)


def extract_iris_scaled_landmarks(
    image_path,
    iris_diameter_mm=DEFAULT_IRIS_DIAMETER_MM,
    max_iris_mismatch=0.25,
):
    """Extract 468 face landmarks and estimate metric scale from the irises.

    The mean detected iris diameter from the left and right eyes is mapped to
    ``iris_diameter_mm``. The resulting scale factor (mm/pixel) is applied
    uniformly to x, y, and MediaPipe's relative z coordinate.

    No fixed-face-height fallback is used. If iris detection is inconsistent,
    the image is rejected so that one dataset does not mix scaling methods.
    """
    image_path = Path(image_path)

    face_mesh_model = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Image not found or unreadable: {image_path}")

        height, width, _ = image.shape
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh_model.process(rgb)

        if not results.multi_face_landmarks:
            raise ValueError(f"No face landmarks detected: {image_path}")

        landmarks = results.multi_face_landmarks[0].landmark

        if len(landmarks) < 478:
            raise ValueError(
                "Iris landmarks were not returned. "
                "Use MediaPipe FaceMesh with refine_landmarks=True."
            )

        right_iris_px = _iris_diameter_px(
            landmarks,
            RIGHT_IRIS_RING,
            width,
            height,
        )
        left_iris_px = _iris_diameter_px(
            landmarks,
            LEFT_IRIS_RING,
            width,
            height,
        )

        if right_iris_px <= 1e-8 or left_iris_px <= 1e-8:
            raise ValueError("Invalid iris diameter detected.")

        mean_iris_px = 0.5 * (right_iris_px + left_iris_px)

        relative_mismatch = (
            abs(right_iris_px - left_iris_px)
            / mean_iris_px
        )

        if relative_mismatch > max_iris_mismatch:
            raise ValueError(
                "Left/right iris diameters are too inconsistent "
                f"(left={left_iris_px:.2f}px, "
                f"right={right_iris_px:.2f}px, "
                f"mismatch={relative_mismatch:.1%}). "
                "Use a more frontal, sharp image with both irises visible."
            )

        mm_per_pixel = iris_diameter_mm / mean_iris_px

        # Only the first 468 landmarks belong to the base facial mesh.
        face_landmarks = landmarks[:BASE_FACE_LANDMARK_COUNT]

        x = np.array(
            [landmark.x * width for landmark in face_landmarks],
            dtype=float,
        )
        y = np.array(
            [landmark.y * height for landmark in face_landmarks],
            dtype=float,
        )
        z = np.array(
            [landmark.z * width for landmark in face_landmarks],
            dtype=float,
        )

        # Center before metric scaling.
        x -= np.mean(x)
        y -= np.mean(y)
        z -= np.mean(z)

        # Uniform iris-based metric scale.
        x *= mm_per_pixel
        y *= mm_per_pixel
        z *= mm_per_pixel

        estimated_face_height_mm = float(np.max(y) - np.min(y))
        estimated_face_width_mm = float(np.max(x) - np.min(x))

        scale_info = {
            "left_iris_px": left_iris_px,
            "right_iris_px": right_iris_px,
            "mean_iris_px": mean_iris_px,
            "iris_diameter_mm": float(iris_diameter_mm),
            "mm_per_pixel": float(mm_per_pixel),
            "iris_mismatch": float(relative_mismatch),
            "estimated_face_height_mm": estimated_face_height_mm,
            "estimated_face_width_mm": estimated_face_width_mm,
        }

        return np.stack([x, y, z], axis=-1), scale_info

    finally:
        face_mesh_model.close()


# ---------------------------------------------------------------------------
# 2. Front-face mesh construction
# ---------------------------------------------------------------------------

def subdivide_triangle_with_continuity(
    v1,
    v2,
    v3,
    resolution,
    vertices_dict=None,
):
    """Subdivide a triangle while sharing coincident vertices."""
    if resolution < 1:
        raise ValueError("resolution must be >= 1")

    if vertices_dict is None:
        vertices_dict = {}

    def get_or_add_vertex(vertex):
        vertex_tuple = tuple(np.round(vertex, decimals=8))
        if vertex_tuple not in vertices_dict:
            vertices_dict[vertex_tuple] = len(vertices_dict)
        return vertices_dict[vertex_tuple]

    index_map = []

    for i in range(resolution + 1):
        row_indices = []

        for j in range(resolution + 1 - i):
            w1 = i / resolution
            w2 = j / resolution
            w3 = 1 - w1 - w2

            point = w1 * v1 + w2 * v2 + w3 * v3
            row_indices.append(get_or_add_vertex(point))

        index_map.append(row_indices)

    triangles = []

    for i in range(resolution):
        for j in range(resolution - i):
            p1 = index_map[i][j]
            p2 = index_map[i + 1][j]
            p3 = index_map[i][j + 1]

            triangles.append([p1, p2, p3])

            if j < resolution - i - 1:
                p4 = index_map[i + 1][j + 1]
                triangles.append([p2, p4, p3])

    return triangles, vertices_dict


def hc_laplacian_smoothing(
    vertices,
    faces,
    iterations,
    alpha=0.3,
    beta=0.5,
):
    """Apply the HC-style Laplacian smoothing from the original script."""
    vertices = np.asarray(vertices, dtype=float)
    original_vertices = vertices.copy()

    adjacency_list = {i: [] for i in range(len(vertices))}

    for face in faces:
        for i in range(3):
            adjacency_list[face[i]].extend(
                face[j] for j in range(3) if j != i
            )

    for key in adjacency_list:
        adjacency_list[key] = list(set(adjacency_list[key]))

    for _ in range(iterations):
        q = vertices.copy()

        for i, neighbors in adjacency_list.items():
            if neighbors:
                q[i] = np.mean(vertices[neighbors], axis=0)

        b = np.zeros_like(vertices)

        for i in range(len(vertices)):
            b[i] = q[i] - (
                alpha * original_vertices[i]
                + (1 - alpha) * vertices[i]
            )

        new_vertices = q.copy()

        for i, neighbors in adjacency_list.items():
            if neighbors:
                avg_b = np.mean(b[neighbors], axis=0)
                new_vertices[i] = q[i] - (
                    beta * b[i] + (1 - beta) * avg_b
                )

        vertices = new_vertices

    return vertices


def build_front_face_mesh(
    landmarks,
    resolution=3,
    iterations=10,
    alpha=0.3,
    beta=0.5,
):
    """Create the subdivided and smoothed open facial surface."""
    subdivided_vertices_dict = {}
    subdivided_faces = []

    for triangle in mesh_index:
        v1, v2, v3 = landmarks[triangle]

        local_faces, subdivided_vertices_dict = (
            subdivide_triangle_with_continuity(
                v1,
                v2,
                v3,
                resolution=resolution,
                vertices_dict=subdivided_vertices_dict,
            )
        )

        subdivided_faces.extend(local_faces)

    vertices = np.array(
        list(subdivided_vertices_dict.keys()),
        dtype=float,
    )
    faces = np.asarray(subdivided_faces, dtype=int)

    vertices = hc_laplacian_smoothing(
        vertices,
        faces,
        iterations=iterations,
        alpha=alpha,
        beta=beta,
    )

    return vertices, faces


# ---------------------------------------------------------------------------
# 3. Back-head reconstruction / watertight closure
# ---------------------------------------------------------------------------

def find_boundary_edges(faces):
    """Find edges that occur in exactly one triangle."""
    edge_count = {}

    for face in faces:
        edges = [
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ]

        for edge in edges:
            edge = tuple(sorted(edge))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    return [
        edge
        for edge, count in edge_count.items()
        if count == 1
    ]


def order_boundary_loop(boundary_edges):
    """Order facial boundary edges into one loop."""
    if not boundary_edges:
        raise ValueError("No boundary edges found.")

    adjacency = {}

    for v1, v2 in boundary_edges:
        adjacency.setdefault(v1, []).append(v2)
        adjacency.setdefault(v2, []).append(v1)

    start_vertex = boundary_edges[0][0]
    loop = [start_vertex]

    previous = None
    current = start_vertex

    while True:
        neighbors = adjacency[current]
        next_candidates = [
            vertex
            for vertex in neighbors
            if vertex != previous
        ]

        if not next_candidates:
            break

        next_vertex = next_candidates[0]

        if next_vertex == start_vertex:
            break

        loop.append(next_vertex)

        previous = current
        current = next_vertex

        if len(loop) > len(boundary_edges) + 5:
            raise RuntimeError(
                "Boundary loop ordering did not converge."
            )

    return loop


def create_back_head_mesh(
    vertices,
    faces,
    n_rings=36,
    back_depth_ratio=0.7,
    shrink_power=1.2,
    top_lift_ratio=0.4,
    crown_back_ratio=0.5,
    upper_occipital_ratio=0.13,
    upper_occipital_back_ratio=0.85,
    upper_occipital_width=0.85,
    smooth_top=True,
):
    """Close the face using the final back-head parameters from v6_2."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)

    boundary_edges = find_boundary_edges(faces)
    boundary_loop = order_boundary_loop(boundary_edges)
    boundary_pts = vertices[boundary_loop]

    center = np.mean(boundary_pts, axis=0)

    x_min = np.min(boundary_pts[:, 0])
    x_max = np.max(boundary_pts[:, 0])
    y_min = np.min(boundary_pts[:, 1])
    y_max = np.max(boundary_pts[:, 1])

    width = x_max - x_min
    height = y_max - y_min
    face_size = max(width, height)

    x_center = 0.5 * (x_min + x_max)

    # Same direction used by the original final version.
    back_dir = np.array([0.0, 0.0, 1.0])

    back_depth = face_size * back_depth_ratio
    top_lift = height * top_lift_ratio
    crown_back = back_depth * crown_back_ratio

    new_vertices = vertices.tolist()
    new_faces = faces.tolist()

    ring_indices = [boundary_loop]

    number_of_boundary_vertices = len(boundary_loop)
    back_head_start_idx = len(new_vertices)

    for ring_number in range(1, n_rings + 1):
        t = ring_number / n_rings

        scale = np.cos(t * np.pi / 2) ** shrink_power
        posterior_weight = np.sin(t * np.pi / 2)

        z_offset = back_dir * back_depth * posterior_weight

        current_ring = []

        for point in boundary_pts:
            y_norm = (
                (point[1] - y_min)
                / (height + 1e-8)
            )

            top_weight = np.clip(
                1.0 - y_norm,
                0.0,
                1.0,
            )
            top_weight = top_weight ** 0.65

            radial = point - center
            new_point = (
                center
                + radial * scale
                + z_offset
            )

            if smooth_top:
                crown_weight = np.exp(
                    -(
                        (
                            back_depth * posterior_weight
                            - crown_back
                        ) ** 2
                    )
                    / (
                        2
                        * (0.45 * back_depth) ** 2
                    )
                )

                lift = top_lift * top_weight * (
                    0.45 * posterior_weight
                    + 0.85 * crown_weight
                )

                new_point[1] -= lift

                new_point += (
                    back_dir
                    * back_depth
                    * 0.12
                    * top_weight
                    * crown_weight
                )

                x_rel = (
                    (new_point[0] - x_center)
                    / (0.5 * width + 1e-8)
                )

                center_weight = np.exp(
                    -(x_rel ** 2)
                    / (
                        2
                        * upper_occipital_width ** 2
                    )
                )

                occipital_weight = np.exp(
                    -(
                        (
                            posterior_weight
                            - upper_occipital_back_ratio
                        ) ** 2
                    )
                    / (2 * 0.18 ** 2)
                )

                upper_weight = top_weight ** 1.2

                occipital_bulge = (
                    upper_occipital_ratio
                    * height
                    * upper_weight
                    * center_weight
                    * occipital_weight
                )

                # Fill upward and backward to strengthen the upper occiput.
                new_point[1] -= occipital_bulge
                new_point += (
                    back_dir
                    * occipital_bulge
                    * 0.65
                )

            new_vertices.append(new_point.tolist())
            current_ring.append(
                len(new_vertices) - 1
            )

        ring_indices.append(current_ring)

    # Connect adjacent rings using triangle strips.
    for ring_number in range(
        len(ring_indices) - 1
    ):
        ring_a = ring_indices[ring_number]
        ring_b = ring_indices[ring_number + 1]

        for i in range(number_of_boundary_vertices):
            a0 = ring_a[i]
            a1 = ring_a[
                (i + 1)
                % number_of_boundary_vertices
            ]
            b0 = ring_b[i]
            b1 = ring_b[
                (i + 1)
                % number_of_boundary_vertices
            ]

            new_faces.append([a0, b0, a1])
            new_faces.append([a1, b0, b1])

    # Cap the final ring.
    last_ring = ring_indices[-1]
    last_points = np.array(
        [new_vertices[i] for i in last_ring]
    )

    cap_center = np.mean(last_points, axis=0)

    if smooth_top:
        cap_center[1] -= top_lift * 0.12
        cap_center += (
            back_dir
            * back_depth
            * 0.02
        )

    cap_index = len(new_vertices)
    new_vertices.append(cap_center.tolist())

    for i in range(number_of_boundary_vertices):
        v1 = last_ring[i]
        v2 = last_ring[
            (i + 1)
            % number_of_boundary_vertices
        ]

        new_faces.append(
            [v1, cap_index, v2]
        )

    new_vertices = np.asarray(
        new_vertices,
        dtype=float,
    )
    new_faces = np.asarray(
        new_faces,
        dtype=int,
    )

    # Original postprocessing:
    # smooth only the newly generated back-head region.
    back_head_indices = set(
        range(
            back_head_start_idx,
            len(new_vertices),
        )
    )

    adjacency = {
        index: set()
        for index in back_head_indices
    }

    for face in new_faces:
        for i in range(3):
            vertex_index = face[i]

            if vertex_index in back_head_indices:
                for j in range(3):
                    if (
                        i != j
                        and face[j]
                        in back_head_indices
                    ):
                        adjacency[
                            vertex_index
                        ].add(face[j])

    smooth_iterations = 5
    smooth_lambda = 0.9

    for _ in range(smooth_iterations):
        smoothed = new_vertices.copy()

        for vertex_index in back_head_indices:
            neighbors = adjacency.get(
                vertex_index,
                set(),
            )

            if not neighbors:
                continue

            average = np.mean(
                new_vertices[
                    list(neighbors)
                ],
                axis=0,
            )

            smoothed[vertex_index] = (
                (1 - smooth_lambda)
                * new_vertices[vertex_index]
                + smooth_lambda
                * average
            )

        new_vertices = smoothed

    return new_vertices, new_faces


# ---------------------------------------------------------------------------
# 4. STL output
# ---------------------------------------------------------------------------

def save_watertight_stl(
    vertices,
    faces,
    output_path,
):
    """Fix face winding, validate the closed mesh, and export STL.

    Final coordinate convention:
        +x : right
        +y : up
        +z : face/front (nose direction)

    The y/z sign flips are preserved from the original back_head_v6_2.py.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    vertices = np.asarray(
        vertices,
        dtype=float,
    ).copy()

    faces = np.asarray(
        faces,
        dtype=int,
    )

    # Original final orientation conversion:
    # image y-down -> +y up
    # MediaPipe depth direction -> +z face/front
    vertices[:, 1] *= -1
    vertices[:, 2] *= -1

    output_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )

    # The reconstructed mesh is geometrically closed, but the original
    # triangle ordering can contain inconsistent winding. Normalize it here.
    output_mesh.fix_normals()

    # Do not silently save a broken geometry.
    if not output_mesh.is_watertight:
        raise RuntimeError(
            "Generated mesh is not watertight."
        )

    if not output_mesh.is_winding_consistent:
        raise RuntimeError(
            "Generated mesh has inconsistent face winding."
        )

    if not output_mesh.is_volume:
        raise RuntimeError(
            "Generated mesh is not a valid closed volume."
        )

    output_mesh.export(str(output_path))

    print(
        "Mesh validation:"
        f" watertight={output_mesh.is_watertight},"
        f" winding_consistent={output_mesh.is_winding_consistent},"
        f" valid_volume={output_mesh.is_volume},"
        f" volume={output_mesh.volume:.6f}"
    )


# ---------------------------------------------------------------------------
# 5. Complete image -> STL pipeline
# ---------------------------------------------------------------------------

def convert_image_to_stl(
    image_path,
    output_path,
    resolution=3,
    iterations=10,
    iris_diameter_mm=DEFAULT_IRIS_DIAMETER_MM,
    max_iris_mismatch=0.25,
):
    """Convert one image directly to one metric-scaled watertight STL."""
    landmarks, scale_info = extract_iris_scaled_landmarks(
        image_path,
        iris_diameter_mm=iris_diameter_mm,
        max_iris_mismatch=max_iris_mismatch,
    )

    print(
        "Iris scale:"
        f" left={scale_info['left_iris_px']:.2f}px,"
        f" right={scale_info['right_iris_px']:.2f}px,"
        f" mean={scale_info['mean_iris_px']:.2f}px"
    )
    print(
        f"Metric scale: {scale_info['mm_per_pixel']:.6f} mm/pixel "
        f"(mean iris -> {scale_info['iris_diameter_mm']:.2f} mm)"
    )
    print(
        "Estimated face landmark size:"
        f" width={scale_info['estimated_face_width_mm']:.2f} mm,"
        f" height={scale_info['estimated_face_height_mm']:.2f} mm"
    )

    vertices, faces = build_front_face_mesh(
        landmarks,
        resolution=resolution,
        iterations=iterations,
        alpha=0.3,
        beta=0.5,
    )

    vertices, faces = create_back_head_mesh(
        vertices,
        faces,
        n_rings=36,
        back_depth_ratio=0.7,
        shrink_power=1.2,
        top_lift_ratio=0.4,
        crown_back_ratio=0.5,
        upper_occipital_ratio=0.13,
        upper_occipital_back_ratio=0.85,
        upper_occipital_width=0.85,
        smooth_top=True,
    )

    save_watertight_stl(
        vertices,
        faces,
        output_path,
    )

    return Path(output_path), scale_info


def collect_images():
    """Collect all supported images inside IMAGE_DIR."""
    if not IMAGE_DIR.exists():
        return []

    return sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_SUFFIXES
        )
    )


def resolve_input_path(input_value):
    """Resolve a filename inside IMAGE_DIR or accept an absolute path."""
    input_path = Path(input_value)

    if not input_path.is_absolute():
        input_path = (
            IMAGE_DIR
            / input_path
        )

    return input_path.resolve()


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Convert frontal face JPG/JPEG/PNG images to "
            "iris-scaled watertight STL."
        )
    )

    parser.add_argument(
        "--input",
        help=(
            "Optional single input image. "
            "A bare filename is resolved inside "
            "ai-cfd-data/01_images. "
            "If omitted, all supported images "
            "in 01_images are processed."
        ),
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=3,
        help=(
            "Triangle subdivision resolution "
            "(original default: 3)."
        ),
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help=(
            "HC Laplacian smoothing iterations "
            "(original default: 10)."
        ),
    )

    parser.add_argument(
        "--iris-diameter-mm",
        type=float,
        default=DEFAULT_IRIS_DIAMETER_MM,
        help=(
            "Reference mean iris diameter in mm "
            f"(default: {DEFAULT_IRIS_DIAMETER_MM})."
        ),
    )

    parser.add_argument(
        "--max-iris-mismatch",
        type=float,
        default=0.25,
        help=(
            "Maximum allowed relative difference between left/right "
            "detected iris diameters (default: 0.25 = 25%%)."
        ),
    )

    return parser


def main():
    args = build_parser().parse_args()

    STL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.input:
        image_paths = [
            resolve_input_path(
                args.input
            )
        ]
    else:
        image_paths = collect_images()

    if not image_paths:
        print(
            "No JPG/JPEG/PNG images found in: "
            f"{IMAGE_DIR}"
        )
        return 0

    failures = []

    for image_path in image_paths:
        if not image_path.is_file():
            print(
                "[FAILED] Input file not found: "
                f"{image_path}"
            )
            failures.append(
                image_path
            )
            continue

        if (
            image_path.suffix.lower()
            not in SUPPORTED_SUFFIXES
        ):
            print(
                "[FAILED] Unsupported image type: "
                f"{image_path}"
            )
            failures.append(
                image_path
            )
            continue

        output_path = (
            STL_DIR
            / f"{image_path.stem}.stl"
        )

        print("=" * 72)
        print(
            f"Input : {image_path}"
        )
        print(
            f"Output: {output_path}"
        )

        try:
            convert_image_to_stl(
                image_path=image_path,
                output_path=output_path,
                resolution=args.resolution,
                iterations=args.iterations,
                iris_diameter_mm=args.iris_diameter_mm,
                max_iris_mismatch=args.max_iris_mismatch,
            )

            print(
                f"[OK] Saved: {output_path}"
            )

        except Exception as exc:
            print(
                f"[FAILED] "
                f"{image_path.name}: {exc}"
            )
            failures.append(
                image_path
            )

    if failures:
        print(
            f"Finished with "
            f"{len(failures)} failure(s)."
        )
        return 1

    print(
        f"Finished successfully: "
        f"{len(image_paths)} image(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
