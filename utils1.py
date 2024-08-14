from dataclasses import dataclass
import math

class_map = {
    0: "none",
    1: "20",
    2: "30",
    3: "50",
    4: "60",
    5: "70",
    6: "80",
    7: "80 end",
    8: "100",
    9: "120",
    10: "no overtaking",
    11: "no overtaking for trucks",
    12: "priority at next intersection",
    13: "priority road",
    14: "give way",
    15: "stop",
    16: "no traffic both ways",
    17: "no trucks",
    18: "no entry",
    19: "danger",
    20: "bend left",
    21: "bend right",
    22: "bend",
    23: "uneven road",
    24: "slippery road",
    25: "road narrows",
    26: "construction",
    27: "traffic signal",
    28: "pedestrian crossing",
    29: "school crossing",
    30: "cycles crossing",
    31: "snow",
    32: "animals",
    33: "restriction ends",
    34: "right",
    35: "left",
    36: "straight",
    37: "straight or right",
    38: "straight or left",
    39: "keep right",
    40: "keep left",
    41: "roundabout",
    42: "overtaking end",
    43: "overtaking trucks end"
}

color_map = {
    "red": [1,2,3,4,5,6,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
    "yellow": [13],
    "blue": [34, 35, 36, 37, 38, 39, 40, 41],
    None: [7, 33, 42,43],
}

@dataclass
class Sign:
    bottomRightX: float
    bottomRightY: float
    topLeftX: float
    topLeftY: float
    name: int

def collate_fn(batch):
    return tuple(zip(*batch))

def extract_signs(image, boxes):
    images = []
    for box in boxes:
        x1, y1, x2, y2 = math.floor(box[0].item()), math.floor(box[1].item()), math.ceil(box[2].item()), math.ceil(box[3].item())
        cropped_image = image[y1:y2, x1:x2]
        images.append(cropped_image)
    return images