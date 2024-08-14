from utils1 import *
import typing
import numpy as np
import torch
from torchvision.ops import nms, box_iou
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def evaluate_with_results(actual, predicted):
    correct_class = 0
    wrong_class = 0
    not_found = 0
    too_many = 0
    IoUs = []
    wrong_detailed = {}
    for i in range(44):
        wrong_detailed[class_map[i]] = 0

    for i in range(len(actual)):
        actual_signs = actual[i]
        predicted_signs = predicted[i]
        result = evaluate_single(predicted_signs, actual_signs)
        correct_class += result[0]
        wrong_class += result[1]
        not_found += result[2]
        too_many += result[3]
        IoUs.extend(result[4])

        for key in result[5]:
            wrong_detailed[key] += result[5][key]
        
    print(f"Correct class: {correct_class}")
    print(f"Wrong class: {wrong_class}")
    print(f"Not found: {not_found}")
    print(f"Too many: {too_many}")
    avg_IoU = (sum(IoUs) / len(IoUs)).item() if len(IoUs) > 0 else 0.0
    print(f"Mean IoU: {avg_IoU}")
    print(f"Detailed results:")
    for key in sorted(wrong_detailed, key=wrong_detailed.get, reverse=True):
        if wrong_detailed[key] > 0:
            print(f"\t{key}: {wrong_detailed[key]}")


def evaluate(model, test_loader):
    model.eval()
    correct_class = 0
    wrong_class = 0
    not_found = 0
    too_many = 0
    IoUs = []
    for data in tqdm(test_loader):
        for i in range(4):
            image = data[0][i].to(device)
            outputs = model([image])
            boxes = outputs[0]["boxes"].detach().cpu()
            scores = outputs[0]["scores"].detach().cpu()
            labels = outputs[0]["labels"].detach().cpu()

            # Apply non-maximum suppression
            to_keep = nms(boxes, scores, 0.2)
            boxes = boxes[to_keep]
            scores = scores[to_keep]

            signs = []
            for box, label in zip(boxes, labels):
                sign = Sign(box[2], box[3], box[0], box[1], label.item())
                signs.append(sign)
            
            actual_boxes = data[1][i]["boxes"]
            actual_labels = data[1][i]["labels"]
            actual_signs = []
            for box, label in zip(actual_boxes, actual_labels):
                sign = Sign(box[2], box[3], box[0], box[1], label.item())
                actual_signs.append(sign)
            result = evaluate_single(signs, actual_signs)
            correct_class += result[0]
            wrong_class += result[1]
            not_found += result[2]
            too_many += result[3]
            IoUs.extend(result[4])
        
    print(f"Correct class: {correct_class}")
    print(f"Wrong class: {wrong_class}")
    print(f"Not found: {not_found}")
    print(f"Too many: {too_many}")
    avg_IoU = (sum(IoUs) / len(IoUs)).item() if len(IoUs) > 0 else 0.0
    print(f"Mean IoU: {avg_IoU}")

    return correct_class, wrong_class, not_found, too_many, avg_IoU

def evaluate_single(predicted_signs: typing.List[Sign], actual_signs: typing.List[Sign]):
    correct_class = 0
    wrong_class = 0
    wrong_detailed = {}
    not_found = 0
    too_many = 0
    IoUs = []
    handled_actual = []
    handled_predicted = []
    for actual_sign in actual_signs:
        if actual_sign.name not in wrong_detailed:
            wrong_detailed[actual_sign.name] = 0
        for predicted_sign in predicted_signs:
            iou = box_iou(torch.tensor([sign_to_box(actual_sign)]), torch.tensor([sign_to_box(predicted_sign)]))
            if iou > 0.7:
                IoUs.append(iou)
                if actual_sign.name == predicted_sign.name:
                    correct_class += 1
                else:
                    wrong_detailed[actual_sign.name] += 1
                    wrong_class += 1
                    if actual_sign.name == "stop":
                        print(predicted_sign.name)
                handled_actual.append(actual_sign)
                handled_predicted.append(predicted_sign)

    not_found = len(actual_signs) - len(handled_actual)
    too_many = len(predicted_signs) - len(handled_predicted)
    return correct_class, wrong_class, not_found, too_many, IoUs, wrong_detailed

def sign_to_box(sign: Sign):
    return (sign.topLeftX, sign.topLeftY, sign.bottomRightX, sign.bottomRightY)