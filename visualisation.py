import typing
from utils1 import *
import cv2

def show_image_with_signs(image_to_show, signs: typing.List[Sign], colors):
    for i, sign in enumerate(signs):
        # Draw the bounding box on the image
        cv2.rectangle(image_to_show, (int(sign.topLeftX), int(sign.topLeftY)), (int(sign.bottomRightX), int(sign.bottomRightY)), (0, 255, 0), 2)

        label = f'{sign.name} - {colors[i]}'
        
        # Choose a font and get the text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        
        # Position the text at the top-left corner of the bounding box
        text_x = int(sign.topLeftX) - 1
        text_y = int(sign.topLeftY)
        
        # Draw the text background rectangle
        cv2.rectangle(image_to_show, (text_x, text_y - text_size[1] - 2), (text_x + text_size[0], text_y), (0, 255, 0), cv2.FILLED)
        
        # Put the text on the image
        cv2.putText(image_to_show, label, (text_x, text_y - 2), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    cv2.imshow("Image", image_to_show)
    cv2.waitKey(0)