# File: slr/utils/display.py

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from .translation import translate_to_hindi

class HindiDisplay:
    def __init__(self):
        try:
            # Try different fonts that support Hindi
            fonts = [
                "arial.ttf",
                "C:/Windows/Fonts/mangal.ttf",
                "C:/Windows/Fonts/Nirmala.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
            for font_path in fonts:
                if os.path.exists(font_path):
                    self.font = ImageFont.truetype(font_path, 32)
                    break
        except:
            self.font = ImageFont.load_default()

    def draw_text(self, image, text, position):
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        draw.text(position, text, font=self.font, fill=(255, 255, 255))
        return np.array(pil_image)

def show_result(result_image, handedness, hand_sign_text):
    """Show recognized gesture and its Hindi translation"""
    hindi_display = HindiDisplay()
    
    if hand_sign_text:
        # Draw English text
        result_image = hindi_display.draw_text(
            result_image,
            f"Sign: {hand_sign_text}",
            (10, 30)
        )
        # Draw Hindi translation
        hindi_text = translate_to_hindi(hand_sign_text)
        result_image = hindi_display.draw_text(
            result_image,
            f"हिंदी: {hindi_text}",
            (10, 70)
        )
    
    return result_image