"""Mock Sense HAT module for non-Raspberry Pi systems."""
import random
from typing import Dict, Any, List, Optional, Tuple

class SenseHat:
    def __init__(self):
        self.rotation = 0
        self.pixels = [[(0, 0, 0) for _ in range(8)] for _ in range(8)]
        
    def set_rotation(self, rotation: int, redraw: bool = True):
        self.rotation = rotation
        
    def set_pixels(self, pixel_list: List[Tuple[int, int, int]]):
        for i in range(min(64, len(pixel_list))):
            x, y = i % 8, i // 8
            self.pixels[y][x] = pixel_list[i]
    
    def get_temperature(self) -> float:
        return round(random.uniform(20.0, 30.0), 1)
    
    def get_humidity(self) -> float:
        return round(random.uniform(30.0, 70.0), 1)
    
    def get_pressure(self) -> float:
        return round(random.uniform(900.0, 1100.0), 1)
    
    def get_orientation_degrees(self) -> Dict[str, float]:
        return {
            'pitch': round(random.uniform(0, 360), 1),
            'roll': round(random.uniform(0, 360), 1),
            'yaw': round(random.uniform(0, 360), 1)
        }
    
    def show_message(self, text_string: str, scroll_speed: float = 0.1, text_colour=(255, 255, 255), back_colour=(0, 0, 0)):
        print(f"[SENSE HAT MOCK] Showing message: {text_string}")

# Create a singleton instance
sense = SenseHat()
