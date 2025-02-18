import base64

from io import BytesIO
from PIL import Image, ImageDraw

def b64image_to_pilimage(b64image):
    image_data = base64.b64decode(b64image)
    
    image_buffer = BytesIO(image_data)
    
    pilimage = Image.open(image_buffer)
    
    return pilimage

def pilimage_to_b64image(pilimage):
    image_buffer = BytesIO()
    pilimage.save(image_buffer, format = 'PNG')
    b64image = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
    
    return b64image

class BBoxDrawer():
    def __init__(self):
        pass
    
    def draw_on_b64image(self, b64image, xywhs, labels, colors, line_width = 2):
        if len(xywhs) == 0:
            return b64image
        
        pilimage = b64image_to_pilimage(b64image)
        
        for xywh, label, color in zip(xywhs, labels, colors):
            x, y, w, h = xywh
            
            draw = ImageDraw.Draw(pilimage)
            
            draw.rectangle(xy = (x, y, x + w, y + h), outline = color, width = line_width)
                        
        drawn_b64image = pilimage_to_b64image(pilimage)
        
        return drawn_b64image
