import base64

from io import BytesIO
from PIL import Image

def b64image_to_pilimage(b64image):
    image_data = base64.b64decode(b64image)
    
    image_buffer = BytesIO(image_data)
    
    pilimage = Image.open(image_buffer, format = 'PNG')
    
    return pilimage
