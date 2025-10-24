from PIL import Image
from io import BytesIO

with open("abc.jpg", "rb") as f:
    data = f.read()

# Find the start of the JPEG (FF D8) and end (FF D9)
start = data.find(b'\xff\xd8')
end = data.find(b'\xff\xd9')

if start != -1 and end != -1:
    jpeg_bytes = data[start:end+2]
    img = Image.open(BytesIO(jpeg_bytes))
    img.show()
    img.save("fixed_xyz.jpg")
    print("✅ Extracted and saved as fixed_xyz.jpg")
else:
    print("❌ JPEG data not found in file")