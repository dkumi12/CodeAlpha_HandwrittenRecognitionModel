import requests
import io
from PIL import Image, ImageDraw

# 1. Create a dummy image (White background with a Black 'A')
# This simulates a handwritten character
img = Image.new('RGB', (224, 224), color='white')
d = ImageDraw.Draw(img)
# Draw a simple cross/line to simulate writing
d.line((50, 50, 150, 150), fill='black', width=5)

# Save to memory buffer
buf = io.BytesIO()
img.save(buf, format='PNG')
buf.seek(0)

# 2. Send to API
url = "http://localhost:8000/predict"
files = {"file": ("test_image.png", buf, "image/png")}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, files=files)

    # 3. Print Result
    if response.status_code == 200:
        print("\n✅ SUCCESS!")
        print("Response:", response.json())
    else:
        print(f"\n❌ FAILED (Status {response.status_code})")
        print("Detail:", response.text)

except Exception as e:
    print(f"\n❌ Connection Error: {e}")
