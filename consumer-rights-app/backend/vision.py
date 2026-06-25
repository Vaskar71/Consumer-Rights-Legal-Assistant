import os
import base64
import io
from groq import Groq
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

VISION_PROMPT = """You are analyzing an evidence image submitted as part of a consumer rights dispute in Bangladesh.

Look at this image carefully and extract ALL relevant information you can see, including:
- Order details (order ID, date, product name, price, quantity)
- Product condition (damaged, defective, incorrect item, missing parts)
- Seller or platform name
- Any communication between buyer and seller (chat messages, emails)
- Delivery information (courier, tracking, delivery date)
- Refund or return status
- Any dates, amounts, or reference numbers visible
- Any other detail that could be relevant to a consumer complaint

Be thorough and specific. Do not summarize — extract every detail visible in the image.
If the image is unclear or unreadable, say so explicitly."""

def describe_image(uploaded_file) -> str:
    """Send an image to Llama 3.2 Vision on Groq and return a detailed text description."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Read and encode image as base64
    image_bytes = uploaded_file.read()
    uploaded_file.seek(0)  # Reset file pointer after reading

    # Resize if too large (Groq vision has size limits)
    image = Image.open(io.BytesIO(image_bytes))
    if max(image.size) > 1568:
        image.thumbnail((1568, 1568), Image.LANCZOS)
        buffer = io.BytesIO()
        fmt = image.format if image.format else "JPEG"
        image.save(buffer, format=fmt)
        image_bytes = buffer.getvalue()

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Detect media type
    filename = uploaded_file.name.lower()
    if filename.endswith(".png"):
        media_type = "image/png"
    elif filename.endswith((".jpg", ".jpeg")):
        media_type = "image/jpeg"
    elif filename.endswith(".webp"):
        media_type = "image/webp"
    else:
        media_type = "image/jpeg"  # fallback

    try:
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": VISION_PROMPT
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"[Image could not be analyzed: {str(e)}]"
