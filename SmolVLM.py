import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import cv2

# Set up device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device is:", DEVICE)

# Load the model and processor
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.bfloat16,
).to(DEVICE)

# Function to process webcam frame and generate output
def describe_image(image):
    # Convert input to PIL Image
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Prepare the prompt
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the scene."}]}],
        add_generation_prompt=True,
    )
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(DEVICE)

    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            early_stopping=True, 
            max_new_tokens=200, 
            no_repeat_ngram_size=3,
        )
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

    # Shorten the response gracefully
    full_response = generated_texts[0]
    sentences = full_response.split('. ')
    shortened_response = '. '.join(sentences[:2])  # Keep the first 2 sentences
    if not shortened_response.endswith('.'):
        shortened_response += '.'

    return shortened_response

print("Opening Webcam")

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Generate description
    description = describe_image(frame)

    # Display the description on the frame
    cv2.putText(frame, description, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Webcam Feed with Description', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()