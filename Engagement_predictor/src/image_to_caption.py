from transformers import pipeline # type: ignore

captioner = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base"
)

def generate_caption(image_path):
    result = captioner(image_path)
    return result[0]['generated_text']