import os
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
from PIL import Image

import torch, gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

gc.collect()
torch.cuda.empty_cache()

# Load the model
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda", non_blocking=True)


def resize_image(image, max_width, max_height):
    # 현재 이미지 크기
    width, height = image.size
    # 비율 계산
    aspect_ratio = width / height

    if width > height:
        new_width = min(max_width, width)
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = min(max_height, height)
        new_width = int(new_height * aspect_ratio)

    return image.resize((new_width, new_height), Image.LANCZOS)

def load_image_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Define the directory containing images
image_dir = "./level2-cv-datacentric-cv-04/data/thai_receipt/img/train"

# Load images from the directory
image_files = load_image_files(image_dir)

# Process images in batches of 5 for training
num_epochs = 10  # Set to 5 epochs
prompt = "a realistic receipt"

# Process images for training (5 epochs)
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        init_image = load_image(image_path).convert("RGB")

        # 영수증 비율에 맞춰 리사이즈 (최대 크기 설정)
        init_image = resize_image(init_image, 512, 512)

        # 그래디언트 계산 활성화
        # init_image를 PIL.Image 형식으로 확인
        if isinstance(init_image, Image.Image):
            output = pipe(prompt, image=init_image)  # 학습을 위한 호출
        else:
            print("Error: The input image is not in the correct format.")

        del init_image  # 사용 후 메모리 해제
        torch.cuda.empty_cache()  # CUDA 캐시 비우기

#Define the directory containing images
image_dir = "/data/ephemeral/home/nayoung/Noisy/Thai" 

# Load images from the directory
image_files = load_image_files(image_dir)

# Process images for training (5 epochs)
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    init_image = load_image(image_path).convert("RGB")
    # init_image = resize_image(init_image, 1024, 1024)

    # 그래디언트 계산 활성화
    output = pipe(prompt, image=init_image)  # 학습을 위한 호출
    # 생성된 이미지를 저장
    if isinstance(output.images, list) and len(output.images) > 0:
        output.images[0].resize((720, 720), Image.Resampling.LANCZOS).save(f"./denoised/denoised_{image_file}")  # 각 이미지에 대해 저장
    else:
        print(f"Error: The output image for {image_file} is not in the correct format.")

    del init_image  # 사용 후 메모리 해제
    torch.cuda.empty_cache()  # CUDA 캐시 비우기

print("All images processed and generated successfully.")