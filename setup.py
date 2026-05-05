from setuptools import setup, find_packages

setup(
    name="coco-text2image-ldm",
    version="0.1.0",
    description="Latent Diffusion Model for text-to-image generation trained on COCO",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "Pillow>=9.5.0",
        "tqdm>=4.65.0",
        "omegaconf>=2.3.0",
        "einops>=0.6.1",
        "transformers>=4.30.0",
        "pycocotools>=2.0.6",
        "pytorch-lightning>=2.0.0",
        "torchmetrics>=0.11.4",
        "kornia>=0.6.12",
        "scipy>=1.10.0",
    ],
)
