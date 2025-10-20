from setuptools import setup

setup(
    name="jhutil",
    version="0.1.10",
    description="jhutil",
    long_description_content_type="text/markdown",
    url="https://github.com/jk4011/jhutil",
    author="Jinhyeok Kim",
    author_email="jinhyuk@unist.ac.kr",
    install_requires=[
        "requests",
        "numpy",
        "openai",
        "colorama",
        "lovely_tensors==0.1.18",
        "easydict",
        "notion_client",
        "slack_sdk",
        "pillow",
        "trimesh",
        "plotly",
        "PyYAML",
        "pytorch_memlab",
        "coloredlogs",
        "pythreejs",
        "einops",
        "pyntcloud",
        "Pillow",  # If visualization not shown in ipynb, use Pillow==9.5.0
        "opencv-python",
    ],
    packages=["jhutil"],
    python_requires=">=3.6, <4",
    project_urls={
        "Source": "https://github.com/jk4011/jhutil",
    },
)
