from setuptools import setup

setup(
    name="jhutil",
    version="0.0.0",
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
    ],
    packages=["jhutil"],
    python_requires=">=3.7, <4",
    project_urls={
        "Source": "https://github.com/jk4011/jhutil",
    },
)