from setuptools import find_packages, setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mymanus",
    version="0.1.0",
    author="mannaandpoem and MyManus Team",
    author_email="jason2003zzz-ai@users.noreply.github.com",
    description="A versatile agent that can solve various tasks using multiple tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jason2003zzz-ai/MyManus",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.13.4,<3.0.0",
        "openai>=1.58.1,<1.67.0",
        "tenacity~=9.0.0",
        "pyyaml~=6.0.2",
        "loguru~=0.7.3",
        "numpy",
        "fastembed>=0.7.4,<1.0.0",
        "datasets>=3.2,<3.5",
        "fastapi>=0.139.0,<1.0.0",
        "python-multipart>=0.0.20,<1.0.0",
        "html2text~=2024.2.26",
        "gymnasium>=1.0,<1.2",
        "pillow>=10.4,<11.2",
        "uvicorn>=0.49.0,<1.0.0",
        "unidiff~=0.7.5",
        "googlesearch-python~=1.3.0",
        "aiofiles~=24.1.0",
        "pydantic_core>=2.46.4,<3.0.0",
        "mcp>=1.28.1,<2.0.0",
        "colorama~=0.4.6",
        "python-docx~=1.2.0",
        "openpyxl~=3.1.5",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    entry_points={
        "console_scripts": [
            "openmanus=main:main",
            "openmanus-web=app.web.server:run_web",
            "mymanus=main:main",
            "mymanus-web=app.web.server:run_web",
        ],
    },
)
