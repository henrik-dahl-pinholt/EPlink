from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="EPlink",
    version="0.1.0",
    author="Henrik Dahl Pinholt",
    author_email="pinholt@mit.edu",
    description="Library for linking live enhancer-promoter distance measurements to live transcription readouts through model inference.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/henrik-dahl-pinholt/EPlink",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        # List your package dependencies here
    ],
)
