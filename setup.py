from setuptools import find_packages, setup


setup(
    name="asr",
    version="1.0",
    author="khaykingleb",
    package_dir={"": "asr"},
    packages=find_packages("asr"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
