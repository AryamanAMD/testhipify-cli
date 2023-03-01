import setuptools

with open("README.md", "r", encoding="utf-8") as fhand:
    long_description = fhand.read()

setuptools.setup(
    name="testhipify",
    version="0.0.1",
    author="Aryaman Mishra",
    author_email="aryamanatamd@gmail.com",
    description=("Convert CUDA Samples "
                "to HIP Samples,compilation and execution is automated."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AryamanAMD/testhipify",
    project_urls={
        "Bug Tracker": "https://github.com/AryamanAMD/testhipify/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "testhipify = runner.cli:main",
        ]
    }
) 