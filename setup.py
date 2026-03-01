from setuptools import setup, find_packages

setup(
    name="data_meta_map",
    version="0.1.0",
    author="...",
    description="...",
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    url="https://github.com/intsystems/DataMetaMap",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
    ],
)
