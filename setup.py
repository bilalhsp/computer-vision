from setuptools import setup, find_packages


setup(
    name="computer_vision",
    version="0.1.0",
    packages=find_packages(),
    author="Bilal Ahmed",
    author_email="bilalhsp@gmail.com",
    url="https://github.com/bilalhsp/computer-vision",
    description="Repo for computer vision course.",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # install_requires=[
    #     'torch', 'datasets', 'transformers', 'yaml'
    # ],
    python_required='>=3.7',
)