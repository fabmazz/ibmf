from setuptools import setup, find_packages


setup(
    name="stat-meanfield",
    version="0.1",
    author="https://github.com/sphinxteam, Fabio Mazza",
    packages=find_packages(),
    description="Statistical mean field algorithms",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "numba",
    ]
)
