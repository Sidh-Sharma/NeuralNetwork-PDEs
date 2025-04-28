from setuptools import setup, find_packages

setup(
    name="fokker_planck_nn",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # add your dependencies here, e.g.
        "torch",
        "numpy",
        "matplotlib",
        "pyyaml",
        "tqdm"
    ],
    python_requires=">=3.8",
)
