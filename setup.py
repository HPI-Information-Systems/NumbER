from setuptools import setup

if __name__ == "__main__":
    setup(
        name="NumbER",
        version="0.0.1",
        description="Entity Resolution on Numerical Data",
        author="Lukas Laskowski",
        author_email="lukas.laskowski@student.hpi.de",
        url="https://github.com/HPI-Information-Systems/NumbER",
        license="MIT",
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
        ],
        # packages=find_packages(exclude=("tests", "tests.*")),
        packages=["NumbER"],
        package_data={"NumbER": ["py.typed"]},
        # install_requires=load_dependencies(),
        python_requires=">=3.7, <3.10",
        # test_suite="tests",
        # cmdclass={
        #     "test": PyTestCommand,
        #     "typecheck": MyPyCheckCommand,
        #     "clean": CleanCommand,
        # },
        zip_safe=False,
    )