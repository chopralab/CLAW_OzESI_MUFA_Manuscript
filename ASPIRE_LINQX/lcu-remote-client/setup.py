from setuptools import setup

setup(
    name="LCUClient",
    version="0.1.0",
    description="Remote LCU client library",
    url="https://github.com/shuds13/pyexample",
    author="Patrick Grinaway",
    author_email="pgrinaway@onai.com",
    license="BSD 2-clause",
    packages=["LCUClient"],
    install_requires=[
        "protobuf",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
    ],
)
