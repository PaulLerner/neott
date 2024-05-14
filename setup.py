import os

import pkg_resources
from setuptools import setup

setup(
    name='neott',
    packages=['neott'],
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ]
)
