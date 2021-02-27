import pathlib

from setuptools import find_packages, setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

INSTALL_REQUIRES = [
    "tensorflow~=2.4.1`",
]

# noinspection SpellCheckingInspection
setup(
        name="semi_supervised_clustering",
        version="0.1.0",
        description="Cluster context-less language data in a semi-supervised manner.",
        long_description=README,
        long_description_content_type="text/markdown",
        url="https://github.com/RubenPants/semi_supervised_clustering",
        author="RubenPants",
        author_email="broekxruben@gmail.com",
        license="CC BY-NC-ND 3.0",
        classifiers=["Programming Language :: Python :: 3", "Programming Language :: Python :: 3.8", ],
        packages=find_packages(exclude=("tests", "notebooks", "doc", "scripts")),
        include_package_data=True,
        package_data={"": ["data/synonym_config.pkl"]},
        install_requires=INSTALL_REQUIRES,
)
