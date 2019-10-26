from setuptools import setup

setup(
    name="img_gist_feature",
    version="1.0",
    author="Kalafinaian",
    author_email="Kalafinaian@outlook.com",
    keywords="image gist feature",
    packages=["gist", "feature"],
    description="image gist feature",
    install_requires=[
        "numpy",
        "opencv-python",
    ],
)