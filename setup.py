import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='MoViNet-pytorch',
    version='0.2.2',
    author='Lorenzo Atzeni',
    author_email='loryatze@gmail.com',
    description='MoViNet pytorch implementation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Atze00/MoViNet-pytorch',
    license='MIT',
    packages=setuptools.find_packages(exclude=['tests']),
    install_requires=['torch',
                      'fvcore',
                      'einops'
                     ],
)
