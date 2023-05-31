from setuptools import setup, find_packages

VERSION = '0.0.0'
DESCRIPTION = 'NeuroLink is a Python package in active development that aims to provide various AI capabilities, ' \
              'including a chatbot, image processing, audio processing, Django & Flask support, and more.'

with open("README.md", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()

# Setting up
setup(
    name="neurolink",
    version=VERSION,
    author="Muhammad Fiaz",
    author_email="",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url='https://github.com/muhammad-fiaz/neurolink',
    packages=find_packages(),
    keywords=['neural networks', 'neurolink', 'machine learning', 'artificial intelligence', 'ai', 'ml', 'chatbot',
              'image processing', 'audio processing', 'django', 'flask'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires='>=3.8',
    install_requires=[
        'pip==23.1.2',
        'attrs',
        'argparse',
        'wheel',
        'packages',
        'google',
        'cryptography',
        'Pillow',
        'docutils',
        'Jinja2',
        'keyring',
        'setuptools',
        'packaging',
        'pyparsing',
        'zipp',
        'nltk',
        'numpy',
        'tensorflow',
        'tflearn',
        'torch'
    ],
    license='Apache Software License (Apache 2.0)',  # License metadata
    # Tags metadata
    project_urls={
        'Tags': 'tensorflow, tensor, machine, learning',
    },
)

print("Happy Coding!")
