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
        'attrs==23.1.0',
        'argparse~=1.4.0',
        'wheel==0.40.0',
        'packages~=0.1.0',
        'google~=3.0.0',
        'cryptography==41.0.0',
        'Pillow==9.5.0',
        'docutils~=0.20.1',
        'Jinja2~=3.1.2',
        'keyring~=23.13.1',
        'setuptools==67.8.0',
        'packaging~=23.0',
        'pyparsing~=3.0.9',
        'zipp~=3.15.0',
        'nltk~=3.8.1',
        'numpy==1.24.3',
        'tensorflow~=2.12.0rc1',
        'tflearn~=0.5.0',
        'torch~=2.0.0'
    ],
    license='Apache Software License (Apache 2.0)',  # License metadata
    # Tags metadata
    project_urls={
        'Tags': 'tensorflow, tensor, machine, learning',
    },
)

print("Happy Coding!")
