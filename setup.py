from setuptools import setup, find_packages

setup(
    name='keyword_extractor',
    version='0.1.0',
    description='Extracts keywords from a list of texts using embeddings and noun chunks.',
    author='Sinan Parmar',
    author_email='sinanparmar@uchicago.edu',
    #url='https://github.com/yourusername/your-package-name',
    packages=find_packages(),
    package_data={'': ['__init__.py']},
    install_requires=[
        'numpy==1.26.1',
        'scikit-learn==1.3.2',
        'spacy==3.7.2',
        'sentence-transformers==2.2.2',
    ],
)