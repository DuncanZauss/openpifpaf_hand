from setuptools import setup, find_packages

setup(
    name='openpifpaf_hand',
    packages= ['openpifpaf_hand'],
    license='MIT',
    version = '0.1.0',
    description='OpenPifPaf RHD and FreiHand PlugIn',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Duncan Zauss',
    url='https://github.com/DuncanZauss/openpifpaf_hand',

    install_requires=[
        'matplotlib',
        'openpifpaf>=0.12b1',
    ],
)
