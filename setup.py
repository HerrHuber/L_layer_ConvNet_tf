from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = '1.0.1'

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

setup(
    name='L_layer_ConvNet_tf',
    version=__version__,
    description="L layer convolutional neural network in tensorflow 1",
    long_description=long_description,
    url='https://github.com/HerrHuber/L_layer_ConvNet_tf',
    download_url='https://github.com/HerrHuber/L_layer_ConvNet_tf/tarball/' + __version__,
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='deep neural network model',
    packages=find_packages(exclude=['docs', 'tests*', 'datasets', 'images']),
    include_package_data=True,
    author='Benedikt Huber',
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email='asignal.business@gmail.com',
    entry_points={
        'console_scripts': [
            'LLayerConvNetTF = L_layer_ConvNet_tf.L_layer_ConvNet_tf:main'
        ]},

)
