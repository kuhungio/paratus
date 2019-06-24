from setuptools import setup, find_packages

setup(name='paratus',
      version='0.1',
      description='Ready to use Keras models',
      url='https://github.com/jaume-ferrarons/paratus',
      author='Jaume Ferrarons',
      author_email='',
      packages=find_packages(),
      install_requires=[
          'matplotlib>=3.0.3',
          'numpy>=1.9.1',
          'pandas>=0.24.2',
          'tensorflow>=1.12.0'
      ],
      extras_require={
          'tests': ['pytest',
                    'pytest-cov'],
      },
      zip_safe=False)
