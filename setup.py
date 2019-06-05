from setuptools import setup

setup(name='paratus',
      version='0.1',
      description='Ready to use Keras models',
      url='https://github.com/jaume-ferrarons/paratus',
      author='Jaume Ferrarons',
      author_email='',
      packages=['paratus'],
      install_requires=[
          'matplotlib>=3.0.3',
          'numpy>=1.9.1',
          'tensorflow>=1.12.0'
      ],
      zip_safe=False)
