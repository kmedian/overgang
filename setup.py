from setuptools import setup


def read(fname):
    import os
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='overgang',
      version='0.3.0',
      description='(deprecated package) Markov Modeling',
      long_description=read('README.md'),
      long_description_content_type='text/markdown',
      url='http://github.com/kmedian/overgang',
      author='Ulf Hamster',
      author_email='554c46@gmail.com',
      license='MIT',
      packages=['overgang'],
      install_requires=['numpy', 'datetime', 'scipy'],
      python_requires='>=3',
      zip_safe=False)

