from setuptools import setup

setup(name='overgang',
      version='0.1.2',
      description='transition matrix estimation',
      url='http://github.com/ulf1/overgang',
      author='Ulf Hamster',
      author_email='554c46@gmail.com',
      license='MIT',
      packages=['overgang'],
      install_requires=['numpy', 'datetime', 'scipy'],
      python_requires='>=3',
      zip_safe=False)

