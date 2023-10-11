from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='burgersDA',
    version='0.1.0',
    description='Python test-bench for variational DA',
    long_description=readme,
    author='Jose Arnal',
    author_email='jose.arnal@mail.utoronto.ca',
    url='https://github.com/josearnal/burgersDA',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
