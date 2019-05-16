from setuptools import setup

try:
    # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:
    # for pip <= 9.0.3
    from pip.req import parse_requirements

def load_requirements(fname):
    reqs = parse_requirements(fname, session="test")
    return [str(ir.req) for ir in reqs]

setup(
    name='circuit',
    version='0.0.2',
    packages=['circuit'],
    license='LICENSE.txt',
    description='circuit model',
    long_description=open('README.md').read(),
    install_requires=load_requirements("requirements.txt"),
)
