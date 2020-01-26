from distutils.core import setup

setup(
    name='cln-core',
    version='0.1dev',
    packages=['cln'],
    # packages=['cln','cln.interval', 'cln.neurosmt', 'cln.smtlib_parser'],
    long_description=open('README.md').read(),
    install_requires=[]
)
