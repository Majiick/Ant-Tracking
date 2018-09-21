from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='Ant Tracking',
    version='0.1.0',
    description='Ant path tracking in videos using OpenCV',
    long_description=readme,
    author='Zan Smirnov, Ronalds Upenieks, Evgeny Timoshin',
    url='https://github.com/Majiick/Ant-Tracking',
    license=license,
    packages=find_packages()
)