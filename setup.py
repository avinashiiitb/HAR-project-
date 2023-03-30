
from setuptools import find_packages,setup
from typing import List

HYPEN_E = '-e .'

def get_requirements(path:str) ->List:
    """
    This will return the list of requirements that needs to be installed
    """
    requirements =[]
    with open(path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n"," ") for req in requirements]
    if HYPEN_E in requirements:
        requirements.remove(HYPEN_E)
    return requirements



setup(name='HAR Project',
      version='0.0.1',
      description='HAR Project',
      author='Avinash Dixit',
      author_email='avinash.dixit@iiitb.ac.in',
      
      packages =find_packages(),
      install_requires = get_requirements('requirements.txt'),
     )