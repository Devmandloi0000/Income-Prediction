from setuptools import find_packages,setup
from typing import List

hypon_e_dot = "-e ."

def get_requirements(file_name):
    requirements = []
    with open(file_name) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        if hypon_e_dot in requirements:
            requirements.remove(hypon_e_dot)
    return requirements


setup (
    name = "LOGISTIC_REGRESSION",
    author ="Dev_mandloi",
    author_email= "devmandloi37@gmail.com",
    version = "0.0.0.1",
    install_requires = get_requirements("requirements.txt"),
    packages = find_packages()
    
)