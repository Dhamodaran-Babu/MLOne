import setuptools

setuptools.setup(
    name="MLOne", # Replace with your own username
    version="0.0.51",
    author="Suhas and Dhamodaran",
    author_email="srisuhas2000@gmail.com",
    description="A Python package with which users can just drop their dataset and download the best ML model for their dataset",
    long_description=open('README.md',errors='ignore').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Dhamodaran-Babu/ML-Thunai",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"],
    install_requires=[
       "pandas",
       "numpy",
       "matplotlib",
       "imblearn",
       "sklearn",
       "mlxtend >= 0.17.3",
       "seaborn",
       "joblib"
    ]
   
)