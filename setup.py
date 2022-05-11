from setuptools import setup, find_packages

path = "requirements.txt"

with open(path) as fp:
    requirements = fp.read().splitlines()

if __name__ == "__main__":
    setup(
        name="BayesianLogitRegression",
        version="0.0.1",
        description="Fit a Bayesian logit regression model to your data",
        url="#",
        author="Shreyas Rao",
        license='MIT',
        install_requires=requirements,
        author_email="shreyas@brown.edu",
        packages=find_packages(),
        include_package_data=True,
        python_requires=">=3.8",
        zip_safe=False,
    )
