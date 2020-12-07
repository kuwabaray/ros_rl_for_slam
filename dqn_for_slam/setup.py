from setuptools import setup


REQUIRES_PYTHON = '>=3.5.0'

setup(
    name="dqn_for_slam",
    version='0.0.1',
    description='reinforce learning for the robot',
    python_requires=REQUIRES_PYTHON,
    install_requires=[
        'gym==0.17.3',
        'matplotlib==3.3.3',
        'tensorflow==2.3.1',
        'keras-rl2==1.0.4',
        'numpy==1.18.5',
        'rospkg==1.2.9'
    ]
)
