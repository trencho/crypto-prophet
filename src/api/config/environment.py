from os import environ

from definitions import environment_variables

collections = ['summary', 'pollution', 'weather']


def check_environment_variables():
    for environment_variable in environment_variables:
        env = environ.get(environment_variable)
        if env is None:
            print(f'The environment variable "{environment_variable}" is missing')
            exit(-1)
