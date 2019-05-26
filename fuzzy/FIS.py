"""
    The code has been corrected from Fuzzy Q-Learning implementation (Author named Seyed Saeid Masoumzadeh)
    Link: https://github.com/seyedsaeidmasoumzadeh/Fuzzy-Q-Learning
"""
import numpy as np
import itertools
import operator
import functools
import random
import globalvars

np.random.seed(globalvars.GLOBAL_SEED)
random.seed(globalvars.GLOBAL_SEED)


class InputStateVariable(object):
    def __init__(self, *args):
        self.fuzzy_set_list = args

    def get_fuzzy_sets(self):
        return self.fuzzy_set_list


class Trapeziums(object):
    def __init__(self, left, left_top, right_top, right):
        self.left = left
        self.right = right
        self.left_top = left_top
        self.right_top = right_top

    def membership_value(self, input_value):
        if (input_value >= self.left_top) and (input_value <= self.right_top):
            membership_value = 1.0
        elif (input_value <= self.left) or (input_value >= self.right):
            membership_value = 0.0
        elif input_value < self.left_top:
            membership_value = (input_value - self.left) / (self.left_top - self.left)
        elif input_value > self.right_top:
            membership_value = (input_value - self.right) / (self.right_top - self.right)
        else:
            membership_value = 0.0
        return membership_value


class Rules(object):
    def __init__(self, *args):
        self.list_of_input_variable = args

    def get_input(self):
        return self.list_of_input_variable

    def get_number_of_rules(self):
        number_of_rules = 1
        for input_variable in self.list_of_input_variable:
            number_of_rules = (number_of_rules * self.get_number_of_fuzzy_sets(input_variable))
        return number_of_rules

    def get_number_of_fuzzy_sets(self, input_variable):
        return len(input_variable.get_fuzzy_sets())


class FIS(object):
    def __init__(self, **kwargs):
        if 'Rules' in kwargs:
            self.rules = kwargs['Rules']
        else:
            raise KeyError('No Rules found')

    def truth_values(self, state_value):
        truth_values = []
        L = []
        input_variables = self.rules.list_of_input_variable
        for index, variable in enumerate(input_variables):
            m_values = []
            fuzzy_sets = variable.get_fuzzy_sets()
            for fuzzy_set in fuzzy_sets:
                membership_value = fuzzy_set.membership_value(state_value[index])
                m_values.append(membership_value)
            L.append(m_values)
        for element in itertools.product(*L):
            truth_values.append(functools.reduce(operator.mul, element, 1))
        return truth_values
