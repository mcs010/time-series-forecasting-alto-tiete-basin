"""
Author: Matheus Cardoso da Silva

This file contains custom modules
"""

# Functions to calculate dissolved oxygen saturation (uses air temperature)

def saturated_DO_concentration(temperature, altitude):
  """
  First step to calculate saturated DO
  Receives the temperature and altitude of the water body
  Returns the concentration of DO saturation
  """
  return (14.62-0.3898*temperature+0.006969*pow(temperature, 2)-0.00005896*pow(temperature, 3))*pow(1-0.0000228675*altitude, 5.167)


def DO_percentage(DO, temperature, altitude):
  """
  Receives DO and the concentration of DO saturation
  Return percentage of DO saturation
  """
  return (100*DO)/saturated_DO_concentration(temperature, altitude)