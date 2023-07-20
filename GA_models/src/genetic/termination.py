#!/usr/bin/env python

# This module decides when the genetic algorithm should stop. We only use a
# maximum number of generations for now.

from GA_models.src import config


def shouldTerminate(population, gen, duration):
    
    exit = False
    if gen > config.maxGen:
        exit=True
    elif duration > config.maxDuration:
        exit = True
    
    return exit
