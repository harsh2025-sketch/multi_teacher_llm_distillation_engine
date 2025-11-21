#!/usr/bin/env python3
"""
Student architecture module - exports classes for automated distillation engine.
This is an alias/wrapper module that imports from model.py for compatibility.
"""

from model import StudentConfig, StudentModel

# Create aliases for the automated distillation engine
StudentArchitectureConfig = StudentConfig
StudentLLM = StudentModel

__all__ = ['StudentArchitectureConfig', 'StudentLLM', 'StudentConfig', 'StudentModel']
