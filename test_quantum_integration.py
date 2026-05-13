#!/usr/bin/env python
"""Quick test to verify quantum_regressor is properly configured"""

from main.model_scripts import quantum_regressor
from main.model_scripts.base import validate_module

print('=' * 60)
print('QUANTUM REGRESSOR INTEGRATION TEST')
print('=' * 60)

# Check module attributes
print(f'\n✓ MODULE ATTRIBUTES:')
print(f'  MODEL_NAME: {quantum_regressor.MODEL_NAME}')
print(f'  SUPPORTED_PROBLEM_TYPES: {quantum_regressor.SUPPORTED_PROBLEM_TYPES}')
print(f'  Has Model class: {hasattr(quantum_regressor, "Model")}')
print(f'  Has train_model function: {hasattr(quantum_regressor, "train_model")}')

# Validate with the same validator used by training system
is_valid, reason = validate_module(quantum_regressor)
print(f'\n✓ VALIDATION CHECK:')
print(f'  Valid: {is_valid}')
print(f'  Reason: {reason}')

# Check Model class
if hasattr(quantum_regressor, 'Model'):
    print(f'\n✓ MODEL CLASS:')
    print(f'  Has train_model method: {hasattr(quantum_regressor.Model, "train_model")}')
    print(f'  MODEL_NAME attribute: {quantum_regressor.Model.MODEL_NAME}')
    print(f'  SUPPORTED_PROBLEM_TYPES: {quantum_regressor.Model.SUPPORTED_PROBLEM_TYPES}')

print('\n' + '=' * 60)
if is_valid:
    print('✓ READY: quantum_regressor will be auto-discovered!')
    print('When someone trains a regression model via UI/API,')
    print('quantum_regressor will automatically be included.')
else:
    print('✗ ISSUE: quantum_regressor not properly configured')
    print(f'Reason: {reason}')
print('=' * 60)
