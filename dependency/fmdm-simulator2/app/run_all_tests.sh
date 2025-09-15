#!/bin/bash

# Runs tests that ensure all basic functionality and all functionality related to the Franka Emika robot is working correctly
pytest --capture=fd -m 'base_functionality' tests/
pytest --capture=fd -m 'franka_v0' tests/
pytest --capture=fd -m 'franka_v1' tests/

echo "Finished running tests."
