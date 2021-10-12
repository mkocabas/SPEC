#!/usr/bin/env bash
set -e

echo "Creating virtual environment"
python3.7 -m venv spec-env
echo "Activating virtual environment"

source $PWD/spec-env/bin/activate

$PWD/spec-env/bin/pip install -r requirements.txt