#!/bin/bash

# set training flag default false
TRAIN=false
TEST=false
EVALUATE=false

# set train test eval based on arg given
while [[ $# -gt 0 ]]; do
  case "$1" in
    --train) TRAIN=true ;;
    --test) TEST=true ;;
    --evaluate) EVALUATE=true ;;
    *) echo "Unknown parameter: $1" >&2; exit 1;;
  esac
  shift
done

# echo params if true for debugging
if [ "$TRAIN" = true ]; then
    echo "Training enabled"
fi

if [ "$TEST" = true ]; then
  echo "Test enabled"
fi

if [ "$EVALUATE" = true ]; then
    echo "Evaluate enabled"
fi

# Train part
if [ "$TRAIN" = true ]; then
    echo "Running trainer"
    python -m src.trainer
fi
# Test part
if [ "$TEST" = true ]; then
  echo "Running test"
  python -m src.evaluation
fi

# Evaluate part
if [ "$EVALUATE" = true ]; then
    echo "Running evaluation"
    python -m src.evaluation
fi
