#!/bin/bash
# Upgrade hypha-rpc
pip install --upgrade hypha-rpc

# Pass all arguments to the Python script
python register_sam_service.py "$@"
