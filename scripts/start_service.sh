#!/bin/bash
# Upgrade hypha-rpc
sudo pip install --upgrade hypha-rpc --root-user-action=ignore

# Pass all arguments to the Python script
python register_sam_service.py "$@"
