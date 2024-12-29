#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <ssh_key_name>"
    return 1 2>/dev/null || exit 1
fi

SSH_KEY="~/.ssh/$1"
TARGET_NODE="node01"
REMOTE_WHEELS_DIR="~/wheels"
LOCAL_WHEEL_PATH="target/wheels/bpekit-0.1.0-cp310-abi3-linux_x86_64.whl"
REMOTE_WHEEL_PATH="$REMOTE_WHEELS_DIR/bpekit-0.1.0-cp310-abi3-linux_x86_64.whl"
PYTHON_ENV_PATH="~/envs/bpekit/bin/activate"

ssh -i "$SSH_KEY" "$TARGET_NODE" "mkdir -p $REMOTE_WHEELS_DIR"
scp -i "$SSH_KEY" "$LOCAL_WHEEL_PATH" "$TARGET_NODE:$REMOTE_WHEELS_DIR/"
srun bash -c "source $PYTHON_ENV_PATH && pip install --force-reinstall --no-deps $REMOTE_WHEEL_PATH"
