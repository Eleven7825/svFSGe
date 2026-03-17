#!/bin/bash

# Script to copy latest partitioned_* folder with selective VTU/BIN file handling
# Configuration
SOURCE_BASE="${SOURCE_BASE:-bouchet:~/svFSGe}"  # Remote source on bouchet (sc3544)
DEST_DIR="${DEST_DIR:-.}"  # Current directory by default
KEEP_LAST_N_PULSATILE=400  # Keep last N pulsatile VTU files (ignore .bin files)
KEEP_LAST_N_STEADY=10      # Keep last N steady VTU files (ignore .bin files)

set -e  # Exit on error

# SSH ControlMaster settings - authenticate once, reuse connection
SSH_HOST="${SOURCE_BASE%%:*}"
CONTROL_PATH="/tmp/ssh_control_${SSH_HOST}_%h_%p_%r"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=$CONTROL_PATH -o ControlPersist=600"

echo "=== Partitioned Folder Copy Script ==="
echo "Source: $SOURCE_BASE"
echo "Destination: $DEST_DIR"
echo ""

# Establish master connection (one-time authentication)
if [[ $SOURCE_BASE == *:* ]]; then
    echo "Establishing SSH connection (you'll authenticate once here)..."
    ssh $SSH_OPTS -t "$SSH_HOST" "echo 'Connected'" > /dev/null 2>&1
    echo ""
fi

# Find the latest partitioned_* directory
if [[ $SOURCE_BASE == *:* ]]; then
    # Remote source - get just the folder name
    REMOTE_HOST="${SOURCE_BASE%%:*}"
    FOLDER_NAME=$(ssh $SSH_OPTS "$REMOTE_HOST" "ls -td ~/svFSGe/partitioned_* 2>/dev/null | head -1 | xargs basename" 2>/dev/null | tr -d '\r')
    LATEST_DIR="$REMOTE_HOST:~/svFSGe/$FOLDER_NAME"
else
    # Local source
    LATEST_DIR=$(ls -td "$SOURCE_BASE"/partitioned_* 2>/dev/null | head -1)
    FOLDER_NAME=$(basename "$LATEST_DIR")
fi

if [ -z "$FOLDER_NAME" ]; then
    echo "Error: No partitioned_* directory found in $SOURCE_BASE"
    exit 1
fi
echo "Found latest folder: $FOLDER_NAME"

# Create destination directory
mkdir -p "$DEST_DIR/$FOLDER_NAME"

# Step 1: Copy everything EXCEPT .vtu and .bin files
echo "Step 1: Copying folder structure and other files (excluding VTU/BIN)..."
if [[ $SOURCE_BASE == *:* ]]; then
    rsync -av -e "ssh $SSH_OPTS" \
        --exclude='pulsatile/*.vtu' \
        --exclude='pulsatile/*.bin' \
        --exclude='steady/*.vtu' \
        --exclude='steady/*.bin' \
        --exclude='gr_restart/*.vtu' \
        "$LATEST_DIR/" "$DEST_DIR/$FOLDER_NAME/"
else
    rsync -av \
        --exclude='pulsatile/*.vtu' \
        --exclude='pulsatile/*.bin' \
        --exclude='steady/*.vtu' \
        --exclude='steady/*.bin' \
        --exclude='gr_restart/*.vtu' \
        "$LATEST_DIR/" "$DEST_DIR/$FOLDER_NAME/"
fi

# Step 2: Copy only last N VTU files to avoid copying huge old files
echo "Step 2: Copying VTU files (ignoring .bin files)..."
for dir in pulsatile steady; do
    # Determine how many files to keep for this directory
    if [ "$dir" = "pulsatile" ]; then
        KEEP_LAST_N=$KEEP_LAST_N_PULSATILE
    else
        KEEP_LAST_N=$KEEP_LAST_N_STEADY
    fi

    echo "  Processing $dir (keeping last $KEEP_LAST_N timesteps)..."

    if [[ $SOURCE_BASE == *:* ]]; then
        # Remote: copy only last N VTU files
        mkdir -p "$DEST_DIR/$FOLDER_NAME/$dir"

        ssh $SSH_OPTS "${SOURCE_BASE%%:*}" "ls -1v ~/$dir/*_*.vtu 2>/dev/null | sort -V | tail -$KEEP_LAST_N" | tr -d '\r' | while read file; do
            if [ -n "$file" ]; then
                rsync -av -e "ssh $SSH_OPTS" "$SOURCE_BASE/$dir/$file" "$DEST_DIR/$FOLDER_NAME/$dir/"
            fi
        done
    else
        # Local: copy last N VTU files
        mkdir -p "$DEST_DIR/$FOLDER_NAME/$dir"

        ls -1v "$LATEST_DIR/$dir"/*_*.vtu 2>/dev/null | sort -V | tail -$KEEP_LAST_N | while read file; do
            cp "$file" "$DEST_DIR/$FOLDER_NAME/$dir/"
        done
    fi
done

echo ""
echo "=== Copy completed successfully! ==="
echo "Folder location: $DEST_DIR/$FOLDER_NAME"

# Clean up SSH master connection
if [[ $SOURCE_BASE == *:* ]]; then
    ssh $SSH_OPTS -O exit "$SSH_HOST" 2>/dev/null || true
fi
