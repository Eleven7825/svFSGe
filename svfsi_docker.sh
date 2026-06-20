#!/bin/bash
# Run svFSI inside the fsg-dev container where PETSc libraries are available.
# Translates the host working directory (/home/shiyi/svFSGe/...) to the
# container mount point (/svFSGe/...).
CWD=$(pwd | sed 's|/home/shiyi/svFSGe|/svFSGe|')
docker exec -w "$CWD" fsg-dev /svfsi/svFSI-build/bin/svFSI "$@"
