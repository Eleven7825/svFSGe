# FSGe: A fast and strongly-coupled 3D fluid-solid-growth interaction method

## Reference
[arXiv:2404.14187](https://arxiv.org/abs/2404.14187)

## Quickstart
1. Install docker and pull the svMultiphyiscs image from 
```bash
docker pull simvascular/solver:latest
```
Make two directories, one for svMultiPhysics(svfsi), one for svFSGe:
```bash
mkdir svFSGe && mkdir svMultiPhysics
```

Start a new svMultiphysics container by:

```bash
docker run -it --user $(id -u):$(id -g) \
           -v ./svMultiPhysics:/svfsi \
           -v ./svFSGe:/svFSGe        \
              simvascular/solver:latest /bin/bash
```
2. Inside the container, install `svFSIplus` from [this branch](https://github.com/Eleven7825/svMultiPhysics/tree/FSGe), to build svfsi, you need to run(inside docker, command line begins with #):
```bash
cd /svfsi
git init
git remote add origin https://github.com/Eleven7825/svMultiPhysics.git
git fetch origin FSGe
git checkout -b FSGe origin/FSGe
```

At the /svfsi directory, build it with
```bash
bash makeCommand.sh
```

3. Stil inside the container, you need to adapt paths in `/svFSGe/in_sim/partitioned_full.json` to make sure they are correct
4. Run
    ```bash
    ./fsg.py in_sim/partitioned_full.json
    ```
You need install neccessary python packages. 

## File overview

- `cylinder.py` generates structured FSI hex-meshes with configuration files in `in_geo`
- `fsg.py` runs partiotioned FSGe coupling using svFSIplus with
  - `in_sim` FSGe configuration files
  - `in_svfsi_plus` svFSIplus input files
  - `in_petsc` PETSc linear solver settings
- `post.py` generate line plots from FSGe results
- `svfsi.py` sets up, executes, and processes svFSIplus simulations
- `utilities.py` IQN-ILS filtering
- `vtk_functions.py` useful VTK functions for file IO
- `scripts` more or less useful scripts
