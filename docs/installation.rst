.. include:: links.rst

------------
Installation
------------
There are two ways to install *PETPrep*:

* using container technologies (RECOMMENDED); or
* within a `Manually Prepared Environment (Python 3.10+)`_, also known as
  *bare-metal installation*.

The ``petprep`` command-line adheres to the `BIDS-Apps recommendations
for the user interface <usage.html#execution-and-the-bids-format>`__.
Therefore, the command-line has the following structure::

  $ petprep <input_bids_path> <derivatives_path> <analysis_level> <named_options>

The ``petprep`` command-line options are documented in the :ref:`usage`
section.

The command as shown works for a *bare-metal* environment set-up (second option above).
If you choose the recommended container-based installation, then
the command-line will be composed of a preamble to configure the
container execution followed by the ``petprep`` command-line options
as if you were running it on a *bare-metal* installation.
The command-line structure above is then modified as follows::

  $ <container_command_and_options> <container_image> \
       <input_bids_path> <derivatives_path> <analysis_level> <petprep_named_options>

Therefore, once specified the container options and the image to be run
the command line is the same as for the *bare-metal* installation but dropping
the ``petprep`` executable name.

Containerized execution (Docker and Apptainer)
================================================
*PETPrep* is a *NiPreps* application, and therefore follows some overarching principles
of containerized execution drawn from the BIDS-Apps protocols.
For detailed information of containerized execution of *NiPreps*, please visit the corresponding
`Docker <https://www.nipreps.org/apps/docker/>`__
or `Apptainer / Singularity <https://www.nipreps.org/apps/singularity/>`__ subsections.
The *NiPreps* portal also contains
`extended details of execution with the Docker wrapper <https://www.nipreps.org/apps/docker/#running-a-niprep-with-a-lightweight-wrapper>`__.

In short, install the ``petprep-docker`` wrapper with pip::

  $ python -m pip install petprep-docker

Warning - since petprep relies on many tools (see below Dependencies), you need about 30 Gb free space (the image is 24GB -- also beware of docker settings which by default will not use the last 10% of your disk space).

Then run the ``petprep-docker`` command-line as if you were running
``petprep`` on a *bare-metal* installation::

  $ petprep-docker <input_bids_path> <derivatives_path> <analysis_level> <named_options>

Alternatively, you can install using a pre-built image using either::

  $ docker pull ghcr.io/nipreps/petprep:main

or if on linux/amd64::
 
  $ docker pull --platform=linux/amd64 ghcr.io/nipreps/petprep:main

or using apptainer::

  $ apptainer build petprep.sif docker://ghcr.io/nipreps/petprep:main

(Find latest on: https://github.com/nipreps/petprep/pkgs/container/petprep)

Then run petprep via apptainer as follows::

  $ apptainer run \
   --bind /path_to_data:/data:ro \
   --bind /path_to_output:/out \
   --bind /path_to_freesurfer/license.txt:/license \
   /path_to_sif/petprep.sif /data /out participant or group \
   --participant-label sub-ID \
   --fs-license-file /license \
   --work-dir /out/work


Manually Prepared Environment (Python 3.10+)
============================================

.. warning::

   This method is not recommended! Please consider using containers.

Make sure all of *PETPrep*'s `External Dependencies`_ are installed.
These tools must be installed and their binaries available in the
system's ``$PATH``.
A relatively interpretable description of how your environment can be set-up
is found in the `Dockerfile <https://github.com/nipreps/petprep/blob/master/Dockerfile>`_.
As an additional installation setting, FreeSurfer requires a license file (see :ref:`fs_license`).

On a functional Python 3.10 (or above) environment with ``pip`` installed,
*PETPrep* can be installed using the habitual command ::

    $ python -m pip install petprep

Check your installation with the ``--version`` argument ::

    $ petprep --version


External Dependencies
---------------------
*PETPrep* is written using Python 3.8 (or above), and is based on
nipype_.

*PETPrep* requires some other neuroimaging software tools that are
not handled by the Python's packaging system (Pypi) used to deploy
the ``petprep`` package:

- FSL_ (version 6.0.7.7)
- ANTs_ (version 2.5.1)
- AFNI_ (version 24.0.05)
- `C3D <https://sourceforge.net/projects/c3d/>`_ (version 1.4.0)
- FreeSurfer_ (version 7.4.1)
- PETPVC_ (version 1.2.10)
- `bids-validator <https://github.com/bids-standard/bids-validator>`_ (version 1.14.0)
- `connectome-workbench <https://www.humanconnectome.org/software/connectome-workbench>`_ (version 1.5.0)
- PETPVC_ (version 1.2.10)

Not running on a local machine? - Data transfer
===============================================
If you intend to run *PETPrep* on a remote system, you will need to
make your data available within that system first.

For instance, here at the Poldrack Lab we use Stanford's
:abbr:`HPC (high-performance computing)` system, called Sherlock.
Sherlock enables `the following data transfer options
<https://www.sherlock.stanford.edu/docs/user-guide/storage/data-transfer/>`_.

Alternatively, more comprehensive solutions such as `Datalad
<https://www.datalad.org/>`_ will handle data transfers with the appropriate
settings and commands.
Datalad also performs version control over your data.
