Installation Guide
==================

Note: this installation process has been tested on
 * windows / amd64 platform 

To install CodPy, you can use pip:

.. code-block:: bash

    pip install codpy

Alternatively, if you are installing from source:

.. code-block:: bash

    git clone https://github.com/codpy2020/codpy
    cd codpy
    pip install .

Requirements
~~~~~~~~~~~~~~~~~

Make sure to have these installed before proceeding.

* `python3.9.XX <https://www.python.org/ftp/python/3.9.7/python-3.9.7-amd64.exe>`_: a valid python python 3.9 installation.

*NOTE* : Python installations differ from one machine to another. The python root folder is denoted "\<path/to/python39>" in the rest of this document. The software `Everything <https://www.voidtools.com/>`_, or other finding files tools can be useful locating the file python.exe on windows machine...

Dev installation
~~~~~~~~~~~~~~~~~

For information, we list the softwares that we are using for our dev configuration :

* `GitHub Desktop <https://desktop.github.com>`_
* `R<https://www.r-project.org>`_: use any CRAN mirror for download
* `RStudio <https://rstudio.com>`_: see the download link, then choose the free version
* `MiKTEX <https://miktex.org>`_: see the download tab
* `Everything <https://www.voidtools.com/downloads/>`_
* `Visual Studio Code <https://code.visualstudio.com>`_

Those installations should be fine using the latest (64 bits) version and the default settings for each software .

*Note* Once R and RStudio are installed, open the latter.
In the console, enter "*install.packages("rmarkdown")*" to install `RMarkdown <https://rmarkdown.rstudio.com/index.html>`_.

Installation procedure
~~~~~~~~~~~~~~~~~

We suppose that there is a valid python installation on the host machine. The reader can 
* either use its main python environment ``<path/to/python39>``
* or create a virtual python environment ``<path/to/venv>``, generally an advisable practice:

First open a command shell ``cmd``,  create a virtual environment and activate it using the commands

.. code-block:: bash

    python -m venv .\venv
    .\venv\Scripts\activate

*NOTE* : In the rest of the installation procedure, we consider a virtual environment <path/to/venv>. One can replace with <path/to/python39> if a main environment installation is desired, for dev purposes for instance.


Open a command shell ```cmd```, and pip install codpy

.. code-block:: bash

    pip install codpy==0.XX.XX

or from a local repository

.. code-block:: bash

    pip install <path/to/codpyrepo>/dist/codpy-XXXX.whl

The installation procedure might take some minutes depending on your internet connection.

Basic checking
~~~~~~~~~~~~~~~~~

open a python shell and import codpy

.. code-block:: bash

    python
    import codpy

It should run silently.

