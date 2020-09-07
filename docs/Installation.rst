Installation instructions
***************************
To install qdiv into a conda environment:

.. code-block:: guess

   conda install -c omvatten qdiv

To install using pip:

.. code-block:: guess

   pip install qdiv

Stepwise instructions
######################

This is how I have installed qdiv:

Download `Anaconda <https://www.anaconda.com/products/individual>`_ or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_

Open the Anaconda prompt (a terminal window).

I use `Spyder <https://www.spyder-ide.org/>`_ to write and execute Python code. If you downloaded Anaconda, Spyder is probably already installed on your system. 
If you downloaded Miniconda, you can install Spyder by the following command:

.. code-block:: guess

   conda install spyder

Now, there are several options for installing qdiv. The first choice is:  

.. code-block:: guess

   conda install -c omvatten qdiv

However, I have had problems with this in Python 3.8, and have not yet figured out why it doesn't work. Then, the next option is:

.. code-block:: guess

   pip install qdiv

Now you can open Spyder from the terminal window by typing:

.. code-block:: guess

   spyder

To make sure qdiv works, you can run the following code:

.. code-block:: python

   import qdiv
   import pandas as pd
   df = pd.DataFrame({'a':[5,5,2,1]})
   print(qdiv.diversity.naive_alpha(df))

Hopefully, this will run without error messages and print a value of 3.388

A third option is to copy the file qdiv_env_win.yml (if you are on a Windows computer) or qdiv_env_linux.yml (if you are on a Linux computer) from the `github pages <https://github.com/omvatten/qdiv>`_.
In the Anaconda prompt (terminal window) go to the folder containing the file (use cd path-to-folder). Then write one of the following:

.. code-block:: guess

   conda env create -f qdiv_env_win.yml
   conda env create -f qdiv_env_linux.yml

This will create a conda environment with qdiv and some other useful packages. Activate the environment by typing:

.. code-block:: guess

   conda activate qdiv_env

Then you can run Spyder from that environment. 

.. code-block:: guess

   spyder
