Installation instructions
***************************

qdiv can be installed with pip or conda.

.. code-block:: console

   pip install qdiv

or

.. code-block:: console

   conda activate qdiv_env


The recommended method is to download `Anaconda <https://www.anaconda.com/products/individual>`_ or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_

Open the Anaconda or Miniconda prompt (a terminal window).

Create a new environment. You can, for example, call it qdiv_env. Then, activate the environment.

.. code-block:: console

   conda create -n qdiv_env python=3.9
   conda activate qdiv_env

Install qdiv using:

.. code-block:: console

   conda install -c omvatten qdiv

To start using qdiv, you need some way of writing and executing Python code. I use `Spyder <https://www.spyder-ide.org/>`_ or Jupyter notebooks. You can install Spyder like this:

.. code-block:: console

   conda install spyder

To run Spyder, simply type:

.. code-block:: console

   spyder

You can install Jupyter like this:

.. code-block:: console

   conda install jupyter

To start a Jupyter notebook, simply type:

.. code-block:: console

   jupyter notebook

To check if qdiv works, you can run the following code:

.. code-block:: python

   import qdiv
   import pandas as pd
   df = pd.DataFrame({'a':[5,5,2,1]})
   print(qdiv.diversity.naive_alpha(df))
   
Hopefully, this will run without error messages and print a value of 3.388