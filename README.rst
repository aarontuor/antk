=========================================
Welcome to Automated Neural Graph Toolkit
=========================================

|aaron| |docs| |pypi| 

Purpose
-------

Automated Neural Graph Toolkit is an extension library for Google's Tensorflow. It is designed to facilitate rapid prototyping of Neural Network models which may consist of multiple models chained together. Multiple input streams and and or multiple output predictions are well supported.

Documentation for ANTk
----------------------

You will find complete documentation for ANTk at `the ANTk readthedocs page`_.

.. _the ANTk readthedocs page: http://antk.readthedocs.io/en/latest/


.. |aaron| image:: docs/_static/snakelogo.png
    :alt: Aaron's web page
    :scale: 100%
    :target: https://sw.cs.wwu.edu/~tuora/aarontuor/index.html

.. |docs| image:: docs/_static/docs.gif
    :alt: Documentation
    :scale: 100%
    :target: http://antk.readthedocs.io/en/latest
    
.. |pypi| image:: docs/_static/pypi_page.gif
    :alt: pypi page
    :scale: 100%
    :target: https://pypi.python.org/pypi/antk/
    
.. _Install tensorflow: https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html
.. _Install graphviz: http://www.graphviz.org/

Platform
--------

ANTk is compatible with linux 64 bit operating systems.

Python Distribution
-------------------

ANTk is written in python 2. Most functionality should be forwards compatible.

Install
-------

A virtual environment is recommended for installation. Make sure that tensorflow is installed in your virtual environment
and graphviz is installed on your system.

`Install tensorflow`_

`Install graphviz`_

From the terminal:

.. code-block:: 

    (venv)$ pip install antk


