.. LCUClient documentation master file, created by
   sphinx-quickstart on Wed Jun 28 12:52:53 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LCUClient's documentation!
=====================================
Currently, the LCU remote client can register resources. To do this, simply follow the following steps::

    from LCUClient important client
    c = client.LCUClient()
    id = "210203-32"
    description = "A new resource"
    response = c.register(id, description)

``response`` will be an object of type ``LCUResponseMessage`` and is indicative of successful registration.

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
