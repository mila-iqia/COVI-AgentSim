Welcome to Covi-Canada simulator documentation!
=======================================================
This is a sub-project of `XXX`_
delegated by `XXX`_.
Read more about the app in XXX.

The simulator is built using `simpy <https://simpy.readthedocs.io/en/latest/simpy_intro/index.html>`_.
It simulates human mobility along with infectious disease (COVID) spreading in a city, where city has houses, grocery stores, parks, workplaces, and other non-essential establishments.
Human mobility simulation is based on Spatial-EPR model.
More details on this model are `here <https://www.nature.com/articles/ncomms9166>`_ and `here <https://www.nature.com/articles/nphys1760>`_.

The infection spread in this simulator is modeled according to what is known about COVID-19.
The assumptions about the COVID-19 spread and mobility implemented in the simulator are in the `Google Doc`_.
The same document also details our current understanding of COVID-19.
Our understanding is based on the published research as well as interactions with the epidemiologists.
We plan to update the simulator as more and more about COVID-19 will be known.

.. toctree::
   :maxdepth: 1
   :caption: Notes

   notes/installation
   notes/CONTRIBUTING
   notes/hydra
   notes/events
   notes/models

.. toctree::
   :maxdepth: 1
   :caption: Python API

   base
   constants
   datastructures
   interventions
   monitors
   simulator
   track
   utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`