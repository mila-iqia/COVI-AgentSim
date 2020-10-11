Welcome to the Mila-IQIA Covid-19 simulator documentation!
==========================================================
This is a sub-project of `Peer-to-Peer AI Tracing App <https://mila.quebec/en/peer-to-peer-ai-tracing-of-covid-19/>`_
delegated by `Prof. Yoshua Bengio <https://yoshuabengio.org/>`_.
Read more about the app in Prof. Bengio's `blog post <https://yoshuabengio.org/2020/03/23/peer-to-peer-ai-tracing-of-covid-19/>`_.

The simulator is built using `simpy <https://simpy.readthedocs.io/en/latest/simpy_intro/index.html>`_.
It simulates human mobility along with infectious disease (COVID) spreading in a city, where city has houses, grocery stores, parks, workplaces, and other non-essential establishments.
Human mobility simulation is based on Spatial-EPR model.
More details on this model are `here <https://www.nature.com/articles/ncomms9166>`_ and `here <https://www.nature.com/articles/nphys1760>`_.

The infection spread in this simulator is modeled according to what is known about COVID-19.
The assumptions about the COVID-19 spread and mobility implemented in the simulator are in the `Google Doc <https://docs.google.com/document/d/1jn8dOXgmVRX62Ux-jBSuReayATrzrd5XZS2LJuQ2hLs/edit?usp=sharing>`_.
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

.. toctree::
   :caption: Python API
   :maxdepth: 1

   run
   human
   utils
   locations
   interventions
   inference
   epidemiology



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`