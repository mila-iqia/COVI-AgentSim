Welcome to the Mila-IQIA Covid-19 simulator documentation!
==========================================================
This is a sub-project of `Peer-to-Peer AI Tracing App <https://mila.quebec/en/peer-to-peer-ai-tracing-of-covid-19/>`_
delegated by `Prof. Yoshua Bengio <https://yoshuabengio.org/>`_.
Read more about the app in Prof. Bengio's `blog post <https://yoshuabengio.org/2020/03/23/peer-to-peer-ai-tracing-of-covid-19/>`_.

The simulator is built using `simpy <https://simpy.readthedocs.io/en/latest/simpy_intro/index.html>`_.
It simulates human mobility along with SARS-CoV-2 spread in a city, where the city has houses, grocery stores, parks, workplaces, and other non-essential establishments.
The infection spread in this simulator is modeled according to what is known about SARS-CoV-2 and COVID-19.
Our understanding is based on the published research as well as interactions with the epidemiologists.
We describe this in several papers (TODO: link to the COVI white paper, ML paper, and simulator paper).


.. toctree::
   :maxdepth: 1
   :caption: Notes

   notes/installation
   notes/CONTRIBUTING
   notes/hydra

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