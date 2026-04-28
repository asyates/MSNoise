.. include:: ../configs.hrst

Stack
-----

.. automodule:: msnoise.s04_stack_mov

.. _stacking_methods:

Stacking Methods
~~~~~~~~~~~~~~~~

MSNoise supports three stacking methods, selected via the ``stack_method``
configuration key (valid for both the moving stack and the reference stack).
All three reduce an array of :math:`N` daily CCF traces
:math:`\{d_j(t)\}_{j=1}^{N}` to a single representative trace.

Linear stack
^^^^^^^^^^^^

The arithmetic mean:

.. math::

   s(t) = \frac{1}{N} \sum_{j=1}^{N} d_j(t)

Incoherent noise cancels at a rate of :math:`1/\sqrt{N}`.  This is the
fastest and most transparent option.  It provides no suppression of
high-amplitude transients (earthquakes, instrument glitches) that survive
pre-processing.

Set ``stack_method = linear`` in the stack or refstack configset.

Phase-weighted stack (``pws``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Introduced by Schimmel & Paulssen (1997) [Schimmel1997]_.  Each sample is weighted by the
instantaneous *phase coherence* :math:`c(t)` of the analytic signal across
all traces:

.. math::

   c(t) = \frac{1}{N} \left|
       \sum_{j=1}^{N} e^{i\,\phi_j(t)}
   \right|, \qquad c(t) \in [0,\,1]

where :math:`\phi_j(t) = \arg\!\bigl(d_j(t) + i\,\mathcal{H}\{d_j\}(t)\bigr)`
is the instantaneous phase of trace :math:`j` obtained via the Hilbert
transform :math:`\mathcal{H}`.  The coherence is smoothed with a boxcar
window of ``pws_timegate`` seconds and then raised to the power
:math:`v` = ``pws_power``:

.. math::

   s(t) = \frac{1}{N} \sum_{j=1}^{N} d_j(t) \cdot c(t)^v

Incoherent transients produce random instantaneous phases across traces, so
:math:`c(t) \approx 0` at those times; truly coherent arrivals yield
:math:`c(t) \approx 1` and are preserved.

Practical notes:

- ``pws_timegate`` (default 10 s) controls the width of the boxcar
  smoothing applied to :math:`c(t)` before the power is taken.  Shorter
  gates resolve rapid phase changes; longer gates produce a smoother weight
  and are more robust with few traces.
- ``pws_power`` (default 2): higher values increase selectivity at the cost
  of amplitude fidelity on weak coherent arrivals.

Time-frequency phase-weighted stack (``tfpws``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The TF extension of PWS introduced by Schimmel & Gallart (2007) [Schimmel2007]_.  Phase
coherence is evaluated independently at each scale of a continuous wavelet
transform (CWT), yielding a time–frequency coherence map
:math:`c(a, t)` that is both scale- and time-dependent.

A complex Morlet wavelet is used:

.. math::

   \psi_{s}(t) =
       \frac{1}{\pi^{1/4}\sqrt{s}}\,
       e^{i\omega_0 t/s}\,
       e^{-t^2/(2s^2)}, \qquad \omega_0 = 5

Scales :math:`\{s_k\}` are log-spaced between ``freqmin`` and ``freqmax``
(inherited from the parent filter configset), with the scale–frequency
relationship :math:`s_k = \omega_0 f_s / (2\pi f_k)`.

The phase coherence at each (scale, lag) point is:

.. math::

   c(s_k, t) = \frac{1}{N} \left|
       \sum_{j=1}^{N} e^{i\,\arg\mathcal{W}_j(s_k,\,t)}
   \right|

where :math:`\mathcal{W}_j(s_k, t)` is the CWT coefficient of trace
:math:`j`.  Averaging over the :math:`A` = ``tfpws_nscales`` scales and
raising to the power :math:`v` yields the per-lag weight:

.. math::

   w(t) = \left[
       \frac{1}{A} \sum_{k=1}^{A} c(s_k,\,t)
   \right]^{v}

The final stack is:

.. math::

   s(t) = \frac{1}{N} \sum_{j=1}^{N} d_j(t) \cdot w(t)

Because coherence is assessed independently at each frequency, tf-PWS is
more sensitive to narrow-band coherent arrivals than time-domain PWS.
Romero & Schimmel (2018) [Romero2018]_ demonstrate this advantage for noise
autocorrelations targeting shallow P-wave reflections (3–18 Hz), where
amplitude-based transients from Pyrenean seismicity would otherwise corrupt
the CCGN result.

Practical notes:

- ``tfpws_nscales`` (default 20): more scales improve frequency resolution
  but increase memory and compute time proportionally.
  Memory scales as :math:`O(N \times A \times T)` complex128.
- ``pws_power`` is shared with the ``pws`` method (default 2).
- ``freqmin`` / ``freqmax`` are taken automatically from the parent filter
  configset — no extra configuration is needed.

Comparison
^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Method
     - Amplitude sensitivity
     - Frequency selectivity
     - Compute cost
     - Best for
   * - ``linear``
     - High
     - None
     - Lowest
     - Dense arrays, clean data
   * - ``pws``
     - Low
     - None (time-domain)
     - Low
     - General use, impulsive noise
   * - ``tfpws``
     - Low
     - High
     - Moderate
     - Narrow-band targets, autocorrelations



.. seealso::

   **Reading these results in Python** — use :class:`MSNoiseResult <msnoise.results.MSNoiseResult>`:

   .. code-block:: python

      from msnoise.results import MSNoiseResult
      from msnoise.core.db import connect
      db = connect()
      r = MSNoiseResult.from_ids(db, ...)  # include the steps you need
      # then call r.get_ccf(...) or r.get_ref(...)

   See :ref:`msnoise_result` for the full guide and all available methods.


   See :ref:`msnoise_result` for the full guide and all available methods.
