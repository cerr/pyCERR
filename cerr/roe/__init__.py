def launch(*args, **kwargs):
    """Launch the ROE GUI (requires the ``viewer`` extra).

    Lazy wrapper so that headless installs can import ``cerr.roe`` and use
    ``cerr.roe.dosimetric_models`` without Qt being present.
    """
    from cerr.roe.roe_gui import launch as _launch
    return _launch(*args, **kwargs)
