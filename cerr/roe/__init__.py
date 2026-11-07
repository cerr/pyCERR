def launch(*args, **kwargs):
    """Launch the ROE GUI (requires the ``viewer`` extra).

    Lazy wrapper so that headless installs can import ``cerr.roe`` and use
    ``cerr.roe.dosimetric_models`` without Qt being present.

    Args:
        args (tuple): Positional arguments for cerr.roe.roe_gui.launch.
        kwargs (dict): Keyword arguments for cerr.roe.roe_gui.launch.

    Returns:
        The value returned by cerr.roe.roe_gui.launch.
    """
    from cerr.roe.roe_gui import launch as _launch
    return _launch(*args, **kwargs)
