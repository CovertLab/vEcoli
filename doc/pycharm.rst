=============
PyCharm Setup
=============

-----------
Interpreter
-----------

To tell PyCharm to use the uv environment in the cloned repository:

#. Follow `the PyCharm documentation <https://www.jetbrains.com/help/pycharm/uv.html>`_
   to configure an existing uv environment. 

#. Click the folder icon next to ``Uv env use`` to select the Python interpreter located
   at ``.venv/bin/python`` inside the cloned repository. On macOS, press ``Cmd + Shift + .``
   to show hidden files (filenames that start with ``.``) in the file dialog.

---------------------
Environment Variables
---------------------

To tell PyCharm to use the environment variables defined in the ``.env`` file
in the cloned repository:

#. Follow `these additional steps <https://www.jetbrains.com/help/pycharm/run-debug-configuration.html#change-template>`_
   to edit the configuration templates for ``Python``, ``Python tests > Autodetect``,
   and ``Python tests > pytest``.

#. For each of the above configuration templates, click the folder icon
   next to ``Paths to .env files`` and select the ``.env`` file in the cloned repository.
   On macOS, press ``Cmd + Shift + .`` to show hidden files (filenames that start with
   ``.``) in the file dialog.
