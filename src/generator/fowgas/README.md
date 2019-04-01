# fowgas

FOils With Gratings And Scattering



## Installation on Windows:

To run the Jupyter notebooks in this repo, the package/module paths need to be added to the `PYTHONPATH` environment variable.
Notebooks in `fowgas\notebooks` (and subdirectories) need `fowgas\packages` to be added to `PYTHONPATH`.
Notebooks in `fowgas\notebooks\basinhopping_recos\surface_grating\` need `fowgas\notebooks\basinhopping_recos\surface_grating` to be added to `PYTHONPATH`.
Notebooks in `fowgas\loose_notebooks` need `fowgas\loose_modules` to be added to `PYTHONPATH`.

`PYTHONPATH` can be created/altered via "Benutzerkonten">"Eigene Umgebungsvariablen aendern".

### If no environment variable `PYTHONPATH` yet exists
*(Replace all instances of* `path_to_local_repo_copy` *with the actual path to the local copy of this repo on your machine.)*
* Add an environment variable `PYTHONPATH`.
* To have `fowgas\abc` in your `PYTHONPATH`, make `path_to_local_repo_copy\fowgas\abc` the content of `PYTHONPATH`.
* To have `fowgas\abc` and `fowgas\def` in your `PYTHONPATH`, make `path_to_local_repo_copy\fowgas\abc;path_to_local_repo_copy\fowgas\def` the content of `PYTHONPATH`.

### If the environment variable `PYTHONPATH` exists
*(Replace all instances of* `path_to_local_repo_copy` *with the actual path to the local copy of this repo on your machine.)*
* To have `fowgas\abc` in your `PYTHONPATH`, append `;path_to_local_repo_copy\fowgas\abc` to the former content of `PYTHONPATH`.

To view the content of `PYTHONPATH`, open a new command window (`cmd`) and type `echo %PYTHONPATH%`.

After properly setting `PYTHONPATH` notebooks can be executed as usual.
