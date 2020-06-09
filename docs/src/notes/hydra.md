# COVI Simulator's configuration

COVI's Simulator uses [Hydra](https://hydra.cc/) to manage configurations and easily run simulations.

It is highly encouraged that you spend 15 minutes reading through the [tutorial](https://hydra.cc/docs/tutorial/simple_cli)

This document will walk you through COVI-specific Hydra usage.

## How to Use

### Main file

Hydra uses YAML configuration files. They **must** all be in `root/src/covid19sim/hydra-configs/simulation`. YAML's main supported types are ~ `int`, `float`, `string`, `list`, `dictionary`. Anything more complicated could trigger parsing errors. Learn more in the *Advanced* section.

```
------------------------- NOTE ------------------------------------
| YAML doesn't have a `tuple` type, and `list`s cannot be used as |
| keys for dictionary                                             |
-------------------------------------------------------------------
```

***First***, Hydra will first read `hydra-configs/simulation/config.yaml` and then load sequentially all the files listed in the `defaults:` values. Each of them will **overwrite the previous ones if they share keys**. This allows for a hierarchy of configurations.

For instance:

```yaml
---------------------------------
# config.yaml

defaults:
  - constants
  - core
  - base_method

---------------------------------
# constants.yaml

VARIABLE_A: 0
VARIABLE_B: 1
VARIABLE_C: 2

---------------------------------
# core.yaml
VARIABLE_A: 3
VARIABLE_B: 100
VARIABLE_D: 4

---------------------------------
# base_method.yaml
VARIABLE_A: 20
VARIABLE_E: 30

---------------------------------
```
Will be parsed into a Python object with the **union** of all keys, as:

```python
conf = {
    "VARIABLE_A": 20,
    "VARIABLE_B": 100,
    "VARIABLE_C": 2,
    "VARIABLE_D": 4,
    "VARIABLE_E": 30,
}
```

### Named configurations

***Second***, Hydra can access specific configurations specified from the command line and stored in directories **within** `hydra-configs/simulation`. The name of such sub-folders will describe how to load them

```yaml
# tracing-method/binary_tracing.yaml
VARIABLE_F: "some string"
VARIABLE_C: 12

---------------------------------
# tracing-method/transformer.yaml
VARIABLE_F: "another string"
VARIABLE_C: 15
---------------------------------
# labconfig/many_tests.yaml
TESTS_AVAILABLE = 0.7

---------------------------------
# labconfig/no_tests.yaml
TESTS_AVAILABLE = 0.0

---------------------------------
```

Running `python run.py tracing-method=transformer labconfig=no_tests` will parse the configurations into this object:

```python
conf = {
    "TESTS_AVAILABLE": 0.0,
    "VARIABLE_A": 20,
    "VARIABLE_B": 100,
    "VARIABLE_C": 15,
    "VARIABLE_D": 4,
    "VARIABLE_E": 30,
    "VARIABLE_F": "another string"
}
```

### Command-line

***Lastly***, Hydra can overwrite any field from the command-line. Keys of nested dicts within a configuration are accessed through dots as`foo.bar=foobar`.

Running `python run.py method=transformer labconfig=no_tests VARIABLE_C=0` will parse the configurations into this object:

```python
conf = {
    "TESTS_AVAILABLE": 0.0,
    "VARIABLE_A": 20,
    "VARIABLE_B": 100,
    "VARIABLE_C": 0,
    "VARIABLE_D": 4,
    "VARIABLE_E": 30,
    "VARIABLE_F": "another string"
}
```

### COVI Files

As explained, COVI has a `config.yaml` which loads 3 defaults, in this order (subsequent ones overwriting the previous ones on the intersection of their keys):

1. `constants.yaml` contains epidemiological constants used in the simulator's logic
2. `core.yaml` contains the default parameters of the simulator. Most of them will not change through-out experiments, though one can overwrite them in other configs
3. `base_method.yaml` contains the default fields specific to tracing methods,
3. `tracing-method/` contains tracing-method specific files. In general, don't change them (this will chang the config for everyone)
4. `release/` contains files which **should never be changed** as they correspond to specific releases (such as plots, publications etc.)

```
---------------------------- NOTE ---------------------------------
| All default configurations share the same namespace and will    |
| therefore overwrite each other, according to the order in which |
| they are loaded. A simple way to protect variables from this is |
| to put them in a specific namespace, e.g.                       |
|                                                                 |
|  unique_field:                                                  |
|    key: value                                                   |
-------------------------------------------------------------------
```


```
---------------------------- WARNING ------------------------------
| YAML files may use the `.yml` or `.yaml` extensions. In this    |
| project, we *systematically*  use **`.yaml`**                   |
-------------------------------------------------------------------
```

You will notice that `hydra-configs` also holds a `search` folder which has not been mentioned yet. Its purpose is to implement a hyper-parameter random search. Ideas are similar to `simulation` which is described in this tutorial. It holds configurations for `random_search.py`.

### Personal setup

If you want to run experiments for a particular set of plots, parameter tuning or exploration, you need not change the values in the aforementioned files/folder. **Rather**, create your directory in `hydra-configs/simulation` (which will not be tracked unless you change `.gitignore`) and put your overwriting configs in there.

For instance:

```yaml
# martin/find_this_param.yaml
VARIABLE_X = 1213

---------------------------------
# martin/find_other.yaml
VARIABLE_X = 9999
```

Then `python run.py tracing-method=transformer martin=find_other`

```
----------------------------- NOTE --------------------------------
| `config.yaml`, `constants.yaml`, `core.yaml` and                |
| `tracing-method/` are the only tracked files/folder by `git`    |
-------------------------------------------------------------------
```

## Transitioning: What's new

Python configuration files are gone, they have been changed to YAMLs:

* `config.py` -> `core.yaml`
* `constants.py` -> `constants.yaml`

Tracing methods YAMLs are now unified with common fields in `base_method.yaml` and method-specific values in `tracing-method/*.yaml`

`run.py` has been modified to handle Hydra's command-line capabilities: it boils down to a single `@hydra.main(config_path="some_math")` decorator around `main()`.

```
----------------------------- WARNING -----------------------------
| Hydra's arguments don't use `--`:                               |
|                                                                 |
|  * DON'T: $ python run.py --tune                                |
|  * DO   : $ python run.py tune=true                             |
-------------------------------------------------------------------
```

The configuration that is generated by Hydra (after `config.yaml`, `defaults` and command-line arguments) is first parsed and transformed into a Python `dict` object in **`utils.py/parse_configuration`**. Current behavior is:

* turn string keys in `APP_USERS_FRACTION_BY_AGE` and `HUMAN_DISTRIBUTION` into `tuple` as expected by the code: `"1-15"` -> `(1, 15)`
* parse `start_time` from a string to a `datetime`



### Testing

`tests/utils.py` contains a helper function to gather Hydra configs and overwrite whatever variables with a test-specific YAML:

```python
from pathlib import Path
from tests.utils import get_test_conf


test_config_name = "test_variables.yaml"
conf = get_test_conf(test_config_name)
```

`get_test_conf` will load `hydra-configs/simulation/config.yaml`'s `defaults` and overwrite the resulting configuration with whatever's in `test_configs/test_variables.yaml`. Test configurations **must** be in `test_configs/`.

The whole Hydra setup is tested in `tests/test_hydra.py`. Contact Victor if you want to add / change tests.

## Advanced

* Hydra uses YAML-based `omegaconf` configuration files: https://omegaconf.readthedocs.io/en/latest/usage.html#access-and-manipulation. They are parsed back to a native Python `dict` using `OmegaConf.to_container(conf)` in `utils.py/parse_configuration`


* Use `-c job` or `--cfg job` to print the configuration you're running



* You can enable tab completion in your shell: https://hydra.cc/docs/tutorial/tab_completion


* You can also run a multi-run, *i.e.* a grid search: https://hydra.cc/docs/tutorial/multi-run

* You can easily handle logging for regular prints and **debug**: https://hydra.cc/docs/tutorial/logging

### Variable interpolation

Basically, access another variable in the file, JavaScript-style

```yaml
server:
  host: localhost
  port: 80

client:
  url: http://${server.host}:${server.port}/
  server_port: ${server.port}

user:
  name: ${env:USER}
  home: /home/${env:USER}
  password: ${env:DB_PASSWORD,12345} # default value is 12345
```

Interpolations are customizable:

```python
OmegaConf.register_resolver("concat", lambda x,y: x+y)
OmegaConf.register_resolver("plus_10", lambda x: int(x) + 10)
cong = OmegaConf.create({
    'key': '${plus_10:990}' # ->  1000
    'key2': '${concat:Hello,World}', # -> HelloWorld
    'key_trimmed': '${concat:Hello , World}', # -> HelloWorld
    'escape_whitespace': '${concat:Hello,\ World}', # -> Hello World
})
```

```
--------------------------- WARNING -------------------------------
| If you want to register a resolver, this should be done before  |
| the call to @hydra.main(), in the same file (e.g. run.py)       |
-------------------------------------------------------------------
```

```
----------------------------- NOTE --------------------------------
| Use resolvers to enable manipulations of basic data in YAML     |
| files, like multiplications, summations, casting and whatnot    |
-------------------------------------------------------------------
```