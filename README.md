# Beam mooseapp-code-plugin

This MooseApp code plugin contains a set of MooseObjects providing the element formulation for EDGE3 beam elements
and corresponding constitutive models.


## Setup for Users

If you want to use this code-plugin in your MooseApp, follow these steps:

- Clone this repository as git-submodule in a sub-folder in `./contrib/` of your MooseApp.
    Assuming you are in the root directory of your MooseApp this could be done with the following command:
  ```shell
  git submodule add https://github.com/Kavan-Khaledi/moose-codeplugin-beam contrib/beam
  ```
- In your MooseApp `Makefile`, insert the following line **directly above** the line `include $(FRAMEWORK_DIR)/app.mk` (if you have multiple code plugins, you need this line only once):
  ```MAKEFILE
  include $(wildcard $(CURDIR)/contrib/*/codeplugin.mk)
  ```

- Compile your MooseApp.
