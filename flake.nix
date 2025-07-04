{
  description = "UV Dev Shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      uv2nix,
      pyproject-nix,
      pyproject-build-systems,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        inherit (nixpkgs) lib;

        # Load a uv workspace from a workspace root.
        # Uv2nix treats all uv projects as workspace projects.
        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

        # Create package overlay from workspace.
        overlay = workspace.mkPyprojectOverlay {
          # Prefer prebuilt binary wheels as a package source.
          # Sdists are less likely to "just work" because of the metadata missing from uv.lock.
          # Binary wheels are more likely to, but may still require overrides for library dependencies.
          sourcePreference = "wheel"; # or sourcePreference = "sdist";
          # Optionally customise PEP 508 environment
          # environ = {
          #   platform_release = "5.10.65";
          # };
        };
        # Extend generated overlay with build fixups
        #
        # Uv2nix can only work with what it has, and uv.lock is missing essential metadata to perform some builds.
        # This is an additional overlay implementing build fixups.
        # See:
        # - https://pyproject-nix.github.io/uv2nix/FAQ.html
        pyprojectOverrides = _final: _prev: {
          # Implement build fixups here.
          # Note that uv2nix is _not_ using Nixpkgs buildPythonPackage.
          # It's using https://pyproject-nix.github.io/pyproject.nix/build.html
        };

        # This example is only using x86_64-linux
        pkgs = nixpkgs.legacyPackages.${system};

        # Use Python 3.13 from nixpkgs
        python = pkgs.python313;

        pythonPackages = pkgs.python313Packages;

        jupyterlabQuarto =
          let
            pname = "jupyterlab_quarto";
            version = "0.3.5";
          in
          pythonPackages.buildPythonPackage {
            inherit pname version;
            src = pkgs.fetchPypi {
              inherit pname version;
              sha256 = "sha256-U1+ep+HmfrwOfSsUsFEmOZVRiB9loOzV0AuEN7ZrqXQ=";
            };
            doCheck = true;
            pyproject = true;
            propagatedBuildInputs = [
              pythonPackages.hatchling
              pythonPackages.hatch-jupyter-builder
              pythonPackages.hatch-nodejs-version
              pythonPackages.jupyterlab
            ];

          };

        # Construct package set
        pythonSet =
          # Use base package set from pyproject.nix builders
          (pkgs.callPackage pyproject-nix.build.packages {
            inherit python;
          }).overrideScope
            (
              lib.composeManyExtensions [
                pyproject-build-systems.overlays.default
                overlay
                pyprojectOverrides
              ]
            );
        inherit (pkgs.callPackages pyproject-nix.build.util { }) mkApplication;
      in
      {

        # This example provides two different modes of development:
        # - Impurely using uv to manage virtual environments
        # - Pure development using uv2nix to manage virtual environments
        devShells = {
          # It is of course perfectly OK to keep using an impure virtualenv workflow and only use uv2nix to build packages.
          # This devShell simply adds Python and undoes the dependency leakage done by Nixpkgs Python infrastructure.
          default = pkgs.mkShell {
            # buildInputs = [
            #   pkgs.python313Packages.venvShellHook
            #   pkgs.autoPatchelfHook
            # ];
            # venvDir = "./.venv";
            # postVenvCreation = ''
            #   pip install -U jupyter
            #   autoPatchelf ./.venv
            # '';
            # postShellHook = ''
            #   unset SOURCE_DATE_EPOCH
            #   jupyter notebook
            # '';

            packages = [
              # python
              pkgs.quarto
              pkgs.uv
              jupyterlabQuarto
              (python.withPackages (
                p: with p; [
                  p.ruff

                  # packages for managing notebook experience
                  p.jupyter
                  p.ipython
                  p.ipykernel
                  p.nbdev
                  p.setuptools

                  # Packages for molten
                  p.jupyter_client
                  p.cairosvg
                  p.pnglatex
                  # p.plotly
                  p.kaleido
                  p.pyperclip
                  p.pynvim
                  p.nbformat
                  p.pillow
                  p.requests
                  p.websocket-client
                  p.jupytext
                ]
              ))

              # Ruby and jekyll are needed for the documentation build
              pkgs.ruby
              pkgs.rubyPackages.jekyll
            ];
            env =
              {
                # Prevent uv from managing Python downloads
                UV_PYTHON_DOWNLOADS = "never";
                # Force uv to use nixpkgs Python interpreter
                UV_PYTHON = python.interpreter;
              }
              // lib.optionalAttrs pkgs.stdenv.isLinux {
                # Python libraries often load native shared objects using dlopen(3).
                # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of libraries without using RPATH for lookup.
                # TODO: Replace with your Python's manylinux package.
                LD_LIBRARY_PATH = lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1;
              };
            shellHook = ''
              # ipython kernel install --name "python3" --user



                            unset PYTHONPATH
            '';
          };
          # This devShell uses uv2nix to construct a virtual environment purely from Nix, using the same dependency specification as the application.
          # The notable difference is that we also apply another overlay here enabling editable mode ( https://setuptools.pypa.io/en/latest/userguide/development_mode.html ).
          #
          # This means that any changes done to your local files do not require a rebuild.
          #
          # Note: Editable package support is still unstable and subject to change.
          uv2nix =
            let
              # Create an overlay enabling editable mode for all local dependencies.
              editableOverlay = workspace.mkEditablePyprojectOverlay {
                # Use environment variable
                root = "$REPO_ROOT";
                # Optional: Only enable editable for these packages
                # members = [ "hello-world" ];
              };

              # Override previous set with our overrideable overlay.
              editablePythonSet = pythonSet.overrideScope (
                lib.composeManyExtensions [
                  editableOverlay

                  # Apply fixups for building an editable package of your workspace packages
                  (final: prev: {
                    hello-world = prev.hello-world.overrideAttrs (old: {
                      # It's a good idea to filter the sources going into an editable build
                      # so the editable package doesn't have to be rebuilt on every change.
                      # src = lib.fileset.toSource {
                      #   root = old.src;
                      #   fileset = lib.fileset.unions [
                      #     (old.src + "/pyproject.toml")
                      #     (old.src + "/README.md")
                      #     (old.src + "/src/__init__.py")
                      #   ];
                      # };

                      # Hatchling (our build system) has a dependency on the `editables` package when building editables.
                      #
                      # In normal Python flows this dependency is dynamically handled, and doesn't need to be explicitly declared.
                      # This behaviour is documented in PEP-660.
                      #
                      # With Nix the dependency needs to be explicitly declared.
                      nativeBuildInputs =
                        old.nativeBuildInputs
                        ++ final.resolveBuildSystem {
                          editables = [ ];
                        };
                    });

                  })
                ]
              );

              # Build virtual environment, with local packages being editable.
              #
              # Enable all optional dependencies for development.
              virtualenv = editablePythonSet.mkVirtualEnv "tammmm-dev-env" workspace.deps.all;

            in
            pkgs.mkShell {
              packages = [
                virtualenv
                pkgs.uv
                pkgs.ruff
                pkgs.ruff-lsp
              ];

              env = {
                # Don't create venv using uv
                UV_NO_SYNC = "1";

                # Force uv to use Python interpreter from venv
                UV_PYTHON = "${virtualenv}/bin/python";

                # Prevent uv from downloading managed Python's
                UV_PYTHON_DOWNLOADS = "never";
              };

              shellHook = ''
                # Undo dependency propagation by nixpkgs.
                unset PYTHONPATH

                # Get repository root using git. This is expanded at runtime by the editable `.pth` machinery.
                export REPO_ROOT=$(git rev-parse --show-toplevel)
              '';
            };
        };
      }
    );
}
