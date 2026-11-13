{
  description = "Podcast transcription (MLX: VibeVoice-ASR + Parakeet) on aarch64-darwin";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
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

  outputs = { nixpkgs, pyproject-nix, uv2nix, pyproject-build-systems, ... }:
    let
      system = "aarch64-darwin";
      pkgs = nixpkgs.legacyPackages.${system};
      lib = nixpkgs.lib;

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
      overlay = workspace.mkPyprojectOverlay {
        # Use the exact wheels pinned in uv.lock (incl. mlx + mlx-metal 0.32.4).
        sourcePreference = "wheel";
      };

      python = pkgs.python313;
      pythonSet =
        (pkgs.callPackage pyproject-nix.build.packages { inherit python; }).overrideScope
          (lib.composeManyExtensions [
            pyproject-build-systems.overlays.wheel
            overlay
            # The mlx wheel finds libmlx.dylib / mlx.metallib via @rpath relative to its own
            # (store-path-resolved) mlx/ dir, but the PyPI split puts those in the separate
            # mlx-metal wheel. Merge mlx-metal's files into the mlx output so rpath resolves.
            (final: prev: {
              mlx = prev.mlx.overrideAttrs (old: {
                postInstall = (old.postInstall or "") + ''
                  cp -R ${final.mlx-metal}/${python.sitePackages}/. $out/${python.sitePackages}/
                '';
              });
            })
          ]);

      pythonEnv = pythonSet.mkVirtualEnv "transcribe-episode-env" workspace.deps.default;

      # dev shell additionally gets the dev dependency group (pytest, httpx)
      devEnv = pythonSet.mkVirtualEnv "transcribe-episode-dev-env" workspace.deps.all;

      ffmpeg = pkgs.ffmpeg-headless;

      # CLI script + importable `transcriber` package (the worker subprocess runs `python -m transcriber.worker`).
      src = lib.fileset.toSource {
        root = ./.;
        fileset = lib.fileset.unions [ ./transcribe_episode.py ./transcriber ];
      };

      transcribe-episode = pkgs.writeShellApplication {
        name = "transcribe-episode";
        runtimeInputs = [ ffmpeg ];
        text = ''
          export HF_HOME="''${HF_HOME:-$HOME/.cache/huggingface}"
          exec ${pythonEnv}/bin/python ${src}/transcribe_episode.py "$@"
        '';
      };

      # HTTP job service (STATE_DIR, HOST, PORT, API_TOKEN from the environment).
      transcribe-server = pkgs.writeShellApplication {
        name = "transcribe-server";
        runtimeInputs = [ ffmpeg ];
        text = ''
          export HF_HOME="''${HF_HOME:-$HOME/.cache/huggingface}"
          export PYTHONPATH="${src}"
          exec ${pythonEnv}/bin/python -m transcriber.server "$@"
        '';
      };

      # Download both models into HF_HOME (resumable; hf_hub skips files already cached). Model IDs come from the pipeline.
      fetch-models = pkgs.writeShellApplication {
        name = "fetch-models";
        text = ''
          export HF_HOME="''${HF_HOME:-$HOME/.cache/huggingface}"
          export PYTHONPATH="${src}"
          exec ${pythonEnv}/bin/python -m transcriber.fetch_models "$@"
        '';
      };
    in
    {
      packages.${system} = {
        default = transcribe-episode;
        server = transcribe-server;
        inherit transcribe-episode transcribe-server fetch-models;
      };

      apps.${system} = {
        default = {
          type = "app";
          program = "${transcribe-episode}/bin/transcribe-episode";
        };
        server = {
          type = "app";
          program = "${transcribe-server}/bin/transcribe-server";
        };
        fetch-models = {
          type = "app";
          program = "${fetch-models}/bin/fetch-models";
        };
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [ devEnv ffmpeg pkgs.uv ];
        env = {
          # Nix provides the interpreter/deps; stop uv from creating or syncing a venv.
          UV_NO_SYNC = "1";
          UV_PYTHON_DOWNLOADS = "never";
          UV_PYTHON = "${devEnv}/bin/python";
        };
        shellHook = ''
          export HF_HOME="''${HF_HOME:-$HOME/.cache/huggingface}"
          unset PYTHONPATH
          export PYTHONPATH="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"  # make `transcriber` importable
        '';
      };
    };
}
