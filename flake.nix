{
  description = "Pre-randomization for the NBack experiment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f system);
    in {
      apps = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          prerandomize = pkgs.writeShellApplication {
            name = "prerandomize";
            runtimeInputs = [ pkgs.python3 ];
            text = ''
              exec python3 ${./prerandomize.py} "$@"
            '';
          };
        in {
          prerandomize = {
            type = "app";
            program = "${prerandomize}/bin/prerandomize";
            meta.description = "Generate the N-back letter lists into ./lists/";
          };
          default = self.apps.${system}.prerandomize;
        });

      devShells = forAllSystems (system:
        let pkgs = nixpkgs.legacyPackages.${system}; in {
          default = pkgs.mkShell {
            packages = [ pkgs.python3 ];
          };
        });
    };
}
