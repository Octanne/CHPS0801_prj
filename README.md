# CHPS0801 | Lisseurs pour le Débruitage d’Images

## Objectif
L'objectif de ce projet est de réaliser un programme à la fois avec Kokkos et OpenMP en implémentant un filtrage de Jacobi et Gauss Seidel et en testant ces approches en version séquentielle et multi-threadée (cpu & gpu). Pour pouvoir comparer ces deux approches de parallélisation.

## Kokkos
### Compilation
```bash
make
```

### Exécution
Pour exécuter le programme :
```bash
cd kokkos
./kokkos_jacobi.host --input=img/lena.jpg --filter=jacobi --iteration=100 --cpu=8
```
Pour avoir la liste des filtres :
```bash
./kokkos_jacobi.host --help
```

### Structure du projet Kokkos
- img/ : contient les images d'entrée.
- res/ : contient les images résultantes.
- Makefile : fichier de configuration pour la compilation.

